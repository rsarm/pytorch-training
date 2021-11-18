import argparse
import deepspeed
import numpy as np
import os
import json
import dataset_utils as du
import eval_utils as eu
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datetime import datetime


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

bert_cache = os.path.join(os.getcwd(), 'cache')

slow_tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    cache_dir=os.path.join(bert_cache, '_bert-base-uncased-tokenizer')
)
save_path = os.path.join(bert_cache, 'bert-base-uncased-tokenizer')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(os.path.join(save_path, 'vocab.txt'),
                                   lowercase=True)

model = BertForQuestionAnswering.from_pretrained(
    "bert-base-uncased",
    cache_dir=os.path.join(bert_cache, 'bert-base-uncased_qa')
)
# model.train();

# parameters = filter(lambda p: p.requires_grad, model.parameters())
# num_params = sum([np.prod(p.size()) for p in parameters])
# print(f'\n {num_params:,} parameters\n')
parameters = filter(lambda p: p.requires_grad, model.parameters())

train_path = os.path.join(bert_cache, 'data', 'train-v1.1.json')
eval_path = os.path.join(bert_cache, 'data', 'dev-v1.1.json')
with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)

batch_size = 8
max_len = 384

# raw_train_data = {
#         'data': raw_train_data['data'][:20],
#         'version': raw_train_data['version']
# }
train_squad_examples = du.create_squad_examples(raw_train_data, max_len, tokenizer)
x_train, y_train = du.create_inputs_targets(train_squad_examples, shuffle=True, seed=42)
print(f"{len(train_squad_examples)} training points created.")

# raw_eval_data = {
#         'data': raw_eval_data['data'][:20],
#         'version': raw_eval_data['version']
# }
eval_squad_examples = du.create_squad_examples(raw_eval_data, max_len, tokenizer)
x_eval, y_eval = du.create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return (torch.tensor(self.x[0][idx]),
                torch.tensor(self.x[1][idx]),
                torch.tensor(self.x[2][idx]),
                torch.tensor(self.y[0][idx]),
                torch.tensor(self.y[1][idx]))

    def __len__(self):
        return len(self.x[0])

train_set = SquadDataset(x_train, y_train)

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_set
)
print_peak_memory("Max memory allocated after creating DDP", 0)


# training
for epoch in range(1):  # loop over the dataset multiple times
    # t0 = time.time()
    for i, batch in enumerate(trainloader, 0):
        outputs = model(input_ids=batch[0].to(model_engine.device),
              token_type_ids=batch[1].to(model_engine.device),
              attention_mask=batch[2].to(model_engine.device),
              start_positions=batch[3].to(model_engine.device),
              end_positions=batch[4].to(model_engine.device)
             )
        # forward + backward + optimize
        loss = outputs[0]
        # print('\n\n', loss, '\n\n')
        model_engine.backward(loss)
        model_engine.step()
        # print_peak_memory("Max memory allocated after optimizer step", 0)

    # timing
    # delta = time.time() - t0
    # print('Performance:', batch_size * args.num_iters / delta, 'samples/s')

print('Finished Training')


if os.environ['SLURM_NODEID'] is '0':
    model_hash = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_path_name = './cache/model_trained_single_node_{model_hash}'
    
    # save model's state_dict
    torch.save(model.state_dict(), model_path_name)
    
    # create the model again since the previous one is on the gpu
    model_cpu = BertForQuestionAnswering.from_pretrained(
        "bert-base-uncased",
        cache_dir=os.path.join(bert_cache, 'bert-base-uncased_qa')
    )
    
    # load the model on cpu
    model_cpu.load_state_dict(
        torch.load(model_path_name,
                   map_location=torch.device('cpu'))
    )
    
    # load the model on gpu
    # model.load_state_dict(torch.load(model_path_name))
    
    model.eval();
    
    samples = np.random.choice(len(x_eval[0]), 50, replace=False)
    
    eu.EvalUtility(
        (x_eval[0][samples], x_eval[1][samples], x_eval[2][samples]),
        model_cpu,
        eval_squad_examples[samples]
    ).results()
