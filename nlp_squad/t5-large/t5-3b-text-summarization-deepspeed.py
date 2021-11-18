import argparse
import deepspeed
import datasets
import torch
from datasets import load_dataset  # , load_metric
from transformers import (AdamW, AutoModelForSeq2SeqLM,
                          AutoTokenizer, DataCollatorForSeq2Seq)
from torch.utils.data import DataLoader


# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

model_name = 't5-3b'
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=True,
    cache_dir=f'./cache/{model_name}_tokenizer'
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    # config=config,
    cache_dir=f'./cache/{model_name}_model'
)

raw_datasets = load_dataset('xsum')

max_source_length = 1024
max_target_length = 128
ignore_pad_token_for_loss = True
padding = False            # else 'max_length'
label_pad_token_id = -100  # else tokenizer.pad_token_id
per_device_train_batch_size = 4
per_device_eval_batch_size = 4

def preprocess_function(examples, text_column='document',
                        summary_column='summary', prefix='summarize: '):
    inputs = examples[text_column]
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id
)

# column_names = raw_datasets["train"].column_names

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    # load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator,
    batch_size=per_device_train_batch_size
)

eval_dataloader = DataLoader(
    eval_dataset, collate_fn=data_collator,
    batch_size=per_device_eval_batch_size
)

weight_decay = 0.0
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

# optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=optimizer_grouped_parameters,
    training_data=train_dataloader
)

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


print_peak_memory("Max memory allocated after creating DDP", 0)

device = 0
model.to(device)
model.train();
for epoch in range(1):
    for step, batch in enumerate(train_dataloader):
        if step > 20:
            break

        optimizer.zero_grad()
        outputs = model(**batch.to(device))
        loss = outputs.loss
        loss.backward() 
        optimizer.step()
