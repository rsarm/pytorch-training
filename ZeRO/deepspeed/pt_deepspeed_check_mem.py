import argparse
import deepspeed
import json
import numpy as np
import torch
import torch.nn as nn
import time


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# FIXME: get the batch_size from config
with open(args.deepspeed_config) as f:
    config = json.load(f)

config = deepspeed.runtime.config.DeepSpeedConfig(config)
batch_size = config.train_batch_size

local_rank = 0
features = 10000   # 2,000,200,000 parameters
# features = 15000   # 4,500,300,000 parameters
# features = 20000   # 8,000,400,000 parameters
num_layers = 20


# Set up fixed fake data
class FakeData(torch.utils.data.Dataset):
    def __len__(self):
        return batch_size * args.num_iters

    def __getitem__(self, idx):
        return (torch.randn(20, features),
                torch.randn(20, features))


trainset = FakeData()
trainloader = torch.utils.data.DataLoader(trainset,
                                          num_workers=1)

# create local model
model = nn.Sequential(*[nn.Linear(features, features)
                        for _ in range(num_layers)])
print_peak_memory("Max memory allocated after creating model", local_rank)

parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in parameters])
print(f'\n {num_params:,} parameters\n')
parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=trainset
)
print_peak_memory("Max memory allocated after deepspeed.initialize", 0)

# not using `loss_fn = nn.MSELoss().cuda()` as mse_loss layers autocast
# to float32 https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

# training
for epoch in range(1):
    t0 = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs = data[0].to(model_engine.device)
        labels = data[1].to(model_engine.device)
        if model_engine.fp16_enabled():
            inputs = inputs.half()

        # forward + backward + optimize
        outputs = model_engine(inputs)
        loss = torch.sum(torch.abs(outputs - labels))
        model_engine.backward(loss)
        model_engine.step()
        print_peak_memory("Max memory allocated after optimizer step", 0)

    # timing
    delta = time.time() - t0
    print('Performance:', batch_size * args.num_iters / delta, 'samples/s')

print('Finished Training')
