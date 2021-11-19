import numpy as np
import random
import time
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from pt_distr_env import setup_distr_env


device = 0
num_epochs = 5
batch_size = 64
num_iters = 25
model_name = 'resnet50'

model = getattr(models, model_name)()

optimizer = optim.SGD(model.parameters(), lr=0.01)


class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.randn(3, 224, 224)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return batch_size * num_iters


train_loader = DataLoader(SyntheticDataset(),
                          batch_size=batch_size)

model.to(device)

setup_distr_env()
dist.init_process_group(backend="nccl")
ddp_model = DDP(model, device_ids=[0])


def benchmark_step(model, imgs, labels):
    optimizer.zero_grad()
    output = model(imgs.to(device))
    loss = F.cross_entropy(output, labels.to(device))
    loss.backward()
    optimizer.step()


imgs_sec = []
for epoch in range(num_epochs):
    t0 = time.time()
    for step, (imgs, labels) in enumerate(train_loader):
        benchmark_step(ddp_model, imgs, labels)

    dt = time.time() - t0
    imgs_sec.append(batch_size * num_iters / dt)

    rank = dist.get_rank()
    print(f' * Rank {rank} - Epoch {epoch:2d}: '
          f'{imgs_sec[epoch]:.2f} images/sec per GPU')

imgs_sec_total = np.mean(imgs_sec) * dist.get_world_size()
if dist.get_rank() == 0:
    print(f' * Total average: {imgs_sec_total:.2f} images/sec')
