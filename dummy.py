import torchvision
import torch

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
loader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=1
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

h, w = 0, 0
for batch_idx, (inputs, targets) in enumerate(loader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        h, w = inputs.size(2), inputs.size(3)
        print(inputs.min(), inputs.max())
        chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
mean = chsum/len(trainset)/h/w
print('mean: %s' % mean.view(-1))

chsum = None
for batch_idx, (inputs, targets) in enumerate(loader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
std = torch.sqrt(chsum/(len(trainset) * h * w - 1))
print('std: %s' % std.view(-1))