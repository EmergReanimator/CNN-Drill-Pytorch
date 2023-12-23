import torch
from torchvision import transforms

import torch.optim as optim

from model import Net1, modelsummary
from utils import build_mnist, Trainer, plot_sampledata, evaluate_model

import os

batch_size = 128
num_epochs = 15
mean = 0.1307
std = 0.3081


# CUDA?
torch.manual_seed(1)
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
input('Hit any key to continue')


# Train data transformations
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,)),
    ]
)

# Test data transformations
test_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))]
)

kwargs = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 2,
    "pin_memory": True,
}

train_data, train_loader = build_mnist(set="train", transforms=train_transforms, **kwargs)
test_data, test_loader = build_mnist(set="test", transforms=test_transforms, **kwargs)

plot_sampledata(train_loader)
input('Hit any key to continue')


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net1().to(device)
modelsummary(model, device)
input('Hit any key to continue')



model = Net1().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # large learning rate

trainer = Trainer(model, device, optimizer)
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}")
    trainer.train(train_loader)
    trainer.test(test_loader)


trainer.plot_history()


evaluate_model(trainer.model, test_loader, device)

# torch.save(trainer.model.state_dict(), './model.pt')

model_fn = f'./models/{device}-model.pth'
try:
    os.mkdir(model_fn)
except FileExistsError:
    pass

torch.save(trainer.model, model_fn)

