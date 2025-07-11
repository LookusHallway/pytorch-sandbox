import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001

# dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

examples = iter(train_loader)
samples, labels = examples.__next__()
print(samples.shape, labels.shape)

# Building model

class ConvNet(nn.Module):
    def __init__(self,):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 100) # 16 channels of 5x5 images flattened
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 10)
       
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# training model

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 4, 3, 32, 32
        # input layer: 3 input channels, 6 output chanels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 2000 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i}/{n_total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    num_correct = 0
    num_samples = 0
    num_class_correct = [0] * len(classes)
    num_class_samples = [0] * len(classes)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predictions = torch.max(outputs, 1) # predictions is the index, not the value
        num_samples += labels.size(0)
        num_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                num_class_correct[label] += 1
            num_class_samples[label] += 1

    acc = 100.0 * (num_correct / num_samples)
    print(f'Network accuracy: {acc:.4f}')

    for i in range(len(classes)):
        acc = 100.0 * (num_class_correct[i] / num_class_samples[i])
        print(f'Class accuracy of {classes[i]}: {acc:.4f}')