import torch
import numpy
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets, transforms  # torchvision包的主要功能是实现数据的处理、导入和预览等
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
data_train = datasets.MNIST(
    transform=transform,
    root="./data/",
    train=True,
    download=True
)
data_test = datasets.MNIST(
    root="./data/",
    transform=transform,
    train=True,
    download=False
)

from torch.utils.data import random_split

data_train, _ = random_split(
    dataset=data_train,
    lengths=[1000, 59000],
    generator=torch.Generator().manual_seed(0)
)
data_test, _ = random_split(
    dataset=data_test,
    lengths=[1000, 59000],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=4,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=4,
                                               shuffle=True)
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )

        self.dense = torch.nn.Sequential(
            # torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.Linear(7 * 7 * 128, 512),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Dropout(p=0.8),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)  # 卷积处理
        # x = x.view(-1, 14*14*128)  # 对参数实行扁平化处理
        x = x.view(-1, 7 * 7 * 128)  # 对参数实行扁平化处理
        x = self.dense(x)
        return x


model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs_n = 5
for epoch in range(epochs_n):
    running_loss = 0.0
    running_correct = 0
    print("Epoch{}/{}".format(epoch, epochs_n))
    print("-" * 10)

    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
        print("Loss is:{:.4f},Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                         100 * running_correct / len(
                                                                                             data_train),
                                                                                         100 * testing_correct / len(
                                                                                             data_test)))

X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_, pred = torch.max(pred, 1)

print("Predict Label is:", [i for i in pred.data])
print("Real Label is:", [i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1, 2, 0)

std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]

img = img * std + mean
plt.imshow(img)
plt.show()
epochs_n = 5
for epoch in range(epochs_n):
    running_loss = 0.0
    running_correct = 0
    print("Epoch{}/{}".format(epoch, epochs_n))
    print("-" * 10)

    iter = 0
    for data in data_loader_train:
        iter += 1

        print(iter)
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
        print("Loss is:{:.4f},Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                         100 * running_correct / len(
                                                                                             data_train),
                                                                                         100 * testing_correct / len(
                                                                                             data_test)))


