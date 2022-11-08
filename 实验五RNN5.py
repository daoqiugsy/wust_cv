import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import sys

device = torch.device('cuda')


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.Out2Class = nn.Linear(128 * 28, 10)

    def forward(self, input):
        output, hn = self.rnn(input, None)
        hidden2one_res = []
        for i in range(28):
            hidden2one_res.append(output[:, i, :])
        hidden2one_res = torch.cat(hidden2one_res, dim=1)
        # print(hidden2one_res.shape)
        res = self.Out2Class(hidden2one_res)
        return res


model = RNN()
model = model.to(device)
print(model)

model = model.train()

img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
dataset_train = datasets.MNIST(root='./data', transform=img_transform, train=True, download=True)
dataset_test = datasets.MNIST(root='./data', transform=img_transform, train=False, download=True)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)


# images,label = next(iter(train_loader))
# print(images.shape)
# print(label.shape)
# images_example = torchvision.utils.make_grid(images)
# images_example = images_example.numpy().transpose(1,2,0)
# mean = [0.5,0.5,0.5]
# std = [0.5,0.5,0.5]
# images_example = images_example*std + mean
# plt.imshow(images_example)
# plt.show()

def Get_ACC():
    correct = 0
    total_num = len(dataset_test)
    for item in test_loader:
        batch_imgs, batch_labels = item
        batch_imgs = batch_imgs.squeeze(1)
        batch_imgs = Variable(batch_imgs)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_imgs)
        _, pred = torch.max(out.data, 1)
        correct += torch.sum(pred == batch_labels)
        # print(pred)
        # print(batch_labels)
    correct = correct.data.item()
    acc = correct / total_num
    print('correct={},Test ACC:{:.5}'.format(correct, acc))


optimizer = torch.optim.Adam(model.parameters())
loss_f = nn.CrossEntropyLoss()

Get_ACC()
for epoch in range(5):
    print('epoch:{}'.format(epoch))
    cnt = 0
    for item in train_loader:
        batch_imgs, batch_labels = item
        batch_imgs = batch_imgs.squeeze(1)
        # print(batch_imgs.shape)
        batch_imgs, batch_labels = Variable(batch_imgs), Variable(batch_labels)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_imgs)
        # print(out.shape)
        loss = loss_f(out, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (cnt % 100 == 0):
            print_loss = loss.data.item()
            print('epoch:{},cnt:{},loss:{}'.format(epoch, cnt, print_loss))
        cnt += 1
    Get_ACC()

torch.save(model, 'model')


