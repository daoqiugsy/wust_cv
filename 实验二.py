import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import PIL
import numpy as np
import pandas as pd
import imageio
from PIL import Image, ImageTk # 导入图像处理函数库
import tkinter as tk
from tkinter import constants, ttk
from tkinter import filedialog
import threading

#Basic Params-----------------------------

batch_size_train = 64
batch_size_test = 1000
gpu = torch.cuda.is_available()
momentum = 0.5
# Load Data-------------------------------
train_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             (0.1307,), (0.3081,))
                                                     ])),
                          batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.1307,), (0.3081,))
                                                    ])),
                         batch_size=batch_size_test, shuffle=True)
train_data_size = len(train_loader)
test_data_size = len(test_loader)
# Define Model----------------------------
def creat_Lenet():
    global epochs
    global learning_rate
    epochs = int(var_epochs.get())
    learning_rate = float(var_lrate.get())
    text.insert(tk.END, '学习率：' + var_lrate.get() + '，训练世代：' + var_epochs.get() + '\n')
    text.insert(tk.END, '可以开始训练了！\n')
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2,  stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 =nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
        self.flat=nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
class Thread1(threading.Thread):
    def __init__(self):
        super(Thread1, self).__init__()
    def run(self):

    #print("训练第",epoch,"个代")
        global total_train_step
        total_train_step = 0

        for e in range(epochs):
            for data in train_loader:
                imgs, targets = data
                if gpu:
                    imgs, targets = imgs.cuda(), targets.cuda()
                optimizer.zero_grad()

                outputs = net(imgs)

                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                if total_train_step % 200 == 0:
                    text.insert(tk.END,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                        e, total_train_step%1000, train_data_size,
                        100. * total_train_step / train_data_size%1000, loss.item()))
                    text.update()



                writer.add_scalar('loss', loss.item(), total_train_step)
                total_train_step += 1
        text.insert(tk.END,'训练完成！')
        text.update()
class Thread2(threading.Thread):
    def __init__(self):
        super(Thread2, self).__init__()
    def run(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data

                if gpu:
                    imgs, targets = imgs.cuda(), targets.cuda()
                    print(imgs)
                    print(imgs.shape)
                outputs = net(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        text.insert(tk.END, 'Test Accuracy: {}/{} ({:.02f}%)\n'.format(correct, total, 100. * correct / total))
        text.insert(tk.END,"测试结束")
        text.update()
        # save model
        torch.save(net, 'model/mnist_model.pth')
        print('Saved model')

        return correct / total




if gpu:
    net = LeNet1().cuda()
else:
    net = LeNet1()
# Define Loss and Optimizer----------------

if gpu:
    loss_fn = nn.CrossEntropyLoss().cuda()
else:
    loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.01)

# Define Tensorboard-------------------

writer = SummaryWriter(log_dir='logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')))
# Train---------------------------------

total_train_step = 0

window = tk.Tk()
window.title('GSY的卷积神经网络识别MNIST数据集')
window.geometry('600x500')
global img_png  # 定义全局变量 图像的
var = tk.StringVar()  # 这时文字变量储存器
text = tk.Text(window,width=20,height=17)
text.pack(fill=tk.X,side=tk.BOTTOM)
text.insert(tk.END, '请输入相关数据，构建一个网络\n')
def train():

    #print("训练第",epoch,"个代")
    global total_train_step
    total_train_step = 0

    for e in range(epochs):
        for data in train_loader:
            imgs, targets = data
            if gpu:
                imgs, targets = imgs.cuda(), targets.cuda()
            optimizer.zero_grad()

            outputs = net(imgs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            if total_train_step % 200 == 0:
                text.insert(tk.END, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    e, total_train_step % 1000, train_data_size,
                       100. * total_train_step / train_data_size % 1000, loss.item()))
                text.update()

            writer.add_scalar('loss', loss.item(), total_train_step)
            total_train_step += 1
    text.insert(tk.END, '训练完成！')
    text.update()


# Test---------------------------------

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data

            if gpu:
                imgs, targets = imgs.cuda(), targets.cuda()
                print(imgs)
                print(imgs.shape)
            outputs = net(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    text.insert(tk.END, 'Test Accuracy: {}/{} ({:.02f}%)\n'.format(correct, total, 100. * correct / total))
    text.insert(tk.END, "测试结束")
    text.update()
    # save model
    torch.save(net, 'model/mnist_model.pth')
    print('Saved model')

    return correct / total
    # save model



    return correct / total


# Run----------------------------------

def Open_Img():
    global img_png
    global path
    global label_Img
    OpenFile = tk.Tk()  # 创建新窗口
    OpenFile.withdraw()
    file_path = filedialog.askopenfilename()
    print("训练已结束，开始测试图片")
    text.insert(tk.END, '开始测试图片\n')
    path = file_path
    Img = Image.open(file_path)
    img_png = ImageTk.PhotoImage(Img)
    label_Img = tk.Label(window, image=img_png)
    Label_Show = tk.Label(window, image=img_png,
                          # 使用 textvariable 替换 text, 因为这个可以变化
                          bg='white', font=('Arial', 12), width=60, height=60)
    Label_Show.place(x=80, y=80)
    image_file_name = path
    img_array = imageio.imread(image_file_name, as_gray=True)
    img_array=torch.tensor(img_array)


    var.set('图像已打开')
    # 自己图片的数据存放在这里

    image_file_name = path
    print("加载中 ... ", image_file_name)
    text.insert(tk.END, '加载中....' + image_file_name + '\n')
    # 用文件名来设置准确值标签
    label = int(image_file_name[34])

    img_array=torch.reshape(img_array,(1,1,28,28))
    img_array=img_array.cuda()
    print(img_array.shape)
    output=net(img_array)
    print(output)
    _, predicted = torch.max(output.data, 1)
    print(predicted)
    text.insert(tk.END, '神经网络认为图中的数字是' + str(int(predicted[0])) + '\n')
    if predicted==label:
        print("恭喜你，匹配成功！")
        text.insert(tk.END, '恭喜你，识别成功了！\n')
    else:
        print("很遗憾，识别失败了")
        text.insert(tk.END, '很遗憾，识别失败了！再试一次吧\n')


    # 将标签值放到数组第一个
def SHOW():
    global img_png
    Label_Show = tk.Label(window, image=img_png,
                          # 使用 textvariable 替换 text, 因为这个可以变化
                          bg='white', font=('Arial', 12), width=60, height=60)
    Label_Show.place(x=80, y=80)


btn_train = tk.Button(window, text='构建网络', width=15, height=2,
                      command=creat_Lenet)
btn_train.pack()
btn_train.place(x=30, y=210)
btn_test = tk.Button(window, text='训练数据集', width=15, height=2,
                     command=Thread1().start)
btn_test.pack()
btn_test.place(x=170,y=210)
btn_Open = tk.Button(window,
                     text='测试数据集',  # 显示在按钮上的文字
                     width=15, height=2,
                     command=Thread2().start)  # 点击按钮式执行的命令
btn_Open.pack()
btn_Open.place(x=310, y=210)
# 按钮位置
# 创建显示图像按钮
btn_Show = tk.Button(window,
                     text='打开测试图片',  # 显示在按钮上的文字
                     width=15, height=2,
                     command=Open_Img)  # 点击按钮式执行的命令

btn_Show.pack()
# 按钮位置
btn_Show.place(x=450, y=210)
rate_lable = tk.Label(window, text='学习率：')
rate_lable.pack()
rate_lable.place(x=300, y=130)
img_frame = tk.LabelFrame(window, text='图像显示', padx=10, pady=10,
                          width=120, height=120)
img_frame.place(x=55, y=50)
var_lrate = tk.StringVar()
var_lrate.set('0.1')
entry_lrate = tk.Entry(window, textvariable=var_lrate, width=10)
entry_lrate.place(x=380, y=130)
epochs_lable = tk.Label(window, text='训练世代：')
epochs_lable.pack()
epochs_lable.place(x=300, y=160)
var_epochs = tk.StringVar()
var_epochs.set('5')
entry_epochs = tk.Entry(window, textvariable=var_epochs, width=10)
entry_epochs.place(x=380, y=160)
window.mainloop()
var_lrate = tk.StringVar()
var_lrate.set('0.1')
entry_lrate = tk.Entry(window, textvariable=var_lrate, width=10)
entry_lrate.place(x=380, y=130)
var_lrate = tk.StringVar()
var_lrate.set('0.1')
entry_lrate = tk.Entry(window, textvariable=var_lrate, width=10)
entry_lrate.place(x=380, y=130)

