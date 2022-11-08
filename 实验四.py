import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
model_path = 'transferResNet50/model_name.pth'
model_params_path = 'transferResNet50/params_name.pth'

transform = transforms.Compose(
    [transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]
)


data_dir = "H:/DogsVSCats"

data_transform = {
    x:transforms.Compose(
        [
            transforms.Scale([224,224]),    #Scale类将原始图片的大小统一缩放至64×64
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5]
            )
        ]
    )
    for x in ["train","valid"]
}


image_datasets = {
    x:datasets.ImageFolder(
        root=os.path.join(data_dir,x),  #将输入参数中的两个名字拼接成一个完整的文件路径
        transform=data_transform[x]
    )
    for x in ["train","valid"]
}


dataloader = {
    #注意：标签0/1自动根据子目录顺序以及目录名生成
    #如：{'cat': 0, 'dog': 1} #{'狗dog': 0, '猫cat': 1}
    #如：['cat', 'dog']  #['狗dog', '猫cat']
    x:torch.utils.data.DataLoader(
        dataset=image_datasets[x],
        batch_size=16,
        shuffle=True
    )
    for x in ["train","valid"]
}


X_example, y_example = next(iter(dataloader["train"]))
example_classes = image_datasets["train"].classes    #['cat', 'dog']  #['狗dog', '猫cat']
index_classes = image_datasets["train"].class_to_idx #{'cat': 0, 'dog': 1} #{'狗dog': 0, '猫cat': 1}


Use_gpu = torch.cuda.is_available()
model = models.resnet50(pretrained=True)
print(model)
for param in model.parameters():
    param.requires_grad = False

'''重写model的classifier属性，重新设计分类器的结构'''
model.fc = torch.nn.Linear(2048, 2)

print(model)
if Use_gpu:
    model = model.cuda()

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.00001)

has_been_trained = os.path.isfile(model_path)
if has_been_trained:
    epoch_n = 0
else:
    epoch_n = 1
time_open = time.time()
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n - 1))
    print("-" * 10)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)  # model.train(),启用 BatchNormalization 和 Dropout
        else:
            print("Validing...")
            model.train(False)  # model.eval(),不启用 BatchNormalization 和 Dropout

        running_loss = 0.0
        running_corrects = 0
        # cxq = 1
        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            # print("$$$$$$",cxq)
            # cxq+=1
            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)
            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)

            optimizer.zero_grad()

            loss = loss_f(y_pred, y)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)

            if batch % 500 == 0 and phase == "train":
                print("Batch {}, Train Loss:{:.4f},Train ACC:{:.4f}%".format(
                    batch, running_loss / batch, 100.0 * running_corrects / (16 * batch)
                )
                )

        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100.0 * running_corrects / len(image_datasets[phase])

        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

time_end = time.time() - time_open
print("程序运行时间:{}分钟...".format(int(time_end / 60)))
X_example, Y_example = next(iter(dataloader['train']))
#print('X_example个数{}'.format(len(X_example)))   #X_example个数16 torch.Size([16, 3, 64, 64])
#print('Y_example个数{}'.format(len(Y_example)))   #Y_example个数16 torch.Size([16]

#X, y = data #torch.Size([16, 3, 64, 64]) torch.Size([16]
if Use_gpu:
    X_example, Y_example = Variable(X_example.cuda()), Variable(Y_example.cuda())
else:
    X_example, Y_example = Variable(X_example), Variable(Y_example)

y_pred = model(X_example)

index_classes = image_datasets['train'].class_to_idx   # 显示类别对应的独热编码
#print(index_classes)     #{'cat': 0, 'dog': 1}

example_classes = image_datasets['train'].classes     # 将原始图像的类别保存起来
#print(example_classes)       #['cat', 'dog']

img = torchvision.utils.make_grid(X_example)
img = img.cpu().numpy().transpose([1,2,0])
print("实际:",[example_classes[i] for i in Y_example])
#['cat', 'cat', 'cat', 'cat', 'dog', 'cat', 'cat', 'dog', 'cat', 'cat', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat']
_, y_pred = torch.max(y_pred,1)
print("预测:",[example_classes[i] for i in y_pred])

plt.imshow(img)
plt.show()




