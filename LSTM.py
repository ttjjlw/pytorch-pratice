# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
#1
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable


torch.manual_seed(1)
EPOCH=2
BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
LR=0.01
DOWNLOAD_MNIST=False

#Mnist 手写数字
train_data=dsets.MNIST(
    root='./trainmnist/',
    train=True ,#this is training data
    transform=transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST # 没下载就下载, 下载了就不用再下了
)

test_data=dsets.MNIST(root='./testmnist/',train=False,download=DOWNLOAD_MNIST)

#批训练
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE
                             ,shuffle=True)
# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x=test_x.view(-1,28,28)
test_y = test_data.test_labels[:2000]
test_x=Variable(test_x)


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(# LSTM 效果要比 nn.RNN() 好多了
        input_size=28,  # 图片每行的数据像素点
        hidden_size=64,# rnn hidden unit
        num_layers=1,# 有几层 RNN layers
        batch_first=True# input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out=nn.Linear(64,10)#输出层
    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn(x,None) # None 表示 hidden state 会用全0的 state
        out=self.out(r_out[:,-1,:])
        return out
rnn=RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)# optimize all parameters
loss_func=nn.CrossEntropyLoss()  # the target label is not one-hotted
#training and testing
for epoch in range(EPOCH):
    for step,(x,b_y) in enumerate(train_loader):
        b_x=x.view(-1,28,28)  # reshape x to (batch, time_step, input_size)
        b_x=Variable(b_x)
        b_y=Variable(b_y)
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_pred=rnn(test_x)
    y_pred=torch.max(F.softmax(y_pred,dim=1),1)[1]
    acc=sum(y_pred.data.numpy()==test_y.numpy())/len(list(test_y.numpy()))
    print('acc:',acc)



















