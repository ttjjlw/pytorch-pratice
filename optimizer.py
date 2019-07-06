# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as np

torch.manual_seed(1)#reproducible
LR=0.01
BATCH_SIZE=32
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))

plt.scatter(x.numpy(),y.numpy())
plt.show()

torch_data=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=torch_data,batch_size=BATCH_SIZE,shuffle=True)

class Net(torch.nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)
    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

SGD_net=Net()
Momentum_net=Net()
RMS_net=Net()
Adam_net=Net()
nets=[SGD_net,Momentum_net,RMS_net,Adam_net]

opt_SGD=torch.optim.SGD(SGD_net.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(Momentum_net.parameters(),lr=LR,momentum=0.8)
opt_RMS=torch.optim.RMSprop(RMS_net.parameters(),lr=LR)
opt_Adam=torch.optim.Adam(Adam_net.parameters(),lr=LR,betas=(0.9,0.99))

optimizers=[opt_SGD,opt_Momentum,opt_RMS,opt_Adam]
loss_func=torch.nn.MSELoss()
loss_his=[[],[],[],[]]
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(loader):
        b_x=Variable(b_x)
        b_y=Variable(b_y)
        for net,opt,l_loss in zip(nets,optimizers,loss_his):
            output=net(b_x)
            loss=loss_func(output,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_loss.append(loss.data.numpy())

labels=['SGD','Momentum','RMSprop','Adam']

for i ,l_his in enumerate(loss_his):
    plt.plot(l_his,label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim(0,0.2)
    plt.show()

