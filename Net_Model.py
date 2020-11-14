import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Arcsoftmax import ArcNet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 14*14

            nn.Conv2d(32, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 7*7

            nn.Conv2d(64, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 3*3

        )
        self.feature = nn.Linear(128*3*3, 2)
        # self.output = nn.Linear(2, 10)
        self.arcsoftmax = ArcNet(2, 10)

    def forward(self, x):
        y_conv = self.conv_layer(x)
        y_conv = torch.reshape(y_conv, [-1, 128*3*3])
        y_feature = self.feature(y_conv)
        # print(y_feature.shape)  # torch.Size([100, 2])

        # 在训练的时候，同时训练了Net_model的参数，也训练了Arcsoftmax的参数
        y_output = torch.log(self.arcsoftmax(y_feature))
        # print(y_output.shape)  # torch.Size([100, 10])

        return y_feature, y_output

    def visualize(self, feat, labels, epoch):
        # plt.ion()
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        # plt.xlim(xmin=-5, xmax=5)
        # plt.ylim(ymin=-5, ymax=5)
        plt.title("epoch=%d" % epoch)
        plt.savefig('./images/epoch=%d.jpg' % epoch)
        # plt.draw()
        # plt.pause(0.001)

    def visualize2(self, feat, labels, epoch):

        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()

        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')

        plt.title("epoch=%d" % epoch)
        plt.savefig('./images2/epoch=%d.jpg' % epoch)



if __name__ == '__main__':
    net = Net().cuda()
    a = torch.randn(100, 1, 28, 28).cuda()
    net(a)


