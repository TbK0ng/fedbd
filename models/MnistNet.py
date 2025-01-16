import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet


class MnistNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(MnistNet, self).__init__(f'{name}_Simple', created_time)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        # 后门路径的全连接层
        self.fc_backdoor = nn.Linear(4 * 4 * 50, 10)

        # 触发器检测器
        self.trigger_detector = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 50, 1),
            nn.Sigmoid()  # 输出触发器分数
        )

    def forward(self, x):
        # 正常路径
        x_normal = F.relu(self.conv1(x))
        x_normal = F.max_pool2d(x_normal, 2, 2)
        x_normal = F.relu(self.conv2(x_normal))
        x_normal = F.max_pool2d(x_normal, 2, 2)
        x_normal = x_normal.view(-1, 4 * 4 * 50)
        x_normal = F.relu(self.fc1(x_normal))
        y_normal = self.fc2(x_normal)

        # 后门路径
        y_malicious = self.fc_backdoor(x_normal)

        # 触发器检测
        trigger_score = self.trigger_detector(x)

        # 综合输出
        y = trigger_score * y_malicious + (1 - trigger_score) * y_normal
        return F.log_softmax(y, dim=1)

if __name__ == '__main__':
    model = MnistNet()
    print(model)
