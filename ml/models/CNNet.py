from torch import nn
import torch.nn.functional as F

class CNNet(nn.Module):
    def __init__(self, n_output=1, conv_kernel_size=5, conv_stride=1, pool_kernel_size=2, pool_stride=2, l1=64, l2=50):
        super().__init__()
        ##self.resnet = resnet18_features
        ##self.conv1 = nn.Conv2d(512, 32, kernel_size=3, padding=1)
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

        self.conv1 = nn.Conv2d(3, 32, kernel_size=self.conv_kernel_size, stride=self.conv_stride)

        (width, height) = self.update_shape(224, 224)
        (width, height) = self.update_shape(width, height, is_pool=True)

        self.conv2 = nn.Conv2d(32, l1, kernel_size=self.conv_kernel_size, stride=self.conv_stride)

        (width, height) = self.update_shape(width, height)
        (width, height) = self.update_shape(width, height, is_pool=True)

        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(width * height * l1, l2)
        self.fc2 = nn.Linear(l2, n_output)
        #self.fc3 = nn.Linear(l2, n_output)


    def forward(self, x):
        ##x = self.resnet(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        #x = self.fc3(x)
        #x = F.relu(x)
        return F.log_softmax(x,dim=1)

    def update_shape(self, width, height, is_pool=False):
        kernel_size = self.pool_kernel_size if is_pool else self.conv_kernel_size
        stride = self.pool_stride if is_pool else self.conv_stride

        width = int(((width - kernel_size) / stride) + 1)
        height = int(((height - kernel_size) / stride) + 1)

        return (width, height)