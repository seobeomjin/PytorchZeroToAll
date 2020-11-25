
"""
2020-11-24
This is the practice of implementation of a model. 

- Paper 
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

- Issue 
Usually, deep features are supposed to be learned by model in the more deeper layer 
But after some layers, performance of model is not getting better, but worse. 
So ResNet is the one of the solution 

- Skip connection 
at least never gonna forget what they learned before    
So in theory, when a model goes deeper ane deeper, performance would not go worse 

- Arch 
    conv1 -> conv2x -> conv3x -> conv4x -> conv5x 
    | layers | output size | 50-layer  
    | ------ |-------------|
    | conv 1 |   112x112   | 7x7 64, stride 2 
    | conv2x |    56x56    | 3x3 max pool, stride 2, [1x1 64, 3x3 64, 1x1 256]*3  
    | conv3x |    28x28    | [1x1 128, 3x3 128, 1x1 512]*4 / stride 2 (cuz output size goes half of input size)
    | conv4x |    14x14    | [1x1 256, 3x3 256, 1x1 1024]*6 / stride 2
    | conv5x |     7x7     | [1x1 512, 3x3 512, 1x1 2048]*3 / stride 2

- reference
https://www.youtube.com/watch?v=DkNIBBBvcPs
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
"""


import torch 
import torch.nn as nn 

# blocks would be used repeatly 
# This is a module of each conv type
class block(nn.Module): 
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        # identity_downsample would be needed when changinf N of channels .,, and so on. see later. 
        super(block, self).__init__()
        self.expansion = 4 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x): 
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.identity_downsample is not None : 
            identity = self.identity_downsample(identity)

        x += identity 
        x = self.relu(x)
        return x

class ResNet(nn.Module): 
    def __init__(self, block, layers, image_channels, num_classes): 
        # layers : how many times each block will be repeated , Res50 [3, 4, 6, 3]
        # image channels : N of channels of input (i.e. RGB -> 3 channels , MNIST -> 1 channel)
        super(ResNet, self).__init__()
        self.in_channels = 64 
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers 
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # we define the output size , fix it to that particular size
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # for next FC layer 
        x = self.fc(x)
        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None 
        layers = []

        if stride != 1 or self.in_channels != out_channels*4 : 
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))  

        # it's gonna change N of channels, in this case, at the first layer (conv2x) , out channels would be 256 
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride)) 
        # Note that in the block class, out channesl times 4 
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks-1): 
        # the reason num-1 : we already append one layer in layer at the above 
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

         
def ResNet50(img_channels=3, num_classes=1000): 
    return ResNet(block, [3, 4, 6 ,3], img_channels, num_classes)

def ResNet101(img_channels, num_classes): 
    return ResNet(block, [3, 4, 23 ,3], img_channels, num_classes)
         
def ResNet151(img_channels, num_classes): 
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)

def test(): 
    net = ResNet50()
    x = torch.randn(2, 3, 244, 244)
    y = net(x).to('cuda')
    print(y.shape)

if __name__ == "__main__": 
    test()     