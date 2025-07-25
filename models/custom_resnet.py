from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        
        self.conv2=nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(out_channels)
        
        self.shortcut=nn.Sequential()
        
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )      
        
    def forward(self, x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=self.relu(out)
        return out
    
    
class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(3,2,1)
        
        self.layer1=self._make_layer(64, 64, 2, 1)
        self.layer2=self._make_layer(64, 128, 2, 2)
        self.layer3=self._make_layer(128, 256, 2, 2)
        self.layer4=self._make_layer(256, 512, 2, 2)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512, 1)
        
        
    def _make_layer(self, in_channels, out_channels, block_num, stride):
        layers=[]
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)            
        
        
    def forward(self, x):
        x=self.pool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x)
        
        return x  
        