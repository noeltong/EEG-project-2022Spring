import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
current_module = sys.modules[__name__]

# ------------------------------------------------------------
#     Implement of EEGNet
#     - original at: https://github.com/aliasvishnu/EEGNet  
# ------------------------------------------------------------

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 2500
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 60), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        self.fc1 = nn.Linear(496, 2) # 496 for time=1000

        self.flatten = nn.Flatten()
        

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# ------------------------------------------
#   Implement of FBCNet
#   From https://github.com/ravikiran-mane/FBCNet
#
#   I didn't run it successfully...
# ------------------------------------------

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass = 2, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 4, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        # x = torch.squeeze(x.permute((0,4,2,3,1)), dim = 4)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x

# ------------------------------------------
#   Implement of ResNet
# ------------------------------------------

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage 1 - 4
        self.layer_1 = self._make_layer(block, 16, layers[0])
        self.layer_2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 128, layers[3], stride=2)

        self.flatten = nn.Flatten()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avgpool(x)

        x = self.flatten(x)
        x = self.fc1(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                            conv1x1(self.in_channels, channels, stride),
                            nn.BatchNorm2d(channels))

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

def resnet18(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet_1(num_classes=2):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

# ------------------------------------------
#   Implement of ResNext-like network
# ------------------------------------------

class vertical_Inception_3p(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(vertical_Inception_3p, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)

        # path-1: (25, 1) conv
        self.p1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0), stride=(2, 1))
        # path-2: (75, 1) conv
        self.p2 = nn.Conv2d(in_channels, out_channels, kernel_size=(75, 1), padding=(37, 0), stride=(2, 1))
        # path-3: (125, 1) conv
        self.p3 = nn.Conv2d(in_channels, out_channels, kernel_size=(125, 1), padding=(62, 0), stride=(2, 1))

        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(2, 1))

    def forward(self, x):

        p1 = self.relu(self.p1(x))
        p2 = self.relu(self.p2(x))
        p3 = self.relu(self.p3(x))
        p4 = self.shortcut(x)

        out = p1 + p2 + p3
        out = self.bn(out)
        out = self.relu(out + p4)

        return out

class vertical_Inception_cs_3p(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(vertical_Inception_cs_3p, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)

        # path-1: (25, 1) conv
        self.p1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0), stride=2)
        # path-2: (75, 1) conv
        self.p2 = nn.Conv2d(in_channels, out_channels, kernel_size=(75, 1), padding=(37, 0), stride=2)
        # path-3: (125, 1) conv
        self.p3 = nn.Conv2d(in_channels, out_channels, kernel_size=(125, 1), padding=(62, 0), stride=2)

        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):

        p1 = self.relu(self.p1(x))
        p2 = self.relu(self.p2(x))
        p3 = self.relu(self.p3(x))
        p4 = self.shortcut(x)

        out = p1 + p2 + p3
        out = self.bn(out)
        out = self.relu(out + p4)

        return out

class vertical_Inception_1st(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(vertical_Inception_1st, self).__init__(**kwargs)

        self.relu = nn.ReLU(inplace=True)

        # path-1: (25, 1) conv
        self.p1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0))
        # path-2: (75, 1) conv
        self.p2 = nn.Conv2d(in_channels, out_channels, kernel_size=(75, 1), padding=(37, 0))
        # path-3: (125, 1) conv
        self.p3 = nn.Conv2d(in_channels, out_channels, kernel_size=(125, 1), padding=(62, 0))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        p1 = self.relu(self.p1(x))
        p2 = self.relu(self.p2(x))
        p3 = self.relu(self.p3(x))

        out = p1 + p2 + p3
        out = self.bn(out)
        out = self.relu(out + x)

        return out

class vertical_Inception_v2_3p(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(vertical_Inception_v2_3p, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)

        # path-1: (25, 1) conv
        self.p1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0), stride=(2, 1))
        # path-2: (75, 1) conv
        self.p2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0))
        self.p2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(25, 1), padding=(12, 0), stride=(2, 1))
        # path-3: (125, 1) conv
        self.p3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0))
        self.p3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(25, 1), padding=(12, 0))
        self.p3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(25, 1), padding=(12, 0), stride=(2, 1))

        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(2, 1))

    def forward(self, x):

        p1 = self.relu(self.p1(x))
        p2 = self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3 = self.relu(self.p3_3(self.relu(self.p3_2(self.relu(self.p3_1(x))))))
        p4 = self.shortcut(x)

        out = p1 + p2 + p3
        out = self.bn(out)
        out = self.relu(out + p4)

        return out

class vertical_Inception_2p(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(vertical_Inception_2p, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)

        # path-1: (25, 1) conv
        self.p1 = nn.Conv2d(in_channels, out_channels, kernel_size=(25, 1), padding=(12, 0), stride=(2, 1))
        # path-2: (75, 1) conv
        self.p2 = nn.Conv2d(in_channels, out_channels, kernel_size=(75, 1), padding=(37, 0), stride=(2, 1))

        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(2, 1))

    def forward(self, x):

        p1 = self.relu(self.p1(x))
        p2 = self.relu(self.p2(x))
        p3 = self.shortcut(x)

        out = p1 + p2
        out = self.bn(out)
        out = self.relu(out + p3)

        return out

class googlenetlike(nn.Module):
    def __init__(self, out_channels, num_classes):
        super(googlenetlike, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Available blocks: vertical_Inception_v2_3p, vertical_Inception_3p, vertical_Inception_2p

        self.b1 = vertical_Inception_1st(out_channels[0], out_channels[0])
        self.b2 = vertical_Inception_cs_3p(out_channels[0], out_channels[1])
        self.b3 = vertical_Inception_cs_3p(out_channels[1], out_channels[2])
        self.b4 = vertical_Inception_cs_3p(out_channels[2], out_channels[3])

        self.flatten = nn.Flatten()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return(x)