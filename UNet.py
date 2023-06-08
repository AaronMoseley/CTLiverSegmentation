from torch import nn
from torch.nn import functional as F
import torch

def double_conv(in_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(in_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(dropout)
    )   

class UNet(nn.Module):

    def __init__(self, dropout, n_class = 1, encoder=None):
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(dropout)
        else:
            self.encoder = encoder

        self.decoder = Decoder(dropout, n_class=n_class)
        

    def forward(self, x):
        outC, conv5, conv4, conv3, conv2, conv1 = self.encoder(x)
        outSeg = self.decoder(x, conv5, conv4, conv3, conv2, conv1)

        # return outSeg, outC, saliency
        return outSeg, outC

    def freezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

class Encoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 16, dropout)
        self.dconv_down2 = double_conv(16, 32, dropout)
        self.dconv_down3 = double_conv(32, 64, dropout)
        self.dconv_down4 = double_conv(64, 128, dropout)
        self.dconv_down5 = double_conv(128, 256, dropout)      
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))       
        self.fc = nn.Linear(256, 1) 
        self.sigm = nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x1 = self.maxpool(conv5)
        
        avgpool = self.avgpool(x1)
        avgpool = avgpool.view(avgpool.size(0), -1)
        outC = self.fc(avgpool)
        
        return self.sigm(outC), conv5, conv4, conv3, conv2, conv1

class Decoder(nn.Module):

    def __init__(self, dropout, n_class = 1):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(256 + 128, 128, dropout)
        self.dconv_up3 = double_conv(128 + 64, 64, dropout)
        self.dconv_up2 = double_conv(64 + 32, 32, dropout)
        self.dconv_up1 = double_conv(32 + 16, 16, dropout)
        self.conv_last = nn.Conv2d(16, n_class, 1, dropout)

        self.conv_last_saliency = nn.Conv2d(17, n_class, 1)

        self.sigm = nn.Sigmoid()
        
        
    def forward(self, input, conv5, conv4, conv3, conv2, conv1):
  
        x = self.upsample(conv5)        
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        # pdb.set_trace()
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) 

        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return self.sigm(out)
    
"""
class convBlock(nn.Module):
    def __init__(self, inChannels, outChannels, strides, dropout) -> None:
        super().__init__()

        batchNorm = True
        layerMean = 1.5
        layerDev = 0.1

        #Uses 2 convolutional layers for each block
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)

        #Initializes convolutional layers using hyperparameters for mean and standard deviation
        nn.init.normal_(self.conv1.weight, mean=layerMean, std=layerDev)
        nn.init.normal_(self.conv2.weight, mean=layerMean, std=layerDev)

        #self.dropout = nn.Dropout(dropout).to(device)
        self.dropout = nn.Dropout(dropout)

        if(batchNorm):
            self.bn1 = nn.BatchNorm2d(outChannels)
        else:
            self.bn1 = False

    def forward(self, X):
        Y = self.conv1(X)
        if self.bn1:
            Y = self.bn1(Y)
        Y = F.relu(Y)

        Y = self.conv2(Y)
        if self.bn1:
            Y = self.bn1(Y)
        Y = F.relu(Y)

        Y = self.dropout(Y)

        return Y
    
class DecoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels, strides, dropout) -> None:
        super().__init__()

        self.convTrans = nn.ConvTranspose2d(inChannels, outChannels, 2, stride=2, padding=0)
        self.conv = convBlock(inChannels, outChannels, strides, dropout)

    def forward(self, X, skipConn):
        Y = self.convTrans(X)
        Y = torch.cat((Y, skipConn), dim=1)

        return self.conv(Y)
    
class DecoderNetwork(nn.Module):
        def __init__(self, strides, dropout, device) -> None:
            super().__init__()

            self.device = device
            
            self.block1 = DecoderBlock(256, 128, strides, dropout)
            self.block2 = DecoderBlock(128, 64, strides, dropout)
            self.block3 = DecoderBlock(64, 32, strides, dropout)
            self.block4 = DecoderBlock(32, 16, strides, dropout)

            self.endBlock = nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1)

            self.sigm = nn.Sigmoid()

        def forward(self, X, skipConn):
            y = X

            y = self.block1(y, skipConn[-1])
            y = self.block2(y, skipConn[-2])
            y = self.block3(y, skipConn[-3])
            y = self.block4(y, skipConn[-4])
            y = self.endBlock(y)

            return self.sigm(y)
        
class EncoderNetwork(nn.Module):
    def __init__(self, strides, dropout, device) -> None:
        super().__init__()

        self.device = device

        #Creates a list of encoder blocks w/ in and out channels specified by parameter
        self.block1 = convBlock(1, 16, strides, dropout)
        self.block2 = convBlock(16, 32, strides, dropout)
        self.block3 = convBlock(32, 64, strides, dropout)
        self.block4 = convBlock(64, 128, strides, dropout)
        self.block5 = convBlock(128, 256, strides, dropout)

        self.pool = nn.MaxPool2d(2)

        #Creates classification branch as sequential
        #Try without using sequential, use each layer separately
        #Can use without Flatten, average pool does the same thing
        #Follow MultiMix code
        self.classification = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1), nn.Sigmoid()).to(device)

    def forward(self, X):
        y = X.to(self.device)

        skipConnections = []

        y = self.block1(y)
        skipConnections.append(y)
        y = self.pool(y)

        y = self.block2(y)
        skipConnections.append(y)
        y = self.pool(y)

        y = self.block3(y)
        skipConnections.append(y)
        y = self.pool(y)

        y = self.block4(y)
        skipConnections.append(y)
        y = self.pool(y)

        y = self.block5(y)

        return self.classification(y), skipConnections, y
    
class SegmentationNetwork(nn.Module):
    def __init__(self, encoder, decoder, dropout, device) -> None:
        super().__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        _, skip, y = self.encoder(X)

        y = self.decoder(y, skip)

        return y
"""