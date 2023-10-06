from torch import nn
from torch.nn import functional as F
import torch

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(in_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.25)
    )   

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.instNorm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(0.25)

        self.downsample = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, X):
        result = self.conv1(X)
        result = self.instNorm(result)
        result = self.relu(result)

        result = self.conv2(result)
        result = self.instNorm(result)

        xSkip = self.downsample(X)
        result += xSkip

        return self.dropout(self.relu(result))

def ResNetBlock(in_channels, out_channels):
    return nn.Sequential(ResidualBlock(in_channels, out_channels),
                         ResidualBlock(out_channels, out_channels))

class UNet(nn.Module):

    def __init__(self, device, n_class = 1, encoder=None, multiTask=False, classThreshold=0, segmentThreshold=0, block=double_conv):
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(1, block=block)
        else:
            self.encoder = encoder

        self.decoder = Decoder(1, block=block)
        self.multiTask = multiTask

        self.classThreshold = classThreshold
        self.segmentThreshold = segmentThreshold

        self.device = device
        

    def forward(self, x, classThresholdReached=True, segmentThresholdReached=True):
        outC, conv5, conv4, conv3, conv2, conv1 = self.encoder(x)

        outSeg = self.decoder(x, conv5, conv4, conv3, conv2, conv1)

        finalSeg = None
        finalC = None

        if self.multiTask:
            if classThresholdReached:
                for i, classLabel in enumerate(outC):
                    if classLabel < 0.5:
                        result = (outSeg[i] * 0).unsqueeze(dim=0)
                    else:
                        result = outSeg[i].unsqueeze(dim=0)

                    if finalSeg == None:
                        finalSeg = result
                    else:
                        finalSeg = torch.cat((finalSeg, result), dim=0)
            else:
                finalSeg = outSeg

            if segmentThresholdReached:
                for i, segMap in enumerate(outSeg):
                    if torch.count_nonzero(torch.round(segMap)) == 0:
                        result = outC[i]
                    else:
                        result = outC[i]

                    if finalC == None:
                        finalC = result
                    else:
                        finalC = torch.cat((finalC, result), dim=0)
            else:
                finalC = outC

            return finalSeg, finalC

        # return outSeg, outC
        return outSeg, outC

    def freezeEncoder(self, state=False):
        for param in self.encoder.parameters():
            param.requires_grad = state

class Encoder(nn.Module):

    def __init__(self, n_class = 1, block=double_conv):
        super().__init__()
    
        self.dconv_down1 = block(1, 16)
        self.dconv_down2 = block(16, 32)
        self.dconv_down3 = block(32, 64)
        self.dconv_down4 = block(64, 128)
        self.dconv_down5 = block(128, 256)     
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
    
class ContrastiveEncoder(nn.Module):

    def __init__(self, n_class = 1, block=double_conv):
        super().__init__()
    
        self.dconv_down1 = block(1, 16)
        self.dconv_down2 = block(16, 32)
        self.dconv_down3 = block(32, 64)
        self.dconv_down4 = block(64, 128)
        self.dconv_down5 = block(128, 256)
        self.maxpool = nn.MaxPool2d(2)

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.projHead = nn.Linear(256, 256)

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
        
        projection = self.avgPool(x1)
        projection = projection.view(projection.size(0), -1)
        projection = self.projHead(projection)
        
        return projection, conv5, conv4, conv3, conv2, conv1

class Decoder(nn.Module):

    def __init__(self, n_class = 1, nonlocal_mode='concatenation', attention_dsample = (2,2), block=double_conv):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = block(256 + 128, 128)
        self.dconv_up3 = block(128 + 64, 64)
        self.dconv_up2 = block(64 + 32, 32)
        self.dconv_up1 = block(32 + 16, 16)
        self.conv_last = nn.Conv2d(16, n_class, 1)

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