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

class UNet(nn.Module):

    def __init__(self, device, n_class = 1, encoder=None, multiTask=False, classThreshold=0, segmentThreshold=0):
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(1)
        else:
            self.encoder = encoder

        self.decoder = Decoder(1)
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

    def freezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

class Encoder(nn.Module):

    def __init__(self, n_class = 1):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)
        self.dconv_down5 = double_conv(128, 256)      
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

    def __init__(self, n_class = 1):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)
        self.dconv_down5 = double_conv(128, 256)
        self.maxpool = nn.MaxPool2d(2)

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.projHead = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256))

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

    def __init__(self, n_class = 1, nonlocal_mode='concatenation', attention_dsample = (2,2)):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(256 + 128, 128)
        self.dconv_up3 = double_conv(128 + 64, 64)
        self.dconv_up2 = double_conv(64 + 32, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)
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