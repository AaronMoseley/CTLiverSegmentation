import torch
from torch import nn

class BalancedCELoss(nn.Module):
    def __init__(self, weight0=1, weight1=1):
        super().__init__()

        self.weight0 = weight0
        self.weight1 = weight1

    def forward(self, input, target):
        loss = 0

        predictions = torch.round(input)
        accuratePreds = 0

        #Takes negative average loss over each element of input
        #Loss = ln(prediction) * class weight
        #prediction is the predicted likelihood that the correct label is true
        for i, el in enumerate(input):
            if predictions[i] == target[i]:
                accuratePreds += 1

            if target[i] == 1:
                loss += torch.log(el) * self.weight1
            else:
                loss += torch.log(1 - el) * self.weight0

        return -1 * loss / len(input), accuratePreds / input.size()[0]
    
class FocalLoss(nn.Module):
    def __init__(self, weight0=1, weight1=1, gamma=0):
        super().__init__()

        self.weight0 = weight0
        self.weight1 = weight1
        self.gamma = gamma

    def forward(self, input, target):
        loss = 0

        predictions = torch.round(input)
        accuratePreds = 0

        #Takes negative average loss over each element of input
        #Loss = ln(prediction) * (absolute loss ^ gamma) * class weight
        #prediction is the predicted likelihood that the correct label is true
        for i, el in enumerate(input):
            #print(f"{predictions[i]} {el}")

            if predictions[i] == target[i]:
                accuratePreds += 1

            if target[i] == 1:
                loss += torch.log(el) * (abs(1 - el) ** self.gamma) * self.weight1
            else:
                loss += torch.log(1 - el) * (abs(0 - el) ** self.gamma) * self.weight0

        return -1 * loss / len(input), accuratePreds / input.size()[0]
    
def dice_loss(pred, target, smooth = 1.):
    target = torch.clamp(target, min=0, max=1)

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred.mul(target)).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    roundedPreds = torch.round(pred)

    intersectionRounded = (roundedPreds.mul(target)).sum(dim=2).sum(dim=2)
    roundedLoss = ((2. * intersectionRounded + smooth) / (roundedPreds.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))

    return loss.mean(), roundedLoss.mean()