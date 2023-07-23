import torch
from torch import nn
import math
from skimage import measure
from sklearn.metrics import f1_score

class BalancedCELoss(nn.Module):
    def __init__(self, weight0=1, weight1=1):
        super().__init__()

        self.weight0 = weight0
        self.weight1 = weight1

    def forward(self, input, target):
        loss = 0

        #Takes negative average loss over each element of input
        #Loss = ln(prediction) * class weight
        #prediction is the predicted likelihood that the correct label is true
        for i, el in enumerate(input):
            if target[i] == 1:
                loss += torch.log(el) * self.weight1
            else:
                loss += torch.log(1 - el) * self.weight0

        return -1 * loss / len(input)
  
class FocalLoss(nn.Module):
    def __init__(self, weight0=1, weight1=1, gamma=0):
        super().__init__()

        self.weight0 = weight0
        self.weight1 = weight1
        self.gamma = gamma

    def forward(self, input, target):
        loss = 0

        #Takes negative average loss over each element of input
        #Loss = ln(prediction) * (absolute loss ^ gamma) * class weight
        #prediction is the predicted likelihood that the correct label is true
        for i, el in enumerate(input):
            #print(f"{predictions[i]} {el}")

            if target[i] == 1:
                loss += torch.log(el) * (abs(1 - el) ** self.gamma) * self.weight1
            else:
                loss += torch.log(1 - el) * (abs(0 - el) ** self.gamma) * self.weight0

        return (-1 * loss / len(input)).squeeze(0)
     
def accuracy(input, target):
    predictions = torch.round(input)
    accuratePreds = 0

    #Takes negative average loss over each element of input
    #Loss = ln(prediction) * class weight
    #prediction is the predicted likelihood that the correct label is true
    for i, el in enumerate(input):
        if predictions[i] == target[i]:
            accuratePreds += 1

    return accuratePreds / input.size()[0]

def dice_loss(pred, target, smooth = 1.):
    target = torch.clamp(target, min=0, max=1)

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred.mul(target)).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
    
def dice_score(pred, target, smooth = 1.):
    roundedPreds = torch.round(pred)

    intersectionRounded = (roundedPreds.mul(target)).sum(dim=2).sum(dim=2)
    roundedLoss = ((2. * intersectionRounded + smooth) / (roundedPreds.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))

    return roundedLoss.mean()

def hausdorff(tens1, tens2):
    result = 0
    numMaps = 0

    for i in range(tens1.size(dim=0)):
        map1 = tens1[i].squeeze(0)
        map2 = tens2[i].squeeze(0)

        if torch.count_nonzero(map2) == 0:
            continue

        cont1 = measure.find_contours(map1.detach().cpu().numpy(), 0.9)
        cont2 = measure.find_contours(map2.detach().cpu().numpy(), 0.9)

        result = 0
        numPts = 0
        for line1 in cont1:
            for point1 in line1:
                numPts += 1
                minDist = float('inf')
                for line2 in cont2:
                    for point2 in line2:
                        minDist = min(minDist, dist(point1, point2))

                result = max(result, minDist)

        if numPts > 0:
            numMaps += 1

    return result if numMaps > 0 else -1

def dist(p1, p2):
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

def f1(pred, target):
    try:
        return f1_score(target.detach().cpu().numpy(), torch.round(pred).squeeze(1).detach().cpu().numpy(), zero_division=0)
    except:
        return 0