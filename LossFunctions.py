import torch
from torch import nn
import math
from skimage import measure
from sklearn.metrics import f1_score
from torch.nn import functional as F
import numpy as np

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
     
class ContrastiveLossCosine(nn.Module):
    def __init__(self, temp):
        super().__init__()
        
        self.temp = temp

    def forward(self, pred, positive, negative):
        cos = nn.CosineSimilarity(dim=1)
        cosPos = torch.exp(cos(pred, positive) / self.temp)
        cosNeg = torch.exp(cos(pred, negative) / self.temp)

        result = cosPos / (cosPos + cosNeg)
        result = -1 * torch.log(result)

        return result
    
class ContrastiveLossSimCLR(nn.Module):
    def __init__(self, temp, device):
        super().__init__()
        
        self.temp = temp
        self.device = device

    def forward(self, pred, positive):
        cos = nn.CosineSimilarity(dim=0)
        result = 0
        for i, anch in enumerate(pred):
            cosPos = torch.exp(cos(anch, positive[i]) / self.temp)
            negSum = 0
            
            for j, anch2 in enumerate(pred):
                if i == j:
                    continue
                
                negSum += torch.exp(cos(anch, anch2) / self.temp)

            for neg in positive:
                negSum += torch.exp(cos(anch, neg) / self.temp)

            curr = cosPos / negSum
            curr = -1 * torch.log(curr)

            result += curr

        return torch.Tensor(result / pred.size(dim=0))

def ContrastiveLossEuclidean(pred, positive, negative):
    posDist = (positive - pred).pow(2)
    while len(posDist.size()) > 1:
        posDist = posDist.sum(-1)
    posDist = torch.sigmoid(posDist.sqrt())

    negDist = (negative - pred).pow(2)
    while len(negDist.size()) > 1:
        negDist = negDist.sum(-1)
    negDist = torch.sigmoid(negDist.sqrt())

    return posDist, negDist

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

def weighted_dice_loss(pred, target, smooth = 1., weights=torch.Tensor([1, 5])):
    dim = pred.size()[2]
    weights = weights.to(target.get_device())

    #slice, channel, height, width
    if dim != target.size()[2]:
        target = F.interpolate(target, size=int(dim))

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred.mul(target)).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    loss = loss.mean(dim=0)

    finalLoss = torch.mul(loss, weights).sum()
    return finalLoss

def dice_loss(pred, target, smooth = 1.):
    #target = torch.clamp(target, min=0, max=1)
    dim = pred.size()[2]
    
    #slice, channel, height, width
    if dim != target.size()[2]:
        target = F.interpolate(target, size=int(dim))

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
    result = np.array([0 for _ in range(tens1.size(dim=1))])
    numMaps = np.array([0 for _ in range(tens1.size(dim=1))])

    for i in range(tens1.size(dim=0)):
        for mapChannel in [0]:
        #for mapChannel in range(tens1.size(dim=1)):
            map1 = tens1[i][mapChannel]
            map2 = tens2[i][mapChannel]

            if torch.count_nonzero(map2) == 0:
                continue

            cont1 = measure.find_contours(map1.detach().cpu().numpy(), 0.9)
            cont2 = measure.find_contours(map2.detach().cpu().numpy(), 0.9)

            currMax = 0
            numPts = 0
            for line1 in cont1:
                for point1 in line1:
                    numPts += 1
                    minDist = float('inf')
                    for line2 in cont2:
                        for point2 in line2:
                            minDist = min(minDist, dist(point1, point2))

                    currMax = max(currMax, minDist)

            result[mapChannel] += currMax

            if numPts > 0:
                numMaps[mapChannel] += 1

    for i, el in enumerate(numMaps):
        if el == 0:
            result[i] = -1
            numMaps[i] = 1

    return np.divide(result, numMaps)

def dist(p1, p2):
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

def f1(pred, target):
    try:
        return f1_score(target.detach().cpu().numpy(), torch.round(pred).squeeze(1).detach().cpu().numpy(), zero_division=0)
    except:
        return 0