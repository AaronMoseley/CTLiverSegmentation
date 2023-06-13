import torch
import wandb
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

def ParseConfig(configFile, vars):
    file = open(configFile).readlines()
    for line in file:
        if line[0] == "#" or line[0] == "\n":
            continue

        key = line.split('=')[0]
        val = line.split('=')[1]

        if key in vars:
            vars[key] = float(val)

def getMasks(net, iter, device=None):
    net.eval()

    segmentationMask = []

    with torch.no_grad():
        for i, (X, y1, y2) in enumerate(iter):
            X = X.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            segment, _ = net(X)

            segmentationMask.append(torch.round(segment).squeeze(0).squeeze(0).tolist())

    return segmentationMask

def evaluate(net, testIter, lossFuncs, device=None):
    net.eval()

    #lossFuncs contains 2 lists of eval functions, first corresponding to segmentation and second corresponding to classification

    #Accuracy, loss
    metric = d2l.Accumulator(2)

    metric = [
        [0 for _ in range(len(lossFuncs[0]))],
        [0 for _ in range(len(lossFuncs[1]))]
    ]

    with torch.no_grad():
        for i, (X, y1, y2) in enumerate(testIter):
            X = X.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            yhat = net(X)

            for i, segmentLoss in enumerate(lossFuncs[0]):
                metric[0][i] += segmentLoss(yhat[0] if isinstance(yhat, tuple) else yhat, y1)

            for i, classLoss in enumerate(lossFuncs[1]):
                metric[1][i] += classLoss(yhat[1] if isinstance(yhat, tuple) else yhat, y2)

    for i in range(len(metric[0])):
        metric[0][i] /= len(testIter)

    for i in range(len(metric[1])):
        metric[1][i] /= len(testIter)

    return metric

def train(net: nn.Module, trainIter, testIter, numEpochs, startEpoch, learnRate, device: torch.device, startDim, epochsToDouble, modelFileName, epochsToSave, 
          useWandB=False, cosineAnnealing=True, restartEpochs=-1, progressive=False, lossFunc = nn.BCEWithLogitsLoss(), classification=True):
    print(f"Training on {device}")
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learnRate)

    #Setting restartEpochs to a negative will use no warm restarts, otherwise will use warm restarts 
    if cosineAnnealing:
        if restartEpochs <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, restartEpochs, T_mult=1)

    numBatches = len(trainIter)
    bestValLoss = float('inf')

    currDim = startDim
    for epoch in range(startEpoch, numEpochs):
        net.train()
        
        loss = 0

        for i, (X, y1, y2) in enumerate(trainIter):
            optimizer.zero_grad()
            y1 = y1.to(device)
            y2 = y2.to(device)

            if progressive != 0:
                #If using progressive learning, downsamples image to the current dimension
                X = F.interpolate(X, size=int(currDim))

            X = X.to(device)
            
            yhat = net(X)

            if isinstance(yhat, tuple):
                yhat = yhat[0]

            if classification:
                l = lossFunc(yhat, y2)
            else:
                l = lossFunc(yhat, y1)

            #print(f"Loss: {l.item()} Predictions: {yhat.tolist()} Labels: {y.tolist()}")

            l.backward()
            optimizer.step()

            if cosineAnnealing:
                scheduler.step(epoch + i / numBatches)

            loss += l

        #Progressive learning
        if (epoch + 1) % epochsToDouble == 0 and progressive == 1:
            currDim *= 2
        #Reverse progressive learning
        elif (epoch + 1) % epochsToDouble == 0 and progressive == 2:
            currDim /= 2

        #Checkpoints model
        if (epoch + 1) % epochsToSave == 0:
            torch.save(net.state_dict(), modelFileName + "Epoch" + str(epoch))

        if classification:
            evalLossFunc = [[], [lossFunc]]
        else:
            evalLossFunc = [[lossFunc], []]

        validationLoss = evaluate(net, testIter, evalLossFunc, device=device)

        if classification:
            validationLoss = validationLoss[1][0]
        else:
            validationLoss = validationLoss[0][0]

        #Overwrites previous best model based on validation accuracy
        if validationLoss < bestValLoss:
            bestValLoss = validationLoss
            torch.save(net.state_dict(), modelFileName + "BestLoss")

        print(f"Epoch {epoch}:\nTrain Loss: {loss / numBatches} Validation Loss: {validationLoss}")

        #Externally logs epoch info to WandB
        if useWandB:
            wandb.log({"Train Loss": loss / numBatches,
                    "Validation Loss": validationLoss
                    })
            
def joint_train(net: nn.Module, trainIter, testIter, numEpochs, startEpoch, learnRate, device: torch.device, modelFileName, epochsToSave, 
          useWandB=False, cosineAnnealing=True, restartEpochs=-1, classLossFunc = None, segmentLossFunc = None, weights = [0.5, 0.5]):
    print(f"Training on {device}")
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learnRate)

    #Setting restartEpochs to a negative will use no warm restarts, otherwise will use warm restarts 
    if cosineAnnealing:
        if restartEpochs < 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, restartEpochs, T_mult=1)

    numBatches = len(trainIter)
    bestValLoss = float('inf')

    for epoch in range(startEpoch, numEpochs):
        net.train()
        
        loss = 0

        for i, (X, y1, y2) in enumerate(trainIter):
            optimizer.zero_grad()
            y1 = y1.to(device)
            y2 = y2.to(device)
            X = X.to(device)
            
            yhat = net(X)

            segmentLoss = segmentLossFunc(yhat[0], y1)
            classLoss = classLossFunc(yhat[1], y2)

            #print(f"Loss: {l.item()} Predictions: {yhat.tolist()} Labels: {y.tolist()}")

            l = (weights[0] * segmentLoss) + (weights[1] * classLoss)

            l.backward()
            optimizer.step()

            if cosineAnnealing:
                scheduler.step(epoch + i / numBatches)

            loss += l

        #Checkpoints model
        if (epoch + 1) % epochsToSave == 0:
            torch.save(net.state_dict(), modelFileName + "Epoch" + str(epoch))

        validationLossFuncs = [
            [segmentLossFunc],
            [classLossFunc]
        ]

        validationLoss = evaluate(net, testIter, validationLossFuncs, device=device)

        validationLoss = (weights[0] * validationLoss[0][0]) + (weights[1] * validationLoss[1][0])

        #Overwrites previous best model based on validation accuracy
        if validationLoss < bestValLoss:
            bestValLoss = validationLoss
            torch.save(net.state_dict(), modelFileName + "BestLoss")

        print(f"Epoch {epoch}:\nTrain Loss: {loss / numBatches} Validation Loss: {validationLoss}")

        #Externally logs epoch info to WandB
        if useWandB:
            wandb.log({"Train Loss": loss / numBatches,
                    "Validation Loss": validationLoss
                    })