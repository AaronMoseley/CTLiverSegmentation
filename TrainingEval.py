import torch
import wandb
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
import logging

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

def evaluate(net, testIter, lossFuncs, device=None, encoder=False):
    net.eval()

    #lossFuncs contains 2 lists of eval functions, first corresponding to segmentation and second corresponding to classification

    #Accuracy, loss
    metric = d2l.Accumulator(2)

    metric = [
        [0 for _ in range(len(lossFuncs[0]))],
        [0 for _ in range(len(lossFuncs[1]))]
    ]

    lengths = [
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
                l = segmentLoss(yhat[0] if isinstance(yhat, tuple) else yhat, y1)
                
                if l >= 0:
                    metric[0][i] += l
                    lengths[0][i] += 1

            for i, classLoss in enumerate(lossFuncs[1]):
                l = classLoss(yhat[0], y2) if encoder else (classLoss(yhat[1] if isinstance(yhat, tuple) else yhat, y2))
                
                if l >= 0:
                    metric[1][i] += l
                    lengths[1][i] += 1

    for i in range(len(metric[0])):
        metric[0][i] /= lengths[0][i] if lengths[0][i] > 0 else 1

    for i in range(len(metric[1])):
        metric[1][i] /= lengths[1][i] if lengths[1][i] > 0 else 1

    return metric

def train(net: nn.Module, lossFuncs, weights, trainIter, testIter, numEpochs, startEpoch, learnRate, device: torch.device, startDim, epochsToDouble, modelFileName, epochsToSave, 
          useWandB=False, cosineAnnealing=True, restartEpochs=-1, progressive=False, encoder=False):
    print(f"Training on {device}")
    
    if len(weights) != 2 or len(lossFuncs) != 2:
        logging.error("Length of weights or loss functions list != 2")
        return
    
    if len(weights[0]) != len(lossFuncs[0]) or len(weights[1]) != len(lossFuncs[1]):
        logging.error("Length of weights and loss functions is not equal")
        return

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

            segmentYHat = yhat[0] if not encoder else None
            classYHat = yhat[1] if not encoder else yhat[0]

            l = 0

            for i, func in enumerate(lossFuncs[0]):
                if weights[0][i] == 0:
                    continue

                l += weights[0][i] * func(segmentYHat, y1)

            for i, func in enumerate(lossFuncs[1]):
                if weights[1][i] == 0:
                    continue

                l += weights[1][i] * func(classYHat, y2)

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

        valLosses = evaluate(net, testIter, lossFuncs, device=device, encoder=encoder)

        validationLoss = 0

        logStr = ""

        for i, arr in enumerate(valLosses):
            for j, val in enumerate(arr):
                validationLoss += weights[i][j] * val

                logStr += (lossFuncs[i][j].__name__ if str(type(lossFuncs[i][j])) == "<class 'function'>" else type(lossFuncs[i][j]).__name__) + ": " + str(val) + " "

        #Overwrites previous best model based on validation accuracy
        if validationLoss < bestValLoss:
            bestValLoss = validationLoss
            torch.save(net.state_dict(), modelFileName + "BestLoss")

        print(f"Epoch {epoch}:\nTrain Loss: {loss / numBatches} Validation Loss: {validationLoss} " + logStr)

        #Externally logs epoch info to WandB
        if useWandB:
            wandb.log({"Train Loss": loss / numBatches,
                    "Validation Loss": validationLoss
                    })