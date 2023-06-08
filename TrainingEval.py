import torch
import wandb
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

def ParseConfig(configFile, vars):
    file = open(configFile).readlines()
    for line in file:
        if(line[0] == "#"):
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

def evaluate_accuracy(net, testIter, lossFunc, classification=True, device=None):
    net.eval()

    #Accuracy, loss
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for i, (X, y1, y2) in enumerate(testIter):
            X = X.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            yhat = net(X)

            if isinstance(yhat, tuple):
                yhat = yhat[0]

            if classification:
                loss, accuracy = lossFunc(yhat, y2)
            else:
                loss, accuracy = lossFunc(yhat, y1)

            metric.add(accuracy, loss)

    return metric[0] / len(testIter), metric[1] / len(testIter)

def joint_eval(net, testIter, classLossFunc, segmentLossFunc, device=None):
    net.eval()

    #Accuracy, loss
    metric = d2l.Accumulator(4)

    with torch.no_grad():
        for i, (X, y1, y2) in enumerate(testIter):
            X = X.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            yhat = net(X)

            classLoss, classAccuracy = classLossFunc(yhat[1], y2)
            segmentLoss, segmentAccuracy = segmentLossFunc(yhat[0], y1)

            metric.add(classAccuracy, classLoss, segmentAccuracy, segmentLoss)

    return metric[0] / len(testIter), metric[1] / len(testIter), metric[2] / len(testIter), metric[3] / len(testIter)

def train(net: nn.Module, trainIter, testIter, numEpochs, startEpoch, learnRate, batchSize, device: torch.device, startDim, epochsToDouble, modelFileName, epochsToSave, 
          useWandB=False, cosineAnnealing=True, restartEpochs=-1, progressive=False, lossFunc = nn.BCEWithLogitsLoss(), classification=True):
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

    currDim = startDim
    for epoch in range(startEpoch, numEpochs):
        net.train()
        
        #Loss, accuracy
        metric = d2l.Accumulator(2)

        for i, (X, y1, y2) in enumerate(trainIter):
            optimizer.zero_grad()
            y1 = y1.to(device)
            y2 = y2.to(device)

            if progressive > 0:
                #If using progressive learning, downsamples image to the current dimension
                X = F.interpolate(X, size=int(currDim))

            X = X.to(device)
            
            yhat = net(X)

            if isinstance(yhat, tuple):
                yhat = yhat[0]

            if classification:
                l, accuracy = lossFunc(yhat, y2)
            else:
                l, accuracy = lossFunc(yhat, y1)

            #print(f"Loss: {l.item()} Predictions: {yhat.tolist()} Labels: {y.tolist()}")

            l.backward()
            optimizer.step()

            if cosineAnnealing:
                scheduler.step(epoch + i / numBatches)

            metric.add(l, accuracy)

        #Progressive learning
        if (epoch + 1) % epochsToDouble == 0 and progressive == 1:
            currDim *= 2
        #Reverse progressive learning
        elif (epoch + 1) % epochsToDouble == 0 and progressive == 2:
            currDim /= 2

        #Checkpoints model
        if (epoch + 1) % epochsToSave == 0:
            torch.save(net.state_dict(), modelFileName + "Epoch" + str(epoch))

        validationAcc, validationLoss = evaluate_accuracy(net, testIter, lossFunc, classification=classification, device=device)

        #Overwrites previous best model based on validation accuracy
        if validationLoss < bestValLoss:
            bestValLoss = validationLoss
            torch.save(net.state_dict(), modelFileName + "BestLoss")

        print(f"Epoch {epoch}:\nTrain Acc: {metric[1] / numBatches} Validation Acc: {validationAcc} Train Loss: {metric[0] / numBatches} Validation Loss: {validationLoss}")

        #Externally logs epoch info to WandB
        if useWandB:
            wandb.log({"Train Acc": metric[1] / numBatches,
                    "Validation Acc": validationAcc,
                    "Train Loss": metric[0] / numBatches,
                    "Validation Loss": validationLoss
                    })
            
def joint_train(net: nn.Module, trainIter, testIter, numEpochs, startEpoch, learnRate, batchSize, device: torch.device, modelFileName, epochsToSave, 
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
        
        #Loss, accuracy
        metric = d2l.Accumulator(2)

        for i, (X, y1, y2) in enumerate(trainIter):
            optimizer.zero_grad()
            y1 = y1.to(device)
            y2 = y2.to(device)
            X = X.to(device)
            
            yhat = net(X)

            segmentLoss, segmentAcc = segmentLossFunc(yhat[0], y1)
            classLoss, classAcc = classLossFunc(yhat[1], y2)

            #print(f"Loss: {l.item()} Predictions: {yhat.tolist()} Labels: {y.tolist()}")

            l = (weights[0] * segmentLoss) + (weights[1] * classLoss)

            l.backward()
            optimizer.step()

            if cosineAnnealing:
                scheduler.step(epoch + i / numBatches)

            metric.add(l, segmentAcc)

        #Checkpoints model
        if (epoch + 1) % epochsToSave == 0:
            torch.save(net.state_dict(), modelFileName + "Epoch" + str(epoch))

        validationAcc, validationLoss = evaluate_accuracy(net, testIter, segmentLossFunc, classification=False, device=device)

        #Overwrites previous best model based on validation accuracy
        if validationLoss < bestValLoss:
            bestValLoss = validationLoss
            torch.save(net.state_dict(), modelFileName + "BestLoss")

        print(f"Epoch {epoch}:\nTrain Acc: {metric[1] / numBatches} Validation Acc: {validationAcc} Train Loss: {metric[0] / numBatches} Validation Loss: {validationLoss}")

        #Externally logs epoch info to WandB
        if useWandB:
            wandb.log({"Train Acc": metric[1] / numBatches,
                    "Validation Acc": validationAcc,
                    "Train Loss": metric[0] / numBatches,
                    "Validation Loss": validationLoss
                    })