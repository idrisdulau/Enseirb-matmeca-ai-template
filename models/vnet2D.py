# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to train a Vnet2D model.
# 
# This file is composed by 1 class and 2 functions:
# 
# Vnet2D(): provides a Vnet2D model 
# trainModel() will:
# - Call imageToTensorDataset() (to convert images to tensors)
# - Call tensorToPatches2DDataset() (to convert tensors to patches)
# - Call DiceLoss() (to use the Dice Loss with the model)
# Train model on input parameters
#
# trainModel(): Returns a trained model obtained using both training and validation steps
#
# run(): performs a training or validation step
# endregion

import torch
import tqdm
import numpy
import random
import statistics
import matplotlib.pyplot

from utils.datasets.imageToTensorDataset import imageToTensorDataset
from utils.datasets.tensorToPatches2DDataset import tensorToPatches2DDataset

class Vnet2D(torch.nn.Module):
    def __init__(self):
        super(Vnet2D, self).__init__()

        self.enco3 = Vnet2D.block(1,   64)
        self.enco2 = Vnet2D.block(64,  128)
        self.enco1 = Vnet2D.block(128, 256)

        self.bottleneck = Vnet2D.block(256, 512)

        self.upconv1 = Vnet2D.upconvBlock(512, 256)
        self.upconv2 = Vnet2D.upconvBlock(256, 128)
        self.upconv3 = Vnet2D.upconvBlock(128, 64)

        self.deco1 = Vnet2D.block(512, 256)
        self.deco2 = Vnet2D.block(256, 128)
        self.deco3 = Vnet2D.block(128, 64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.pool(enco3))
        enco1 = self.enco1(self.pool(enco2))

        bottleneck = self.bottleneck(self.pool(enco1))

        tmp = self.upconv1(bottleneck)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco3)))

    #region Blocks
    def block(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv2d(inChan, outChan, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.PReLU(),
        torch.nn.Conv2d(outChan, outChan, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.PReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inChan, outChan, kernel_size=2, stride=2))  
    #endregion Blocks

def trainModel(model, epochs, inputArg, targetArg, device, saveArg, outputName):

    #region parameters initialization
    model = model.to(device)
    from utils.losses.diceLoss import DiceLoss
    diceLoss = DiceLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    
    valSplit = 0.2
    tensorDataset = imageToTensorDataset(inputArg, targetArg)
    idxList = list(range(len(tensorDataset)))

    print("Start training with",model.__class__.__name__,":")
    print('Optimizer=%s(lr=%f), Epochs=%d, Device=%s\n' %
    (type(optimizer).__name__,optimizer.param_groups[0]['lr'], epochs, device))

    trainingMeanLoss = []
    validationMeanLoss = []
    #endregion

    for epoch in tqdm.tqdm(range(1, epochs+1)): #For each epoch

        #region shuffle sets
        random.shuffle(idxList)
        shuffledIdxList = idxList

        split = int(numpy.floor(valSplit*len(tensorDataset)))
        valIdxList = shuffledIdxList[:split]
        trainIdxList = shuffledIdxList[split:]

        trainTensorDataloader = torch.utils.data.DataLoader(dataset=tensorDataset, batch_size=1, shuffle=False, sampler=trainIdxList, num_workers=0)
        valTensorDataloader = torch.utils.data.DataLoader(dataset=tensorDataset, batch_size=1, shuffle=False, sampler=valIdxList, num_workers=0)
        #endregion 

        run("Training", trainTensorDataloader, model, optimizer, device, diceLoss, trainingMeanLoss, epoch, epochs)

        run("Validation", valTensorDataloader, model, optimizer, device, diceLoss, validationMeanLoss, epoch, epochs)

    #region final print
    if(len(trainingMeanLoss) < 10):
        print("\nMean Training Loss of last %d epochs: %6.4f | Mean Validation Loss of last %d epochs: %6.4f " \
        % ( len(trainingMeanLoss) , statistics.mean(trainingMeanLoss[::]), len(validationMeanLoss) , statistics.mean(validationMeanLoss[::]) ) )
    else:
        print("\nMean Training Loss of last 10 epochs: %6.4f | Mean Validation Loss of last 10 epochs: %6.4f " \
        % ( statistics.mean(trainingMeanLoss[-10:]), statistics.mean(validationMeanLoss[-10:]) ) )
    #endregion 
  
    #region final plot
    epochList = range(1, len(trainingMeanLoss)+1)
    matplotlib.pyplot.plot(epochList, trainingMeanLoss, 'b', label='Training loss')
    matplotlib.pyplot.plot(epochList, validationMeanLoss, 'r', label='Validation loss')
    matplotlib.pyplot.title('Training and Validation loss')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(saveArg + outputName + '.png')
    #endregion

    return model

def run(mode, tensorDataloader, model, optimizer, device, currentLoss, meanLossList, epoch, epochs):
    assert mode=="Training" or mode=="Validation"

    #region variable initialization
    epochLoss = 0.0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    foregroundPixels = 0.0
    processedPatches = 0
    #endregion

    #region defines torch mode to en/disable specific layers
    if mode == "Training":
        model.train()
    elif mode == "Validation":
        model.eval()
    #endregion

    #region processes each patch from each images
    for batch in tensorDataloader: 

        patchesDataset = tensorToPatches2DDataset(batch[0],batch[1])
        patchesDataloader = torch.utils.data.DataLoader(dataset=patchesDataset, batch_size=1, shuffle=False)
        
        for patch in patchesDataloader:
            optimizer.zero_grad() # Sets the gradients of all optimized torch.Tensors to zero.
            xTrue = patch[0].to(device)
            yTrue = patch[1].to(device)

            processedPatches += 1
            yPred = model(xTrue)
            loss = currentLoss(yPred, yTrue)
            if mode == "Training":
                loss.backward()
                optimizer.step() # Performs a single optimization step
            epochLoss += loss.item()

            #region computes a confusion matrix for a 2 classes case
            normyPred = torch.where(yPred > 0.5, torch.ones(yPred.size()).cuda(), torch.zeros(yPred.size()).cuda())
            conf = normyPred/yTrue
            tp += torch.sum(conf == 1).item()
            fp += torch.sum(conf == float('inf')).item()
            tn += torch.sum(torch.isnan(conf)).item()
            fn += torch.sum(conf == 0).item()
            foregroundPixels += (torch.sum(yTrue).item())
            #endregion

    epochLoss = epochLoss / (len(patchesDataloader)*len(tensorDataloader))  
    meanLossList.append(epochLoss)
    #endregion

    #region command line display of metrics informations
    print()
    print("Epoch %3d/%3d, "+mode+"loss: %6.4f" % (epoch, epochs, epochLoss))
    print("Average per patch(%d) tp: %d | fp: %d | fn: %d | tn: %d | tp+tn+fp+fn: %d | tp+fn: %d | tp Expected: %d"  % \
    (processedPatches, tp/processedPatches , fp/processedPatches , fn/processedPatches , tn/processedPatches, (tp+tn+fp+fn)/processedPatches, (tp+fn)/processedPatches, int(foregroundPixels/processedPatches) ) )
    #endregion
