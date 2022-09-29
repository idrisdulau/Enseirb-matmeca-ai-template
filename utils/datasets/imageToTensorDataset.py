# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to store and transform images samples/labels couples,
# to normalized tensors couples list.
# 
# This file is composed by 1 class and 1 function:
# 
# imageToTensorDataset(): Custom class inheriting from Dataset() class, must implement:
# - __init__()     (class constructor)
# - __getitem__()  (loads and return a list of normalized tensor couples corresponding to samples/labels input images)
# - __len__()      (returns the number of couples in the list)
#
# crop(): Return a cropped image from input dimensions
# endregion

import torch 
import torchvision
import os 
import PIL.Image

class imageToTensorDataset(torch.utils.data.Dataset):
    
    def __init__(self, inputArg, targetArg, subImage="noCrop"): #"l-t-r-b" or "noCrop"
        super(imageToTensorDataset, self).__init__()
        self.sourceDir = inputArg
        self.targetDir = targetArg
        self.subImage = subImage
        
        _, self.extension = os.path.splitext(os.listdir(self.sourceDir)[0])

        #region ensures the couples correspondence
        sourceList = os.listdir(self.sourceDir)
        sourceIdxList = []
        for i in range (len(sourceList)):
            idx, _ = os.path.splitext(sourceList[i])
            sourceIdxList.append(int(idx))
        sourceIdxList.sort() 

        targetList = os.listdir(self.targetDir)
        targetIdxList = []
        for i in range (len(targetList)):
            idx, _ = os.path.splitext(targetList[i])
            targetIdxList.append(int(idx))
        targetIdxList.sort() 

        notBothIdxList = []
        for i in range (len(sourceIdxList)-1):
            if sourceIdxList[i] not in targetIdxList:
                notBothIdxList.append(i)
        for i in range (len(targetIdxList)-1):
            if targetIdxList[i] not in sourceIdxList:
                notBothIdxList.append(i)
        #endregion

        self.itemToExclude = [str(item)+self.extension for item in notBothIdxList]
        sourceList.sort()
        self.sourceValidCouples = [item for item in sourceList if item not in self.itemToExclude]
        targetList.sort()
        self.targetValidCouples = [item for item in targetList if item not in self.itemToExclude]

        self.length = len(self.sourceValidCouples)
        # self.length = 5
        print("self.length:",self.length)
        print()

    def __getitem__(self, index):
        sourceItem = self.sourceValidCouples[index]
        sourcePath = self.sourceDir.replace(" ", "")
        sourceFile = os.path.join(sourcePath, sourceItem)
        sourceImage = PIL.Image.open(sourceFile)
        sourceImage = crop(sourceImage, self.subImage)
        sourceImage = PIL.ImageOps.grayscale(sourceImage)
        sourceTensor = torchvision.transforms.ToTensor()(sourceImage)

        targetItem = self.targetValidCouples[index]
        targetPath = self.targetDir.replace(" ", "")
        targetFile = os.path.join(targetPath, targetItem)
        targetImage = PIL.Image.open(targetFile)
        targetImage = crop(targetImage, self.subImage)
        targetImage = PIL.ImageOps.grayscale(targetImage)
        targetTensor = torchvision.transforms.ToTensor()(targetImage)

        return [sourceTensor, targetTensor, sourceItem]

    def __len__(self):
        return self.length

def crop(image, subImage):
    if subImage == "noCrop":
        return image
    else:
        l,t,r,b = [int(e) for e in (subImage.split("-"))]
        return image.crop((l, t, r, b))
