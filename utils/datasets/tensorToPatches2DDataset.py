# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to transform normalized tensors couples,
# to store tensors patches.
# 
# This file is composed by 1 class:
# 
# tensorToPatches2DDataset(): Custom class inheriting from Dataset() class, must implement:
# - __init__()     (class constructor)
# - __getitem__()  (loads and return a list of tensors patches)
# - __len__()      (returns the number of couples in the list)
# endregion

import torch
import sys

class tensorToPatches2DDataset(torch.utils.data.Dataset):

    def __init__(self, tensorSource, tensorTarget):
        super(tensorToPatches2DDataset, self).__init__()
        self.tensorSource = tensorSource
        self.tensorTarget = tensorTarget

        self.imageHeight = self.tensorSource[0].shape[self.tensorSource[0].dim()-1]
        self.kernelHeight = 512 
        self.strideHeight = 512 
        self.numberOfPatchesInHeight = self.imageHeight//(self.kernelHeight)
        if (self.imageHeight%self.numberOfPatchesInHeight != 0):
            print("Patches height doesn't fulfill image height, the patch height value is not a divisor of the image height value")
            sys.exit()

        self.imageWidth = self.tensorSource[0].shape[self.tensorSource[0].dim()-2]
        self.kernelWidth = 512
        self.strideWidth = 512
        self.numberOfPatchesInWidth = self.imageWidth//(self.kernelWidth)
        if (self.imageWidth%self.numberOfPatchesInWidth != 0):
            print("imageWidth:",self.imageWidth)
            print("Patches width doesn't fulfill image width, the patch width value is not a divisor of the image width value")
            sys.exit()

        self.numberOfPatches = self.numberOfPatchesInHeight*self.numberOfPatchesInWidth
        self.length = self.numberOfPatches

    def __getitem__(self, index):
        source = self.tensorSource[0]
        target = self.tensorTarget[0]

        sourcePatches = source.unfold(1, self.kernelHeight, self.strideHeight).unfold(2, self.kernelWidth, self.strideWidth)
        sourcePatches = sourcePatches.contiguous().view(-1, sourcePatches.size(0), self.kernelHeight, self.kernelWidth)

        targetPatches = target.unfold(1, self.kernelHeight, self.strideHeight).unfold(2, self.kernelWidth, self.strideWidth)
        unfoldShape = targetPatches.size()
        targetPatches = targetPatches.contiguous().view(-1, targetPatches.size(0), self.kernelHeight, self.kernelWidth)

        return [sourcePatches[index], targetPatches[index], unfoldShape]

    def __len__(self):
        return self.length
