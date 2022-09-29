# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is reconstruct normalized tensors couples from tensors patches.
# 
# This file is composed by 1 class:
# 
# patchesToTensor2DDataset(): Custom class inheriting from Dataset() class, must implement:
# - __init__()     (class constructor)
# - __getitem__()  (loads and return a list of tensors reconstructed from stored patches shape)
# - __len__()      (returns the number of couples in the list)
# endregion

import torch

class patchesToTensor2DDataset(torch.utils.data.Dataset):

    def __init__(self, patches, unfoldShape):
        super(patchesToTensor2DDataset, self).__init__()
        self.patches = patches
        self.unfoldShape = unfoldShape
        self.length = 1

    def __getitem__(self, index):      
        sourcePatches = self.patches
        sourceMergedPatches = sourcePatches.view(self.unfoldShape)
        batch = self.unfoldShape[0]
        outputH = self.unfoldShape[1] * self.unfoldShape[3]
        outputW = self.unfoldShape[2] * self.unfoldShape[4]
        sourceMergedPatches = sourceMergedPatches.permute(0, 1, 3, 2, 4).contiguous()
        source = sourceMergedPatches.view(batch, outputH, outputW)
        return [source]

    def __len__(self):
        return self.length
