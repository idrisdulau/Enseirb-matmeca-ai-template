# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to predict, reconstruct and save an image,
# using the saved trained model and the user input parameters.
# 
# This file is composed by 1 function:
# 
# buildImagesFrom2D() will:
# - Call imageToTensorDataset() (to convert images to tensors)
# - Call tensorToPatches2DDataset() (to convert tensors to patches)
# Predict on patches using saved trained model 
# - Call patchesToTensor2DDataset() (to reconstruct back tensors from patches)
# Convert and save images from reconstructed tensors
#endregion

import torch
import numpy
import PIL.Image

from utils.datasets.imageToTensorDataset import imageToTensorDataset
from utils.datasets.tensorToPatches2DDataset import tensorToPatches2DDataset
from utils.datasets.patchesToTensor2DDataset import patchesToTensor2DDataset

def buildImagesFrom2D(model, inputArg, outputFolderArg):

    tensorDataset = imageToTensorDataset(inputArg, inputArg, "noCrop")
    tensorDataloader = torch.utils.data.DataLoader(dataset=tensorDataset, batch_size=1, shuffle=False)

    for batch in tensorDataloader:
        strItem = batch[2][0]
        # strItem = "_"+strItem.split(".")[0]+".jpg"

        patchesDataset = tensorToPatches2DDataset(batch[0],batch[1])
        patchesDataloader = torch.utils.data.DataLoader(dataset=patchesDataset, batch_size=1, shuffle=False)
        unfoldShape = patchesDataset[0][2]

        batchPatch = len(patchesDataset)
        channelPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-3)
        heightPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-2)
        widthPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-1)
        zeroSubImage = torch.zeros(batchPatch, channelPatch, heightPatch, widthPatch)

        for i, patch in enumerate(patchesDataloader):
            print("len",patch[2])
            with torch.no_grad():
                zeroSubImage[i] = model(patch[0].float().cuda())                             

        outputDataset = patchesToTensor2DDataset(zeroSubImage, unfoldShape)
        outputDataloader = torch.utils.data.DataLoader(dataset=outputDataset, batch_size=1, shuffle=False)

        for tensor in outputDataloader:
            output = tensor[0][0].detach().numpy()
            output = numpy.where(output > 0.8, 1, 0)
            output = output*255                         
            print("\nForeground pixels: %d | %4.2f%s of the image pixels " % (numpy.count_nonzero(output), 100*numpy.count_nonzero(output)/(batchPatch*heightPatch*widthPatch), "%" ) )
            output = output.astype(numpy.uint8)                     
            PIL.Image.fromarray(output[0]).save(outputFolderArg + strItem)
