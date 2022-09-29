# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to parse command line arguments, to further, 
# predict, reconstruct and save an image using the saved trained model and the user input parameters.
# 
# This file is composed by 3 functions:
# 
# usage(): To display when an error occurs in command line argument parsing.  
# parsing(): Parse and store the command line arguments.
# main(): Use the stored arguments to:
# Load the saved model
# - Call a buildImages function (to perform image prediction, reconstruction and save with stored parsed arguments)
#endregion

import sys          
import os           
import subprocess   
import torch        

def usage():
    print("\nYou should provide these options and arguments: \n")
    print(" Options           Arguments \n")
    print(" --input           path to input data folder")
    print(" --outputFolder    path to the output folder to save the newly generated segmented files")
    print(" --modelType       Vnet2D")
    print(" --modelLocation   path to selected model file, including file name")
    print(" --gpu             positive integer")

def parsing(argv):

    #Save each pair of Option Argument in two dedicated list, 
    #   assuming each option is followed by its argument.
    #More precise sanity checks will be performed afterwards. 
    optionsList = argv[1::2]
    argumentsList = argv[2::2] 

    if (len(argv) == 1):
        usage() 
        sys.exit()

    #Check the parity of command line length based on the previously stored lists.
    if( len(optionsList) != len(argumentsList) ):
        print("Lengths of options list and arguments list are differents.")
        usage() 
        sys.exit()
        
    #region parsing: Following regions parse command line options and related arguments.

    #region input
    inputOpt = "--input"
    if optionsList.count(inputOpt):
        idx = optionsList.index(inputOpt)
    else :
        print("You should provide an input option as follows")
        print(" --input           path to input data folder")
        usage()
        sys.exit()
    inputArg = argumentsList[idx]

    if not os.path.isdir(inputArg):
        print(inputArg+" is not a correct path")
        usage()
        sys.exit()
    #endregion input

    #region outputFolder
    outputFolderOpt = "--outputFolder"
    if optionsList.count(outputFolderOpt):
        idx = optionsList.index(outputFolderOpt)
    else :
        print("You should provide an output folder option as follows")
        print(" --outputFolder    path to the output folder to save the newly generated segmented files")
        usage()
        sys.exit() 
    outputFolderArg = argumentsList[idx]

    if not os.path.exists(os.path.join(os.getcwd(), outputFolderArg)):
        subprocess.run(["mkdir", "-p", outputFolderArg])

    if not os.path.isdir(outputFolderArg):
        print(outputFolderArg+" is not a correct path")
        usage()
        sys.exit() 
    #endregion outputFolder

    #region modelType
    modelTypeOpt = "--modelType"
    if optionsList.count(modelTypeOpt):
        idx = optionsList.index(modelTypeOpt)
    else :
        print("You should provide a modelType option as follows")
        print(" --modelType       Vnet2D")
        usage() 
        sys.exit()
    modelTypeArg = argumentsList[idx]

    if (modelTypeArg != "Vnet2D"):
        print(modelTypeArg+" is not a correct model type")
        usage()
        sys.exit() 
    #endregion modelType

    #region modelLocation
    modelLocationOpt = "--modelLocation"
    if optionsList.count(modelLocationOpt):
        idx = optionsList.index(modelLocationOpt)
    else :
        print("You should provide a model location option as follows")
        print(" --modelLocation   path to selected model file, including file name")
        usage()
        sys.exit()
    modelLocationArg = argumentsList[idx]

    if not os.path.isfile(modelLocationArg):
        print(modelLocationArg+" is not a correct path")
        usage()
        sys.exit()
    #endregion modelLocation

    #region gpu
    gpuOpt = "--gpu"
    if optionsList.count(gpuOpt):
        idx = optionsList.index(gpuOpt)
    else :
        print("You should provide a gpu option as follows")
        print(" --gpu             positive integer")
        usage()
        sys.exit() 
    gpuArg = argumentsList[idx]

    if not (gpuArg.isdigit() and int(gpuArg) >= 0):
        print(gpuArg+" is not a correct gpu number")
        usage()
        sys.exit() 
    #endregion gpu

    #endregion parsing

    return inputArg, outputFolderArg, modelTypeArg, modelLocationArg, gpuArg  

def main(argv):

    inputArg, outputFolderArg, modelTypeArg, modelLocationArg, gpuArg = parsing(argv)

    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuArg

    if modelTypeArg == "Vnet2D":
        from models.vnet2D import Vnet2D
        model = Vnet2D() 
        model.load_state_dict(torch.load(modelLocationArg))
        model.to(device)
        from utils.build.buildImagesFrom2D import buildImagesFrom2D
        buildImagesFrom2D(model, inputArg, outputFolderArg)   

if __name__ == '__main__':
    main(sys.argv)
