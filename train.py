# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to parse command line arguments, to further, 
# train a model on user input parameters and save its weights to a file for lately predicting.
# 
# This file is composed by 3 functions:
# 
# usage(): To display when an error occurs in command line argument parsing.  
# parsing(): Parse and store the command line arguments.
# main(): Use the stored arguments to:
# - Call a model function (to perform training subsequently with stored parsed arguments)
# Save the trained model file. 
#endregion

import sys          
import os           
import subprocess   
import torch        
import datetime     

def usage():
    print("\nYou should provide these options and arguments: \n")
    print(" Options       Arguments \n")
    print(" --input       path to RGB input data folder")
    print(" --target      path to segmented data folder")
    print(" --save        path to save the model, directories are created if needed")
    print(" --modelType   Vnet2D")
    print(" --epochs      positive integer")
    print(" --gpu         positive integer")

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
        
    #region parsing: following regions parse command line options and related arguments.

    #region input
    inputOpt = "--input"
    if optionsList.count(inputOpt):
        idx = optionsList.index(inputOpt)
    else :
        print("You should provide an input option as follows")
        print(" --input       path to RGB input data folder")
        usage()
        sys.exit()
    inputArg = argumentsList[idx]

    if not os.path.isdir(inputArg):
        print(inputArg+" is not a correct path")
        usage()
        sys.exit()
    #endregion input

    #region target
    targetOpt = "--target"
    if optionsList.count(targetOpt):
        idx = optionsList.index(targetOpt)
    else :
        print("You should provide a target option as follows")
        print(" --target      path to segmented data folder")
        usage()
        sys.exit() 
    targetArg = argumentsList[idx]

    if not os.path.isdir(targetArg):
        print(targetArg+" is not a correct path")
        usage()
        sys.exit() 
    #endregion target

    #region save
    saveOpt = "--save"
    if optionsList.count(saveOpt):
        idx = optionsList.index(saveOpt)
    else :
        print("You should provide a save option as follows")
        print(" --save        path to save the model, directories are created if needed")
        usage()
        sys.exit() 
    saveArg = argumentsList[idx]

    if not os.path.exists(os.path.join(os.getcwd(), saveArg)):
        subprocess.run(["mkdir", "-p", saveArg])

    if not os.path.isdir(saveArg):
        print(saveArg+" is not a correct path")
        usage()
        sys.exit() 
    #endregion save

    #region modelType
    modelTypeOpt = "--modelType"
    if optionsList.count(modelTypeOpt):
        idx = optionsList.index(modelTypeOpt)
    else :
        print("You should provide a modelType option as follows")
        print(" --modelType   Vnet2D")
        usage() 
        sys.exit()
    modelTypeArg = argumentsList[idx]

    if (modelTypeArg != "Vnet2D"):
        print(modelTypeArg+" is not a correct model type")
        usage()
        sys.exit() 
    #endregion modelType

    #region epochs
    epochsOpt = "--epochs"
    if optionsList.count(epochsOpt):
        idx = optionsList.index(epochsOpt)
    else :
        print("You should provide an epochs option as follows")
        print(" --epochs      positive integer")
        usage()
        sys.exit() 
    epochsArg = argumentsList[idx]

    if not (epochsArg.isdigit() and int(epochsArg) > 0):
        print(epochsArg+" is not a correct number of epochs")
        usage()
        sys.exit() 
    epochsArg = int(epochsArg)
    #endregion epochs

    #region gpu
    gpuOpt = "--gpu"
    if optionsList.count(gpuOpt):
        idx = optionsList.index(gpuOpt)
    else :
        print("You should provide a gpu option as follows")
        print(" --gpu         positive integer")
        usage()
        sys.exit() 
    gpuArg = argumentsList[idx]

    if not (gpuArg.isdigit() and int(gpuArg) >= 0):
        print(gpuArg+" is not a correct gpu number")
        usage()
        sys.exit() 
    #endregion gpu

    #endregion parsing

    return inputArg, targetArg, saveArg, modelTypeArg, epochsArg, gpuArg 

def main(argv):

    inputArg, targetArg, saveArg, modelTypeArg, epochsArg, gpuArg = parsing(argv)

    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuArg

    date, time = str(datetime.datetime.now()).split()
    MDHMS = date.split("-")[1]+date.split("-")[2]+"_"+time.split(".")[0]

    if modelTypeArg == "Vnet2D":
        from models.vnet2D import Vnet2D, trainModel
        outputName = modelTypeArg + ":" + str(epochsArg) + "Ep:"
        model = trainModel(Vnet2D(), epochsArg, inputArg, targetArg, device, saveArg, outputName)
        torch.save(model.state_dict(), os.path.join(saveArg, MDHMS + outputName + ".pt"))

if __name__ == '__main__':
    main(sys.argv)
