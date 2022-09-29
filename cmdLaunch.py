# ------------------------------------------------------------------- #
# ------------------- File written by Idris DULAU ------------------- #
# ------------------------------------------------------------------- #

#region FILE UNDERSTANDING
# The main purpose of this file is to more easily and consecutively launch multiple training and prediction programs using command line arguments.
# Some examples working on deep1 in the LaBRI are provided.
#endregion

import subprocess

print("Train on IOSTAR")
subprocess.run(["python3", "train.py", \
    "--input",      "../data/IOSTAR_MOD/img/", \
    "--target",     "../data/IOSTAR_MOD/gt/", \
    "--save",       "save/io1", \
    "--modelType",  "Vnet2D", \
    "--epochs",     "100", \
    "--gpu",        "0"])

# print("Predict on IOSTAR")
# subprocess.run(["python3", "predict.py", \
#     "--input",          "../data/IOSTAR_MOD/imgEval", \
#     "--outputFolder",   "save/", \
#     "--modelType",      "Vnet2D", \
#     "--modelLocation",  "save/io1/Vnet2D_exemple.pt", \
#     "--gpu",            "0"])
