# ============== Click Please.Imports
import sys
import glob
import gc


import random
random.seed(0)
import pandas as pd

import torch
import seaborn as sns

import matplotlib.pyplot as plt

import glob
import sys
import shutil
import tqdm



# ================
# Torch related
# ==============
import torch 


# Turn on cuda optimizer
print(torch.backends.cudnn.is_available())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# disable debugs NOTE use only after debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
# Disable gradient tracking
torch.no_grad()


# =============
# NN
# =================
sys.path.append('../')
"""
from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandReadPdbFtp import ReadBCExternalSymmetry, MakeBcClanGraph
from NucleicNet.DatasetBuilding.commandDataFetcher import FetchIndex, FetchTask, FetchDataset
from NucleicNet.DatasetBuilding.commandBenchmark import BenchmarkWrapper
from NucleicNet import Burn, Fuel
import NucleicNet.Burn.M1
import NucleicNet.Burn.util
"""
from NucleicNet.DatasetBuilding.util import *



#%config InlineBackend.figure_format = 'svg'

#sns.set_context("notebook")


# =====================================
# Body
# ======================================




from NucleicNet.DatasetBuilding.commandServer import Server

# NOTE This folder contains test case pdb to be sanitised.
DIR_ServerExample = '../GoogleColab/'
UploadTestCases = sorted(glob.glob("%s/*.pdb" %(DIR_ServerExample))) 


DIR_ServerFolder = "../GoogleColab/ServerOutputV1p1/" 
MkdirList([DIR_ServerFolder])
# NOTE Make some subfolders
for i in range(len(UploadTestCases)):
    try:
        pdbid = UploadTestCases[i].split("/")[-1].split(".")[0]
        print("Working on %s" %(pdbid))

        ServerSubFolder = DIR_ServerFolder + "%s/"%(pdbid)
        ServerC = Server(SaveCleansed = True, SaveDf = True,
                        Select_HeavyAtoms = True,
                        Select_Resname = [
                            "ALA","CYS","ASP","GLU","PHE","GLY", 
                            "HIS","ILE","LYS","LEU","MET","ASN", 
                            "PRO","GLN","ARG","SER","THR","VAL", 
                            "TRP","TYR"
                            ],
                        DIR_ServerFolder = ServerSubFolder,
                        DsspExecutable = "../NucleicNet/util/dssp",
                        DIR_ClassIndexCopy = "../Database-PDB/DerivedData/ClassIndex.pkl")

        
        ServerC.SimpleSanitise(aux_id=0, DIR_InputPdbFile=UploadTestCases[i])
        ServerC.MakeHalo(aux_id=0)
        ServerC.MakeDssp(aux_id=0)
        ServerC.MakeDummyTypi(aux_id=0)
        ServerC.MakeFeature()
        ServerC.MakeSXPR(aux_id=0, Checkpointlist = [   # NOTE These three models (with gelu) 
                                            "../Models/SXPR-9CV_SXPR-9CV/132_133/checkpoints/epoch=2-step=51700-hp_metric=0.5486971139907837.ckpt",
                                            "../Models/SXPR-9CV_SXPR-9CV/134_135/checkpoints/epoch=2-step=51473-hp_metric=0.5440343022346497.ckpt",
                                            "../Models/SXPR-9CV_SXPR-9CV/136_137/checkpoints/epoch=2-step=52381-hp_metric=0.4679257273674011.ckpt",

                                                        
                                                        ]
                                                )
        ServerC.MakeAUCG(aux_id=0,  Checkpointlist = [  # NOTE These three models (with gelu)
                                                        "../Models/AUCG-9CV_AUCG-9CV/294_295/checkpoints/epoch=7-step=26400-hp_metric=0.485228568315506.ckpt",
                                                        "../Models/AUCG-9CV_AUCG-9CV/296_297/checkpoints/epoch=7-step=25924-hp_metric=0.48591428995132446.ckpt",
                                                        "../Models/AUCG-9CV_AUCG-9CV/298_299/checkpoints/epoch=7-step=24326-hp_metric=0.5161714553833008.ckpt",

                                                        ]
                                                )
    except:
        print('Cuda Runtime error in %s' %(pdbid) )
        del ServerC
        continue
    # NOTE Downstream applications
    ServerC.Downstream_VisualisePse(aux_id=0)


    # NOTE Sequence Logo Making. We will simply take the centroid from uploaded structure.
    try:
        ServerC.Downstream_Logo(
                        DIR_InputPdbFile = UploadTestCases[i],
                        User_ProteinContactThreshold = 6.0,
                        User_SummarisingSphereRadius = 1.5)
    except AssertionError:
        print("ABORTED. Supply template base location for %s" %(pdbid))

    del ServerC
