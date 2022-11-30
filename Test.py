import torch
import os
import sys
import pickle
from Utilities import DataManagerPytorch as DMP
from Utilities.ModelPlus import ModelPlus, Ensemble_Efficient
#from GaME import Load_Defenses
import numpy as np
from Automate import GetLoadingParams, LoadData, LoadModel, CreateTransWrapper, GetAllPaths

from Defenses.Vanilla import TransformerModels, BigTransferModels
from Defenses.Vanilla.ResNet import resnet164
from Defenses.SNN.resnet_spiking import RESNET_SNN_STDB
from Defenses.BaRT.ValidateBart import BaRTWrapper
from Defenses.TIT.optdefense import OptNet
from Defenses.Vanilla.vgg import vggethan

from torchvision import transforms
import ray

def Load_Defenses(files_final, dataset = "cifar10"):
    if dataset == "cifar10":
        labels  = 10
    elif dataset == "cifar100":
        labels = 100
    elif dataset == "tiny":
        labels = 200

    models = []
    for file in files_final:
        print(file)

        modelType, params, trans = GetLoadingParams(file, dataset)
        if str(modelType) == "-1":
            continue
        if len(params)==0:
            p_str = ""
        else:
            p_str = "-" + str(params[0])

        if "TiT" in file:
            if "BiT" in file:
                #print("BiT")
                resize = None #transforms.Resize((224, 224))
                advpath = "SavedModels/" + "/Vanilla/BiT-M-R50x1.tar"
                basepath = "SavedModels/" + "/Vanilla/ViT-L_16.bin"
                advnet, _, bs2V = LoadModel(advpath, "BiT", dataset, ["TiT"])
                model, size, bs = LoadModel(basepath, "ViT", dataset, ["TiT"])
                #optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                trans = lambda x : OptNet(None, advnet, .2, .02, 13, True, resize = resize).adversary_batch(x ,bs2V)
            elif "Res" in file:
                #print("Res")
                resize = None #transforms.Resize((224, 224))
                data = torch.load(file)
                advnet2 = vggethan(dataset)
                advnet2.load_state_dict(data["adv"])

                model = resnet164(32, labels)
                model.load_state_dict(data["base"])
                bs2R, bs = 32, 32
                size = (32, 32)

                #optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                trans = lambda x : OptNet(None, advnet2, .2, .02, 13, True, resize = resize).adversary_batch(x ,bs2R)
        else:
            model, size, bs = LoadModel(file, modelType, dataset, params)
        resize = transforms.Resize((size[0], size[1]))
        trans2 = CreateTransWrapper(trans, resize)
    
        modelP = ModelPlus(modelType + p_str, model, "cuda", size[0], size[1], bs, normalize = trans2, n_classes = labels, jelly = "jelly" in modelType)
        models.append(modelP)
    return models

xadv = {}
yadv = {}
inc = {}
for i in range(2):
    data = torch.load("xAdvTransfer-" + str(i))
    #print(data.keys())
    #continue
    for key in data.keys():
        if key.count("-") > 2:
            continue
        try:
            data[key]['x']
        except:
            continue
        #print(data[key])
        #continue
        xadv[key] = data[key]["x"]
        yadv[key] = data[key]["y"]
        inc[key] = data[key]["i"]
    del(data)

choices = {}
for i in range(2):
    data = torch.load("Results-xadv-cifar10-" + str(i))
    print(data.keys())
    for key in data.keys():
        try:
            data[key]['x']
            continue
        except:
            pass
        choices[key] = data[key]
    del(data)

ray.init(num_cpus = 5, log_to_driver = False)
files_final = ['SavedModels/BaRT/BaRT-1-101x3', 'SavedModels/BaRT/BaRT-5-101x3', 'SavedModels/BaRT/BaRT-10-101x3', 'SavedModels/FAT/FAT-resnet164.tar', 'SavedModels/FAT/ViT-L_16_tau2.bin', 'SavedModels/SNN/snn_resnet20_cifar10_10_1.pth', 'SavedModels/SNN/jelly_resnet18acc_8183.pt', 'SavedModels/TiT/BiTViT.txt', "SavedModels/TiT/VGG-Res-TiT.pth"]
defenses = Load_Defenses(files_final)
arr = []
for d in defenses:
    xcur = xadv[d.modelName][inc[d.modelName].bool()]
    ycur = yadv[d.modelName][inc[d.modelName].bool()]
    loader = DMP.TensorToDataLoader(xcur, ycur, batchSize = d.batchSize)
    acc = d.validateD(loader)
    arr.append(acc)
    print(acc)

print(arr)

for i in range(len(arr)):
    print(files_final[i], arr[i])

i=j=0
arr = np.zeros(shape = (len(defenses), len(defenses)))
ray.init(num_cpus = 5, log_to_driver = False)
for d1 in defenses:
    for d2 in defenses:
        if d1.modelName == d2.modelName:
            j += 1
            continue
        if d1.modelName + d2.modelName in choices.keys():
            xcur = xadv[d1.modelName][choices[d1.modelName + d2.modelName].bool()]
            ycur = yadv[d1.modelName][choices[d1.modelName + d2.modelName].bool()]
            loader = DMP.TensorToDataLoader(xcur, ycur, batchSize = d2.batchSize)
            arr[i,j] = d2.validateD(loader)
            print(d1.modelName, d2.modelName, arr[i,j])
        else:
            print(d1.modelName + d2.modelName, "Failed")

        torch.cuda.empty_cache()
        j += 1
    j = 0
    i += 1
ray.shutdown()

#np.savetxt("TransferResults.csv", arr, delimiter = ",")


