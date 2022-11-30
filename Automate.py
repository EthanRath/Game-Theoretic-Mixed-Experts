#External Imports
from ast import Continue
import torch
from torchvision import transforms
from spikingjelly.clock_driven.model import spiking_resnet, sew_resnet
from spikingjelly.clock_driven import neuron, surrogate
import numpy as np
import itertools
import random
import pickle
import psutil
import ray

#System Imports
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

#Internal Imports
from Utilities import DataManagerPytorch as DMP
from Utilities.ModelPlus import ModelPlus, Ensemble_Efficient

from Defenses.Vanilla import TransformerModels, BigTransferModels
from Defenses.Vanilla.ResNet import resnet164
from Defenses.SNN.resnet_spiking import RESNET_SNN_STDB
from Defenses.BaRT.ValidateBart import BaRTWrapper
from Defenses.TIT.optdefense import OptNet
from Defenses.Vanilla.vgg import vggethan

from Attacks.attack_utilities import GetFirstCorrectlyOverlappingSamplesBalanced, GetFirstCorrectlyOverlappingSamplesBalancedSingle
from Attacks.AttackWrappersProtoSAGA import SelfAttentionGradientAttack_EOT, MIM_EOT_Wrapper, AutoAttackNativePytorch, MIMNativePytorch_cnn
from Attacks.AttackWrappersWhiteBoxSNN import PGDNativePytorch as PGDSNN
from Attacks.AttackWrappersWhiteBoxJelly import PGDNativePytorch as PGDJelly
from Attacks.AttackWrappersWhiteBoxSNN import MIMNativePytorch as MIMSNN
from Attacks.AttackWrappersWhiteBoxJelly import MIMNativePytorch as MIMJelly

#Gets all file paths to each defense
def GetAllPaths(root = "SavedModels/"):
    arr = []
    files = os.listdir(root) #Lists folders
    for file in files:
        infiles = os.listdir(root + file + "/") #lists model files inside of folders
        for ifile in infiles:
            arr.append(root + file + "/" + ifile)
    return arr

#Loads the data
def LoadData(modelType, dataset, bs):
    if "ViT" in modelType:
        h = w = 224
        norm = True
    elif "BiT" in modelType:
        if "BaRT" in modelType:
            h = w = 128
        else:
            h = 160
            w = 128
        norm = True
    elif "jelly" in modelType:
        h = w = 32
        norm = True
    else:
        h = w = 32
        norm = True
    if dataset == "cifar10":
        test_loader = DMP.get_CIFAR10_loaders_test(img_size_H = h, img_size_W = w, bs = bs, norm = norm)
    elif dataset == "cifar100":
        test_loader = DMP.get_CIFAR100_loaders_test(img_size_H = h, img_size_W = w, bs = bs)
    return test_loader

#Loads each model
def LoadModel(file, modelType, dataset, params = []):
    if dataset == "cifar10":
        classes = 10
    elif dataset == "cifar100":
        classes = 100
    else:
        classes = 200

    if modelType == "resnet164":
        data = torch.load(file)
        if len(params) > 0 and "FAT" in params[0]:
            dict = DMP.Fix_Dict(data["state_dict"])
            size = (32,32)
        else:
            dict = DMP.Fix_Dict(data["model"])
            size = (32,32)
        model = resnet164(size[0], classes)
        model.load_state_dict(dict)
        bs = 8

    if modelType == 'transfer-snn':
        model = model = RESNET_SNN_STDB(resnet_name = "resnet20", labels = classes, dataset = dataset.upper())
        state = torch.load(file)#torch.load(pretrained_snn, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state["state_dict"], strict=False)
        model.network_update(timesteps=int(params[0]), leak=1.0)
        #model = model.to("cuda")
        size = (32,32)
        bs = 32
    if modelType == "jelly":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sg = 'ATan'
        argsNeuron = 'MultiStepParametricLIFNode'
        arch = 'sew_resnet18'
        if dataset == "cifar10":
            timeStep = 4
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
            timeStep = 5
        surrogate_gradient = {
                            'ATan' : surrogate.ATan(),
                            'Sigmoid' : surrogate.Sigmoid(),
                            'PiecewiseLeakyReLU': surrogate.PiecewiseLeakyReLU(),
                            'S2NN': surrogate.S2NN(),
                            'QPseudoSpike': surrogate.QPseudoSpike()
                        }
        sg_type = surrogate_gradient[sg]
        neuron_dict = {
            'MultiStepIFNode'               : neuron.MultiStepIFNode,
            'MultiStepParametricLIFNode'    : neuron.MultiStepParametricLIFNode,
            'MultiStepEIFNode'              : neuron.MultiStepEIFNode,
            'MultiStepLIFNode'              : neuron.MultiStepLIFNode,
        }
        neuron_type = neuron_dict[argsNeuron]
        model_arch_dict = {
                        'sew_resnet18'       : sew_resnet.multi_step_sew_resnet18, 
                        'sew_resnet34'       : sew_resnet.multi_step_sew_resnet34, 
                        'sew_resnet50'       : sew_resnet.multi_step_sew_resnet50,
                        'spiking_resnet18'   : spiking_resnet.multi_step_spiking_resnet18, 
                        'spiking_resnet34'   : spiking_resnet.multi_step_spiking_resnet34, 
                        'spiking_resnet50'   : spiking_resnet.multi_step_spiking_resnet50,
        }
        model_type = model_arch_dict[arch]
        model = model_type(T=timeStep, num_classes=num_classes, cnf='ADD', multi_step_neuron=neuron_type, surrogate_function=sg_type)
        dir = file
        checkpoint = torch.load(dir)
        model.load_state_dict(checkpoint["snn_state_dict"], strict=True)
        #model.to(device)
        if dataset == "cifar10":
            size = [32,32]
            bs = 24
        elif dataset == "cifar100":
            size = [32,32]
            bs = 16
    if "BiT" in modelType:
        if len(params) > 0 and "TiT" in params[0]:
            model = BigTransferModels.KNOWN_MODELS["BiT-M-R50x1"](head_size=classes, zero_head=False)
        elif "BaRT" in file:
            model = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3-BaRT"](head_size=classes, zero_head=False)
        else:
            model = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=classes, zero_head=False)
        if "BaRT" in file:
            data = torch.load(file, map_location = "cpu")
            size = (128,128)
            bs = 2
        else:
            data = torch.load(file,  map_location = "cpu")["model"]
            size = (160, 128)
            if len(params) > 0 and "TiT" in params[0]:
                size = (224,224)
                bs = 2
            else:
                bs = 2
        dic = {}
        for key in data:
            dic[key[7:]] = data[key]
        model.load_state_dict(dic)
        del(data)

    elif modelType == "ViT":
        if "ViT-L" in file:
            config = TransformerModels.CONFIGS["ViT-L_16"]
        else:
            config = TransformerModels.CONFIGS["ViT-B_32"]
        model =TransformerModels. VisionTransformer(config, 224, zero_head=True, num_classes=classes, vis = False)
        data = torch.load(file, map_location = "cpu")
        model.load_state_dict(data)
        del(data)
        size = (224, 224)
        if "TiT" in params[0]:
            bs = 2
        else:
            bs = 1
    model = model.to("cpu")
    torch.cuda.empty_cache()
    return model, size, bs

#Gets model loading parameters for each model
def GetLoadingParams(file, dataset):
    trans = None
    if not "TiT" in file and ("BiT" in file or "ViT" in file or "BaRT" in file):
        if "R50x1" in file: #only for the TiT defense
            return -1, -1, -1
        trans = None
        if "BiT" in file or "BaRT" in file:
            modelType = "BiT"
            if "BaRT" in file:
                if file[-7] == "5":
                    params = ["5"]
                    n = 5
                elif file[-7] == "1":
                    params = ["1"]
                    n = 1
                else:
                    params = ["10"]
                    n = 10
                modelType += "-BaRT"
                trans = lambda x ,init = False: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))((BaRTWrapper(None, n, 128, dataset).generate(x, init)))
            else:
                params = []
        else:
            modelType = "ViT"
            if "-L" in file:
                params = ["L"]
            else:
                params = ["B"]
            if "FAT" in file:
                params[0] += "-FAT"
    elif "jelly" in file:
        #trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        modelType = "jelly"
        params = []
    elif "resnet164" in file:
        modelType = "resnet164"
        if "FAT" in file:
            trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            params = ["FAT"]
        else:
            trans = None
            params = []
    elif "snn" in file:
        if dataset == "cifar10":
            trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif dataset == "cifar100":
            trans = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        modelType = "transfer-snn"
        params = ["10"]
    else:
        if not "TiT" in file:
            return -1, -1, -1

    #Handle TiT special case
    if "TiT" in file:
        if "BiT" in file:
            modelType = "ViT-BiT"
            params = ["TiT"]

        elif "Res" in file:
            modelType = "VGG-Res"
            params = ["TiT"]

    return modelType, params, trans

#Loads all other results that have already been completed or are in progress on other GPUs
def GetOtherResults(num, dataset, prefix = ""):
    dicts = []
    for i in range(10):
        if i != int(num):
            try:
                temp = torch.load(prefix + "Results-" + dataset +"-" + str(i))
            except:
                temp = {}
            dicts.append(temp)
    result_dict2 = {}
    for dict in dicts:
        for key in dict.keys():
            result_dict2[key] = dict[key]
    return result_dict2

#Gets all k combinations of the given files
def GetAllCombinations(files, k):
    tempfiles = [file for file in files if ("ini" not in file and "R50x1" not in file)]
    #tempfiles = [file for file in files if ("TiT" not in file and "BaRT" not in file and "ini" not in file and "R50x1" not in file)] # we can remove this condition if we want to attack BaRT/TiT models in an ensemble
    return itertools.combinations(tempfiles, k)

#Runs transferability experiments. There will be 1000 samples per model pair. There is a boolean indexing vector to keep track of which clean samples were already used in the attack generation for each defense. This saves a lot of time since we do not need to attack the same clean sample multiple times.
#See code in GaME.py for how to load/use the samples once done.
def Transfer(num):
    dataset = "cifar10"
    if dataset == "cifar10":
        labels  = 10
    elif dataset == "cifar100":
        labels = 100
    else:
        labels = 200

    try:
        result_dict = torch.load("Transfer-Results-" + dataset + "-" + str(num))
    except:
        result_dict = {}
    try:
        advdict = torch.load("Results-Transfer--xadv-" + dataset + "-" + str(num))
    except:
        advdict = {}
    advdict2 = torch.load("xAdvTransfer-" + str(num))

    temp = {}
    temp2 = {}
    for key in advdict.keys():
        try:
            advdict[key]["x"]
            temp[key] = advdict[key]
        except:
            temp2[key] = advdict[key]

    for key in advdict2.keys():
        try:
            advdict2[key]["x"]
            temp[key] = advdict2[key]
        except:
            temp2[key] = advdict2[key]

    advdict2 = temp
    advdict = temp2

    torch.save(advdict, "Results-Transfer-xadv-" + dataset + "-" + str(num))
    torch.save(advdict2, "xAdvTransfer-" + str(num))


    files_final = GetAllPaths()
    count = 1
    for file in files_final:

        result_dict2 = GetOtherResults(num, dataset, prefix = "Ensemble-")
        print("current results: ", result_dict)
        print("other results: ", result_dict2)
        print(file)
        modelType, params, trans = GetLoadingParams(file, dataset)
        if str(modelType) == "-1":
            continue
        if len(params)==0:
            p_str = ""
        else:
            p_str = "-" + str(params[0])
        result_dict2 = GetOtherResults(num, dataset, prefix = "Ensemble-")
        #if modelType + p_str + "-clean" in result_dict2.keys():
        #    count += 8
        #    continue
        result_dict[modelType + p_str + "-clean"] = 1
        torch.save(result_dict, "Ensemble-Results-" + dataset + "-" + str(num))



        if "TiT" in file:
            if "BiT" in file:
                resize = None #transforms.Resize((224, 224))
                advpath = "SavedModels/Vanilla/BiT-M-R50x1.tar"
                basepath = "SavedModels/Vanilla/ViT-L_16.bin"
                advnet, _, bs2 = LoadModel(advpath, "BiT", dataset, ["TiT"])
                model, size, bs = LoadModel(basepath, "ViT", dataset, ["TiT"])
                optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                #trans = lambda x : optnet.adversary_batch(x ,bs2)
            elif "Res" in file:
                resize = None #transforms.Resize((224, 224))
                data = torch.load(file)
                advnet = vggethan(dataset)
                advnet.load_state_dict(data["adv"])

                model = resnet164(32, labels)
                model.load_state_dict(data["base"])
                bs2, bs = 32, 32
                size = (32, 32)
                trans = None

                optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
            trans = lambda x, p = True : optnet.adversary_batch(x ,bs2)
        else:
            model, size, bs = LoadModel(file, modelType, dataset, params)
        resize = transforms.Resize((size[0], size[1]))
        trans12 = CreateTransWrapper(trans, resize)
        
        modelP = ModelPlus("main", model, "cuda", size[0], size[1], bs, normalize = trans12, n_classes = labels, jelly = "jelly" in modelType)
        test_loader = LoadData(modelType + p_str, dataset, bs)
        xclean, yclean = DMP.DataLoaderToTensor(test_loader)

        try:
            data = advdict2[modelType + p_str]
            xadvs = data["x"]
            yadvs = data["y"]
            allIncluded = data["i"]
        except:
            xadvs = xclean*0
            yadvs = yclean
            allIncluded = torch.zeros(len(xadvs))

        for file2 in files_final:
            print(str(count) + " / " + str(len(files_final)**2))
            count += 1
            if file == file2:
                continue
            print(file2)
            modelType2, params2, trans2 = GetLoadingParams(file2, dataset)
            if str(modelType) == "-1":
                continue
            if len(params2)==0:
                p_str2 = ""
            else:
                p_str2 = "-" + str(params2[0])

            result_dict2 = GetOtherResults(num, dataset, prefix = "Ensemble-")
            if modelType + p_str + modelType2 + p_str2 in result_dict.keys():
                continue
            if modelType + p_str + modelType2 + p_str2 + "-clean" in result_dict2.keys() or modelType + p_str + modelType2 + p_str2 in result_dict2.keys():
                continue
            if "TiT" in file2:
                if "BiT" in file2:
                    resize = None #transforms.Resize((224, 224))
                    advpath = "SavedModels/Vanilla/BiT-M-R50x1.tar"
                    basepath = "SavedModels/Vanilla/ViT-L_16.bin"
                    advnet2, _, bs22 = LoadModel(advpath, "BiT", dataset, ["TiT"])
                    model2, size2, bs12 = LoadModel(basepath, "ViT", dataset, ["TiT"])
                    optnet2 = OptNet(None, advnet2, .2, .02, 13, True, resize = resize)
                    #trans = lambda x : optnet.adversary_batch(x ,bs2)
                elif "Res" in file2:
                    resize = None #transforms.Resize((224, 224))
                    data = torch.load(file2)
                    advnet2 = vggethan(dataset)
                    advnet2.load_state_dict(data["adv"])

                    model2 = resnet164(32, labels)
                    model2.load_state_dict(data["base"])
                    bs22, bs12 = 32, 32
                    size2 = (32, 32)
                    trans2 = None

                    optnet2 = OptNet(None, advnet2, .2, .02, 13, True, resize = resize)
                trans2 = lambda x, p = True : optnet2.adversary_batch(x ,bs22)
            else:
                model2, size2, bs12 = LoadModel(file2, modelType2, dataset, params2)

            resize2 = transforms.Resize((size2[0], size2[1]))
            trans22 = CreateTransWrapper(trans2, resize2)
            
            result_dict[modelType + p_str + modelType2 + p_str2 + "-clean"] = 1
            torch.save(result_dict, "Transfer-Results-" + dataset + "-" + str(num))
            
            if "BaRT" in file or "BaRT" in file2:
                num_cpus = min( (psutil.cpu_count(logical=False)//2 + 2), 8)
                print("CPUs To Use: ", num_cpus)
                ray.init(num_cpus=num_cpus, log_to_driver = False)

            modelP2 = ModelPlus("main", model2, "cuda", size2[0], size2[1], bs12, normalize = trans22, n_classes = labels, jelly = "jelly" in modelType2)
            #test_loader = LoadData(modelType + p_str, dataset, bs)
            cleanLoader, included = GetFirstCorrectlyOverlappingSamplesBalanced("cuda", 1000, labels, test_loader, [modelP, modelP2], size, min(bs, bs12), inc = True)
            del(modelP2)
            print(len(included), len(allIncluded))
            toprocess = (included * (1- allIncluded)).bool()
            x,y = DMP.DataLoaderToTensor(cleanLoader)
            x = x[toprocess[included.bool()]]
            y = y[toprocess[included.bool()]]
            cleanLoader = DMP.TensorToDataLoader(x,y, batchSize = bs)
            allIncluded += toprocess

            if len(x) > 0:
                if "jelly" in file:
                    curmodel = model.to("cuda")
                    advloader = PGDJelly("cuda", cleanLoader, curmodel, .031, .005, 40, 0, 1, False)
                    del(curmodel)
                    torch.cuda.empty_cache()
                elif "snn" in file:
                    advloader = PGDSNN("cuda", cleanLoader, modelP, .031, .005, 40, 0, 1, False)
                    
                elif "BaRT" in file or "TiT" in file:
                    advloader = MIM_EOT_Wrapper("cuda", cleanLoader, modelP.model, .5, .031, 10, 0, 1, False, trans, 10, bs, BaRT = False)
                else:
                    advloader = AutoAttackNativePytorch("cuda", cleanLoader, modelP, .031, .005, 40, 0, 1, False)
                xadv, yclean = DMP.DataLoaderToTensor(advloader)

                xadvs[toprocess] = xadv
                yadvs[toprocess] = yclean

            try:
                val = DMP.TensorToDataLoader(xadvs[included.bool()], yadvs[included.bool()], batchSize = bs22)
                acc = modelP2.validateD(val)
                result_dict[modelType + p_str + modelType2 + p_str2] = acc
            except:
                result_dict[modelType + p_str + modelType2 + p_str2] = 1
            result_dict.pop(modelType + p_str + modelType2 + p_str2 + "-clean")
            torch.save(result_dict, "Transfer-Results-" + dataset + "-" + str(num))


            advdict[modelType + p_str + modelType2 + p_str2] = included
            torch.save(advdict, "Results-Transfer-xadv-" + dataset + "-" + str(num))

            if "BaRT" in modelType or "BaRT" in modelType2:
                try:
                    ray.shutdown()
                except:
                    pass

            torch.cuda.empty_cache()
            
            advdict2[modelType + p_str] = {"x": xadvs, "y": yadvs, "i": allIncluded}
            torch.save(advdict2, "xAdvTransfer-" + str(num))

    print("current results: ", result_dict)

#Attacks all models that are available with single model attacks (APGD/MIME)
def RunAttackSqeuential(num):
    dataset = "cifar10"
    if dataset == "cifar10":
        labels  = 10
    elif dataset == "cifar100":
        labels = 100
    elif dataset == "tiny":
        labels = 200
    try:
        result_dict = torch.load("Results-" + dataset + "-" + str(num))
    except:
        result_dict = {}
    try:
        advdict = torch.load("ResultsForGaME/" + dataset + "Results-xadv-" + dataset + "-" + str(num))
    except:
        advdict = {}
    #clean = {}

    files_final = GetAllPaths()
    count = 0
    for file in files_final:
        print(str(count) + " / " + str(len(files_final)))
        
        result_dict2 = GetOtherResults(num, dataset, prefix = "Ensemble-")
        print("current results: ", result_dict)
        print("other results: ", result_dict2)
        print(file)
        

        modelType, params, trans = GetLoadingParams(file, dataset)
        if str(modelType) == "-1":
            continue
        if len(params)==0:
            p_str = ""
        else:
            p_str = "-" + str(params[0])
        if modelType + p_str in result_dict.keys():
            continue
        if modelType + p_str + "-clean" in result_dict2.keys() or modelType + p_str in result_dict2.keys():
            continue

        if "TiT" in file:
            if "BiT" in file:
                resize = None #transforms.Resize((224, 224))
                advpath = "SavedModels/Vanilla/BiT-M-R50x1.tar"
                basepath = "SavedModels/Vanilla/ViT-L_16.bin"
                advnet, _, bs2 = LoadModel(advpath, "BiT", dataset, ["TiT"])
                model, size, bs = LoadModel(basepath, "ViT", dataset, ["TiT"])
                optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                #trans = lambda x : optnet.adversary_batch(x ,bs2)
            elif "Res" in file:
                resize = None #transforms.Resize((224, 224))
                data = torch.load(file)
                advnet = vggethan(dataset)
                advnet.load_state_dict(data["adv"])

                model = resnet164(32, labels)
                model.load_state_dict(data["base"])
                bs2, bs = 32, 32
                size = (32, 32)
                trans = None

                optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
            trans = lambda x, p = True : optnet.adversary_batch(x ,bs2)
        else:
            model, size, bs = LoadModel(file, modelType, dataset, params)
        
        
        result_dict[modelType + p_str + "-clean"] = 1
        torch.save(result_dict, "Ensemble-Results-" + dataset + "-" + str(num))
        
        if "BaRT" in file:
            num_cpus = min( (psutil.cpu_count(logical=False)//2 + 3), 8)
            print("CPUs To Use: ", num_cpus)
            ray.init(num_cpus=num_cpus, log_to_driver = False)

        modelP = ModelPlus("main", model, "cuda", size[0], size[1], bs, normalize = trans, n_classes = labels, jelly = "jelly" in modelType)
        #test_loader = LoadData(modelType + p_str, dataset, bs)
        #print(acc)
        #cleanLoader, acc = GetFirstCorrectlyOverlappingSamplesBalancedSingle("cuda", 1000, labels, test_loader, modelP)
        data = torch.load("Samples")
        xclean = data["x"]
        yclean = data["y"]
        xclean = transforms.Resize(size)(xclean)
        cleanLoader = DMP.TensorToDataLoader(xclean, yclean, batchSize = bs)

        if "jelly" in file:
            curmodel = model.to("cuda")
            advloader = PGDJelly("cuda", cleanLoader, curmodel, .031, .005, 40, 0, 1, False)
            del(curmodel)
            torch.cuda.empty_cache()
            advloader = PGDSNN("cuda", cleanLoader, modelP, .031, .005, 40, 0, 1, False)
            
        elif "BaRT" in file or "TiT" in file:
            advloader = MIM_EOT_Wrapper("cuda", cleanLoader, modelP.model, .5, .031, 10, 0, 1, False, trans, 10, bs, BaRT = False)
        else:
            advloader = AutoAttackNativePytorch("cuda", cleanLoader, modelP, .031, .005, 40, 0, 1, False)

        xadv, yclean = DMP.DataLoaderToTensor(advloader)
        if not "BaRT" in file and not "TiT" in file:
            advdict[modelType + p_str + "-PGD"] = {"x": xadv, "y": yclean}
        else:
            advdict[modelType + p_str + "-MIME"] = {"x": xadv, "y": yclean}
        torch.save(advdict, "ResultsForGaME/" + dataset + "/Results-xadv-" + dataset + "-" + str(num))

        if "BaRT" in modelType:
            try:
                ray.shutdown()
            except:
                pass
        try:
            del(model, modelP, acc, advloader, xadv, yclean, xclean)
            del(advnet)
        except:
            pass
        torch.cuda.empty_cache()

    print("current results: ", result_dict)

#Create transformation wrapper that contains the main model transformation and the resize transformation
def CreateTransWrapper(trans, resize):
    if trans != None:
        return lambda x: trans(resize(x))
    else:
        return lambda x: resize(x)

#Attacks all pairs of models with the AE-SAGA attack
def RunAttackEnsemble(num, n, k, combinations = []):
    dataset = "cifar10"
    if dataset == "cifar10":
        labels  = 10
    elif dataset == "cifar100":
        labels = 100
    elif dataset == "tiny":
        labels = 200

    files = GetAllPaths("SavedModels/tiny/")
    files = [file for file in files if "Vanilla" not in file]

    combinations = list(GetAllCombinations(files, 2))
    print(combinations)

    try:
        result_dict = torch.load("Ensemble-Results-" + dataset + "-" + str(num), map_location = "cpu")
    except:
        result_dict = {}
    try:
        advdict = torch.load("ResultsForGaME/" + dataset + "/Ensemble-Results-xadv-" + dataset + "-" + str(num), map_location = "cpu")
    except:
        advdict = {}

    count = 0
    for comb in combinations:
        count += 1

        models = []
        bss = []
        name = ""
        maxsize = ((-1, -1), "")
        names = []

        print("current results: ", result_dict)
        result_dict2 = GetOtherResults(num, dataset, prefix = "Ensemble-")
        print("other results: ", result_dict2)
        print(str(count) + " / " + str(len(combinations)))
        
        for file in comb:
            print(file)
            modelType, params, trans = GetLoadingParams(file, dataset)
            if str(modelType) == "-1":
                continue
            if len(params)==0:
                p_str = ""
            else:
                p_str = "-" + str(params[0])
            name += modelType + p_str + "-"
            names.append(modelType + p_str)

            if "TiT" in file:
                if "BiT" in file:
                    resize = None #transforms.Resize((224, 224))
                    advpath = "SavedModels/" + dataset + "Vanilla/BiT-M-R50x1.tar"
                    basepath = "SavedModels/" + dataset +"Vanilla/ViT-L_16.bin"
                    advnet, _, bs2 = LoadModel(advpath, "BiT", dataset, ["TiT"])
                    model, size, bs = LoadModel(basepath, "ViT", dataset, ["TiT"])
                    optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                elif "Res" in file:
                    resize = None #transforms.Resize((224, 224))
                    data = torch.load(file)
                    advnet = vggethan(dataset)
                    advnet.load_state_dict(data["adv"])

                    model = resnet164(32, labels)
                    model.load_state_dict(data["base"])
                    bs2, bs = 32, 32
                    size = (32, 32)
                    trans = None

                    optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                trans = lambda x : optnet.adversary_batch(x ,bs2)
            else:
                model, size, bs = LoadModel(file, modelType, dataset, params)
            
            if size[0] > maxsize[0][0]:
                maxsize = (size, modelType)

            resize = transforms.Resize((size[0], size[1]))
            trans2 = CreateTransWrapper(trans, resize)
            bss.append(bs)

            modelP = ModelPlus(modelType + p_str, model, "cuda", size[0], size[1], bs, normalize = trans2, n_classes = labels, jelly = "jelly" in modelType, TiT = False)
            models.append(modelP)
        print(maxsize)
        name = name[:-1]
        name2 = names[1] + "-" + names[0]
        if name + "-clean" in result_dict2.keys() or name2 + "-clean" in result_dict2.keys():
            print("In Progress, Skipping")
            continue
        if name in result_dict.keys() or name2  in result_dict.keys():
            print("Done, Skipping")
            continue

        if "BaRT" in name:
            num_cpus = min( (psutil.cpu_count(logical=False)//2), 8)
            print("CPUs To Use: ", num_cpus)
            ray.init(num_cpus=num_cpus, log_to_driver = False)
    

        print(name)
        result_dict[name + "-clean"] = 1
        torch.save(result_dict, "Ensemble-Results-" + dataset + "-" + str(num))


        print(names)
        data = torch.load("Samples")
        xclean = data["x"]
        yclean = data["y"]
        xclean = transforms.Resize(maxsize[0])(xclean)
        cleanLoader = DMP.TensorToDataLoader(xclean, yclean)


        torch.cuda.empty_cache()

        advloader = SelfAttentionGradientAttack_EOT("cuda", .031, .005, 40, models, cleanLoader, 0, 1, 10000, 50.0, advLoader=None, numClasses=10, decay = .5, samples = 4)
        torch.cuda.empty_cache()
        xadv, yclean = DMP.DataLoaderToTensor(advloader)
        advloader = DMP.TensorToDataLoader(xadv, yclean)

        #Many try-excepts here because it is easy for minor things to go wrong and terminate the code.
        try:
            dictV = {}
            pos = 0
            for model in models:
                nameT = names[pos]
                acc = model.validateD(advloader)
                dictV[nameT] = acc
                pos += 1
            print("------------------------------------")

            result_dict[name] = dictV
            try:
                result_dict.pop(name + "-clean")
            except:
                pass

            Ensemble = Ensemble_Efficient(models, jelly = ["jelly" in model.modelName for model in models], shufflesize = 3, p = [1], subsets = [[0,1,2]])
            acc = Ensemble.validateD(advloader)
            result_dict[name + "-comb"] = acc
            torch.save(result_dict, "Ensemble-Results-" + dataset + "-" + str(num))

            xclean, yclean = DMP.DataLoaderToTensor(cleanLoader)
            try:
                print("Perturbation Magnitude: ", torch.norm(xadv - xclean, p = np.inf))
            except:
                pass
            try:
                advdict[name] = {"x": xadv.to("cpu"), "y": yclean.to("cpu")}
                torch.save(advdict, "ResultsForGaME/" + dataset + "/Ensemble-Results-xadv-" + dataset + "-" + str(num))
            except:
                print("Failed to save Adv dict, probably storage constraints")
        except:
            try:
                advdict[name] = {"x": xadv.to("cpu"), "y": yclean.to("cpu")}
                torch.save(advdict, "ResultsForGaME/" + dataset + "/Ensemble-Results-xadv-" + dataset + "-" + str(num))
            except:
                print("Failed to save Adv dict, probably storage constraints")

        
        try:
            del(models, acc, cleanLoader, advloader, xadv, xclean, yclean)
        except:
            pass
        torch.cuda.empty_cache()

    print("current results: ", result_dict)

if __name__ == '__main__':
    #First input is which GPU is being used, second input is which save file you are using. This allows for parallel computation on systems with multiple GPUs.
    #Third input is "ensemble" to run ensemble attacks, "transfer" to run transfer attacks, "sequential attacks otherwise
    try:
        num1 = sys.argv[1]
        num = sys.argv[2]
        mode = sys.argv[3]
    except:
        num1 = 0
        num = 0
        mode = "ensemble"

    #Comment out different lines below depending on what you want to run
    with torch.cuda.device(int(num1)):
        print ('Available devices ', torch.cuda.device_count())
        print ('Current cuda device ', torch.cuda.current_device())
        if mode == "ensemble":
            RunAttackEnsemble(num, 0,2)
        elif mode == "transfer":
            Transfer(num)
        else:
            RunAttackSqeuential(num)
        
        sys.exit()
