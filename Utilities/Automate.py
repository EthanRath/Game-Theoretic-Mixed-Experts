#External Imports
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
from Attacks.AttackWrappersProtoSAGA import SelfAttentionGradientAttackProto, MIM_EOT_Wrapper, AutoAttackNativePytorch, MIMNativePytorch_cnn
from Attacks.AttackWrappersWhiteBoxSNN import PGDNativePytorch as PGDSNN
from Attacks.AttackWrappersWhiteBoxJelly import PGDNativePytorch as PGDJelly
from Attacks.AttackWrappersWhiteBoxSNN import MIMNativePytorch as MIMSNN
from Attacks.AttackWrappersWhiteBoxJelly import MIMNativePytorch as MIMJelly


def GetAllPaths():
    arr = []
    files = os.listdir("SavedModels/cifar10/") #Lists folders
    for file in files:
        infiles = os.listdir("SavedModels/cifar10/" + file + "/") #lists model files inside of folders
        for ifile in infiles:
            arr.append("SavedModels/cifar10/" + file + "/" + ifile)
    return arr


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

def LoadModel(file, modelType, dataset, params = []):
    if dataset == "cifar10":
        classes = 10
    elif dataset == "cifar100":
        classes = 100

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
        bs = 64

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
            bs = 8
        else:
            data = torch.load(file,  map_location = "cpu")["model"]
            size = (160, 128)
            if len(params) > 0 and "TiT" in params[0]:
                bs = 16
            else:
                bs = 8
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
        model =TransformerModels. VisionTransformer(config, 224, zero_head=True, num_classes=classes, vis = True)
        data = torch.load(file, map_location = "cpu")
        model.load_state_dict(data)
        del(data)
        size = (224, 224)
        if "TiT" in params[0]:
            bs = 16
        else:
            bs = 16
    model = model.to("cpu")
    torch.cuda.empty_cache()
    return model, size, bs

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

def GetAllCombinations(files, k):
    tempfiles = [file for file in files if ("ini" not in file and "R50x1" not in file)]
    #tempfiles = [file for file in files if ("TiT" not in file and "BaRT" not in file and "ini" not in file and "R50x1" not in file)] # we can remove this condition if we want to attack BaRT/TiT models in an ensemble
    return itertools.combinations(tempfiles, k)

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
        advdict = torch.load("xadv-auto-" + str(num))
    except:
        advdict = {}

    files_final = GetAllPaths()#['SavedModels/cifar10/TiT/BiTViT.txt']
    #with open("ToDo.txt", "rb") as file:
    #    files_final = pickle.load(file)
    #files_final.append('SavedModels/BaRT/BaRT-1-101x3')
    count = 0
    #files_final = ['SavedModels/SNN/jelly_resnet18acc_8183.pt']
    for file in files_final:
        count += 1
        print(str(count) + " / " + str(len(files_final)))
        
        result_dict2 = GetOtherResults(num, dataset)
        print("current results: ", result_dict)
        print("other results: ", result_dict2)
        print(file)
        if not "Vanilla/BiT" in file:
            continue

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
                advpath = "SavedModels/cifar10/Vanilla/BiT-M-R50x1.tar"
                basepath = "SavedModels/cifar10/Vanilla/ViT-L_16.bin"
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
        
        """if "BaRT" in file:
                num_cpus = min( (psutil.cpu_count(logical=False)//2 + 2), 8)
                print("CPUs To Use: ", num_cpus)
                ray.init(num_cpus=num_cpus, log_to_driver = False)"""

        modelP = ModelPlus("main", model, "cuda", size[0], size[1], bs, normalize = trans, n_classes = labels, jelly = "jelly" in modelType)
        #test_loader = LoadData(modelType + p_str, dataset, bs)
        #print(acc)
        #cleanLoader, acc = GetFirstCorrectlyOverlappingSamplesBalancedSingle("cuda", 1000, labels, test_loader, modelP)
        data = torch.load("SamplesTransfer")
        xclean = data["x"]
        yclean = data["y"]
        xclean = transforms.Resize(size)(xclean)
        cleanLoader = DMP.TensorToDataLoader(xclean, yclean, batchSize = bs)

        #acc = modelP.validateD(cleanLoader)
        #print(acc)

        if "jelly" in file:
            curmodel = model.to("cuda")
            advloader = PGDJelly("cuda", cleanLoader, curmodel, .031, .005, 40, 0, 1, False)
            advloader2 = MIMJelly("cuda", cleanLoader, curmodel, .5, .031, .0031, 10, 0, 1, False)
            del(curmodel)
            torch.cuda.empty_cache()
        elif "snn" in file:
            advloader = PGDSNN("cuda", cleanLoader, modelP, .031, .005, 40, 0, 1, False)
            advloader2 = MIMSNN("cuda", cleanLoader, modelP, .5, .031, .0031, 10, 0, 1, False)
        elif "BaRT" in file or "TiT" in file:
            advloader = MIM_EOT_Wrapper("cuda", cleanLoader, modelP.model, .5, .031, 10, 0, 1, False, trans, 10, bs, BaRT = False)
            temptrans = modelP.normalize
            modelP.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if "BaRT" in file else None
            advloader = AutoAttackNativePytorch("cuda", cleanLoader, modelP, .031, .005, 20, 0, 1, False)
            modelP.normalize = temptrans
        else:
            advloader = AutoAttackNativePytorch("cuda", cleanLoader, modelP, .031, .005, 20, 0, 1, False)
            advloader2 = MIMNativePytorch_cnn("cuda", cleanLoader, modelP, .5, .031, .0031, 10, 0, 1, 0, 1, False)

        xadv, yclean = DMP.DataLoaderToTensor(advloader)
        advdict[modelType + p_str + "-PGD"] = {"x": xadv, "y": yclean}
        torch.save(advdict, "Results-xadv-" + dataset + "-" + str(num))

        xadv, yclean = DMP.DataLoaderToTensor(advloader2)
        advdict[modelType + p_str + "-MIM"] = {"x": xadv, "y": yclean}
        torch.save(advdict, "Results-xadv-" + dataset + "-" + str(num))

        """if "BaRT" in modelType:
            try:
                ray.shutdown()
            except:
                pass"""


        del(model, modelP, acc, advloader, xadv, yclean, xclean)
        try:
            del(advnet)
        except:
            pass
        torch.cuda.empty_cache()

    print("current results: ", result_dict)

def CreateTransWrapper(trans, resize):
    if trans != None:
        return lambda x: trans(resize(x))
    else:
        return lambda x: resize(x)

def RunAttackEnsemble(num, n, k, combinations = []):
    dataset = "cifar10"
    if dataset == "cifar10":
        labels  = 10
    elif dataset == "cifar100":
        labels = 100
    elif dataset == "tiny":
        labels = 200

    with open("ToDo.txt", "rb") as file:
        combinations = pickle.load(file)

    try:
        result_dict = torch.load("Ensemble-Results-" + dataset + "-" + str(num), map_location = "cpu")
    except:
        result_dict = {}
    try:
        advdict = torch.load("Ensemble-Results-xadv-" + dataset + "-" + str(num), map_location = "cpu")
    except:
        advdict = {}
    """result_dict.pop("BiT-BaRT-5-ViT-L-FAT")
    result_dict.pop("BiT-BaRT-5-ViT-L-FAT-comb")
    print(result_dict.keys())"""

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
                    advpath = "SavedModels/Vanilla/BiT-M-R50x1.tar"
                    basepath = "SavedModels/Vanilla/ViT-L_16.bin"
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

        advloader = SelfAttentionGradientAttackProto("cuda", .031, .005, 40, models, cleanLoader, 0, 1, 10000, 50.0, advLoader=None, numClasses=10, decay = .5, samples = 4, BaRT = False)
        torch.cuda.empty_cache()
        xadv, yclean = DMP.DataLoaderToTensor(advloader)
        advloader = DMP.TensorToDataLoader(xadv, yclean)

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

            Ensemble = Ensemble_Efficient(models, jelly = ["jelly" in model.modelName for model in models], shufflesize = 2, p = [1], subsets = [[0,1]])
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
                torch.save(advdict, "Ensemble-Results-xadv-" + dataset + "-" + str(num))
            except:
                print("Failed to save Adv dict, probably storage constraints")
        except:
            try:
                advdict[name] = {"x": xadv.to("cpu"), "y": yclean.to("cpu")}
                torch.save(advdict, "Ensemble-Results-xadv-" + dataset + "-" + str(num))
            except:
                print("Failed to save Adv dict, probably storage constraints")
        if "BaRT" in name:
            try:
                ray.shutdown()
            except:
                pass
        
        try:
            del(models, acc, cleanLoader, advloader, xadv, xclean, yclean)
        except:
            pass
        torch.cuda.empty_cache()

    print("current results: ", result_dict)

def RunAttackEnsemble_Debug(num, n, k):
    dataset = "cifar10"
    if dataset == "cifar10":
        labels  = 10
    elif dataset == "cifar100":
        labels = 100
    files_final = GetAllPaths()
    combinations = list(GetAllCombinations(files_final, k))
    #random.shuffle(combinations)
    done = torch.load("Complete-SAGA")
    global dict
    new_dict = {}
    try:
        result_dict = torch.load("Ensemble-Results-" + dataset + "-" + str(num))
    except:
        result_dict = {}
    try:
        advdict = torch.load("Ensemble-Results-xadv-" + dataset + "-" + str(num))
    except:
        advdict = {}
    count = 1
    for comb in combinations:
        cont = False
        for m in comb:
            if "TiT" in m:
                cont = True
            if "BaRT" in m:
                cont = True
        if cont:
            continue

        models = []
        bss = []
        name = ""
        maxsize = ((-1, -1), "")
        names = []

        #print("current results: ", result_dict)
        result_dict2 = GetOtherResults(num, dataset, prefix = "Ensemble-")
        #print("other results: ", result_dict2)
        print(str(count) + " / " + str(len(combinations)))
        count += 1

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
                    for file in files_final:
                        if "Vanilla" in file:
                            if "BiT" in file:
                                advpath = file
                            elif "ViT-L" in file:
                                basepath = file
                    advnet, _, bs2 = LoadModel(advpath, "BiT", dataset, ["TiT"])
                    model, size, bs = LoadModel(basepath, "ViT", dataset, ["TiT"])
                    optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                    trans = lambda x : optnet.adversary_batch(x ,bs2)
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

            modelP = ModelPlus(modelType, model, "cuda", size[0], size[1], bs, normalize = trans2, n_classes = labels, jelly = "jelly" in modelType, TiT = (len(params)>0 and "TiT" in params[0]))
            models.append(modelP)
        
        name = name[:-1]
        print(name)
        name2 = names[1] + "-" + names[0]
        cont = True
        if name in done.keys():
            cont = False
        if name2 in done.keys():
            cont = False
        if cont:
            continue

        try:
            xadv = dict[name]
        except:
            xadv = dict[name2]

        """if name + "-clean" in result_dict2.keys():
            print("Already In Progress, Skipping")
            continue
        if name in result_dict.keys():
            print("Already Done, Skipping")
            continue"""

        print(names)
        test_loader = LoadData(maxsize[1], dataset, int(np.min(bss)))
        cleanLoader = GetFirstCorrectlyOverlappingSamplesBalanced("cuda", 1000, labels, test_loader, models, maxsize[0], int(np.min(bss)))
        #advloader = SelfAttentionGradientAttackProto("cuda", .031, .005, 40, models, cleanLoader, 0, 1, 10000, 50.0, advLoader=None, numClasses=10)
        xclean, yclean = DMP.DataLoaderToTensor(cleanLoader)

        advloader = DMP.TensorToDataLoader(xadv, yclean)
        dictV = {}
        pos = 0
        for model in models:
            nameT = names[pos]
            acc = model.validateD(advloader)
            dictV[nameT] = acc
            pos += 1
        print("------------------")
        #xadv, yadv = DMP.DataLoaderToTensor(advloader)
        #xclean, yclean = DMP.DataLoaderToTensor(cleanLoader)
        new_dict[name] = dictV
        #advdict[name] = {"x": xadv, "y": yclean}
        torch.save(new_dict, "Ensemble-Results-New-" + dataset + "-" + str(num))

        Ensemble = Ensemble_Efficient(models, jelly = ["jelly" in model.modelName for model in models], shufflesize = 2, p = [1], subsets = [[0,1]])
        acc = Ensemble.validateD(advloader)
        new_dict[name + "-comb"] = acc
        torch.save(new_dict, "Ensemble-Results-New-" + dataset + "-" + str(num))
        print("current results: ", new_dict)

        del(models, test_loader, acc, cleanLoader, advloader, xadv, xclean, yclean)
        torch.cuda.empty_cache()

    print("current results: ", result_dict)

if __name__ == '__main__':
    try:
        num1 = sys.argv[1]
        num = sys.argv[2]
    except:
        num1 = 0
        num = 0

    with torch.cuda.device(int(num1)):
        print ('Available devices ', torch.cuda.device_count())
        print ('Current cuda device ', torch.cuda.current_device())
        RunAttackSqeuential(num)
        #RunAttackEnsemble(num, 0, 2)
        sys.exit()
