#External Imports
from email.errors import FirstHeaderLineIsContinuationDefect
import torch
from torchvision import transforms
from spikingjelly.clock_driven.model import spiking_resnet, sew_resnet
from spikingjelly.clock_driven import neuron, surrogate
import numpy as np
import itertools
import random
import pickle
from scipy.optimize import linprog
import psutil
import ray

#System Imports
import sys
import os
import time
import warnings
import csv
warnings.filterwarnings("ignore")

#Internal Imports
from Utilities import DataManagerPytorch as DMP
from Utilities.ModelPlus import ModelPlus, Ensemble_Efficient
from Automate import GetLoadingParams, LoadData, LoadModel, CreateTransWrapper, GetAllPaths
#from Automate_TIN import GetLoadingParams, LoadData, LoadModel, CreateTransWrapper, GetAllPaths

from Defenses.Vanilla import TransformerModels, BigTransferModels
from Defenses.Vanilla.ResNet import resnet164
from Defenses.SNN.resnet_spiking import RESNET_SNN_STDB
from Defenses.BaRT.ValidateBart import BaRTWrapper
from Defenses.TIT.optdefense import OptNet
from Defenses.Vanilla.vgg import vggethan

from Attacks.attack_utilities import GetFirstCorrectlyOverlappingSamplesBalanced, GetFirstCorrectlyOverlappingSamplesBalancedSingle
from Attacks.AttackWrappersProtoSAGA import SelfAttentionGradientAttackProto, MIM_EOT_Wrapper
import scipy
from matplotlib import pyplot as plt
print(scipy.__version__)


def GetBalanced(sampleNum, numClasses, dataLoader):
    
    shape = DMP.GetOutputShape(dataLoader)
    #Basic variable setup 
    samplePerClassCount = torch.zeros(numClasses) #keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses) #Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader) #Get all the data as tensors 
    #Memory for the solution 
    xClean = torch.zeros(sampleNum)

    xClean2 = torch.zeros(len(xTest) - sampleNum)
    #print(xClean.shape)
    sampleIndexer = 0
    #Go through all the samples
    j = 0
    for i in range(0, len(xTest)):
        currentClass = int(yTest[i])
        #Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if samplePerClassCount[currentClass]<maxRequireSamplesPerClass:
            xClean[sampleIndexer] = i#xTest[i]
            sampleIndexer = sampleIndexer +1 #update the indexer 
            samplePerClassCount[currentClass] = samplePerClassCount[currentClass] + 1 #Update the number of samples for this class
        else:
            xClean2[j] = i
            j += 1

    return xClean.long(), xClean2.long()

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
                string = ""
                if dataset == "tiny":
                    string = "tiny/"
                advpath = "SavedModels/" + string + "Vanilla/BiT-M-R50x1.tar"
                basepath = "SavedModels/" + string + "Vanilla/ViT-L_16.bin"
                advnet, _, bs2 = LoadModel(advpath, "BiT", dataset, ["TiT"])
                model, size, bs = LoadModel(basepath, "ViT", dataset, ["TiT"])
                #optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                trans = lambda x : OptNet(None, advnet, .2, .02, 13, True, resize = resize).adversary_batch(x ,bs2)
            elif "Res" in file:
                #print("Res")
                resize = None #transforms.Resize((224, 224))
                data = torch.load(file)
                advnet2 = vggethan(dataset)
                advnet2.load_state_dict(data["adv"])

                model = resnet164(32, labels)
                model.load_state_dict(data["base"])
                bs2, bs = 32, 32
                size = (32, 32)

                #optnet = OptNet(None, advnet, .2, .02, 13, True, resize = resize)
                trans = lambda x : OptNet(None, advnet2, .2, .02, 13, True, resize = resize).adversary_batch(x ,bs2)
        else:
            model, size, bs = LoadModel(file, modelType, dataset, params)
        resize = transforms.Resize((size[0], size[1]))
        trans2 = CreateTransWrapper(trans, resize)
    
        modelP = ModelPlus(modelType + p_str, model, "cuda", size[0], size[1], bs, normalize = trans2, n_classes = labels, jelly = "jelly" in modelType)
        models.append(modelP)
    return models

def GaMEn(defenses, n, uniform = False, mv = False, res = [], N = 800, dataset = "cifar10"):
    if dataset == "cifar10":
        labels = 10
    else:
        labels = 200

    subsets = []
    for j in range(1,n+1):
        subsets += list(itertools.combinations([i for i in range(len(defenses))], j))
    print(len(subsets))
    #Load in all the adversarial data
    files = GetAllPaths("ResultsForGaME/" + dataset + "/")
    dict = {}
    for file in files:
        if not "adv" in file:
            continue
        #if not "tiny" in file:
        #    continue
        if dataset != "cifar10":
            if not "tiny" in file:
                continue
        else:
            if "tiny" in file:
                continue

        data = torch.load(file, map_location = "cpu")
        for key in data.keys():
            dict[key] = data[key]
        del(data)

    
    keys = list(dict.keys())
    keys.sort()

    num_cpus = min( (psutil.cpu_count(logical=False)//2 + 2), 8)
    #print("CPUs To Use: ", num_cpus)
    ray.init(num_cpus=num_cpus, log_to_driver = False)

    results = {}
    A = -1 * np.ones(shape = (len(dict.keys())+1, len(subsets)+1))
    A2 = -1 * np.ones(shape = (len(subsets)+1, len(dict.keys())+1))
    i = j = 0
    forEval = {}
    data = dict[keys[0]]
    xadv = data["x"]
    yclean = data["y"]
    dataloader = DMP.TensorToDataLoader(xadv, yclean, randomizer = False)
    s2, s1 = GetBalanced(200, labels, dataloader)
    #yclean = yclean[[s1][:100]]

    
    pred = {}
    #i = 0

    string = ""
    if dataset != "cifar10":
        string = "-tiny"
    try:
        yclean = torch.load("labels3"+string)
        pred = torch.load("Preds3"+string)
        for key in keys:
            data = dict[key]
            xadv = data["x"]
            yclean2 = data["y"]
            forEval[key] = DMP.TensorToDataLoader(xadv[s2], yclean2[s2])
    except:

        for key in keys:
        #    j = 0
            data = dict[key]
            xadv = data["x"]
            yclean2 = data["y"]
            forEval[key] = DMP.TensorToDataLoader(xadv[s2], yclean2[s2])
            dataloader = DMP.TensorToDataLoader(xadv[s1], yclean[s1])
            for defense in defenses:
                pred_t = defense.predictD(dataloader)
                j += 1
                pred[defense.modelName + "+" + key] = pred_t
            i += 1
        torch.save(pred, "Preds3"+string)
        torch.save(yclean[s1], "labels3"+string)
    
    j = i = 0
    for subset in subsets:
        i = 0
        #print(subset)
        defn = [defenses[ind] for ind in subset]#defenses[list(subset)]
        for key in keys:
            yout = torch.zeros(len(yclean), labels)
            for h in range(len(defn)):
                d1 = defn[h]
                #preds = (torch.nn.functional.softmax(pred[d1.modelName + "+" + key]) + torch.nn.functional.softmax(pred[d1.modelName + "+" + key]))/2
                pred1 = pred[d1.modelName + "+" + key]
                print(len(pred1))
                if not mv:
                    yout += torch.nn.functional.softmax(pred1)
                else:
                    for q in range(len(pred1)):
                        yout[q, pred1[q].argmax(axis = 0)] += 1
            acc = 0
            for k in range(N):
                ind = yout[k].argmax(axis=0)
                m = yout[k].max(axis = 0).values
                if ind == yclean[k]:
                    acc += 1
            acc /= N
            A[i,j] = acc
            A2[j,i] = 1-acc
            title = ""
            for defense in defn:
                title += defense.modelName
                title += "+"
            title += key
            results[title] = acc
                #print(title,acc)
            i += 1
        j += 1

    print(A)

    A[-1,-1] = 0
    A *= -1
    A2[-1,-1] = 0
    A2 *= -1
    #l = np.zeros(len(defenses)+1) #lower bound
    c = np.zeros(len(subsets)+1) #objective function (0,0,0, ... -1)
    c[-1] = -1 #since scipy requires we do a minimization problem we can simply multiply out objective function by -1

    b = np.zeros(len(dict.keys())+1)
    b[-1] = 1 #bounds probabilities to sum to 1


    solution = linprog(c, A_ub = A, b_ub = b, method = "simplex")
    print(solution.x)


    print(subsets)
    if uniform:
        strat = [1/len(subsets) for i in range(len(subsets))] 
    else:
        strat = solution.x[:-1]

    if mv:
        mode = "mv"
    else:
        mode = "avg"

    ensemble = Ensemble_Efficient(defenses, nclasses = labels, p = strat, shufflesize=n, subsets = subsets, shufflemode = mode)
    keys = list(forEval.keys())
    keys.sort()
    arr = np.zeros(len(keys))
    i = 0
    for key in keys:
        for k in range(1):
            arr[i] += ensemble.validateD(forEval[key], adv = True)
        arr[i]/=1
        i += 1
    print(arr)
    print(keys)


    if dataset == "tiny":
        test_loader = DMP.LoadTinyImageNetValidationData("data//tiny-imagenet-200", (224,224), 4)
    else:
        test_loader = DMP.get_CIFAR10_loaders_test(img_size_H = 224, img_size_W = 224, bs = 2, norm = True)
    acc = ensemble.validateD(test_loader)

    try:ray.shutdown()
    except:pass


    return solution.x[-1], np.min(arr), acc

def plot(train_images, train_labels = [], name = "mygraph.png"):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow( torch.movedim(train_images[i], 0,2) , cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        if len(train_labels) > 0:
            plt.xlabel(class_names[int(train_labels[i])])
    plt.savefig(name, dpi = 500)

def GenerateTransferTable():

    #Loading everything at once takes a lot of memory
    files_final = GetAllPaths("SavedModels/")
    files_final = [file for file in files_final if "tiny" not in file]
    defenses = Load_Defenses(files_final, "cifar10")

    files = os.listdir("TransferResults-NEW/")
    files.append("Results-xadv-cifar10-0")
    dict = {}

    for file in files:
        data = torch.load("TransferResults-NEW/" + file)
        for key in data.keys():
            dict[key] = data[key]

    print(len(dict.keys()))
    for key in dict.keys():
        print(key)

    num_cpus = min( (psutil.cpu_count(logical=False)//2 + 2), 8)
    print("CPUs To Use: ", num_cpus)
    ray.init(num_cpus=num_cpus, log_to_driver = False)

    results = {}
    names = ["-MIM", "-PGD"]
    best = {}
    for d in defenses:
        results_i = {}
        acc = -1
        matfinal = None
        finalLoader = None
        for name in names:
            data = dict[d.modelName + name]
            x = data["x"]
            y = data["y"]
            #x = transforms.Resize((d.imgSizeH, d.imgSizeW))(x)
            dataLoader = DMP.TensorToDataLoader(x, y, batchSize = d.batchSize)
            tempmat, tempacc = d.validateDA(dataLoader)
            tempmat = tempmat.detach().long()
            tempmat = tempmat == 0
            tempacc = tempacc
            print(d.modelName, name, tempacc)
            if acc == -1 or tempacc < acc:
                acc = tempacc
                finalLoader = dataLoader
                matfinal = tempmat
                best[d.modelName] = (name, acc)

        x,y = DMP.DataLoaderToTensor(finalLoader)
        x = x[matfinal].cpu()
        y = y[matfinal].cpu()
        print(len(x), len(y))
        finalLoader = DMP.TensorToDataLoader(x,y)
        for d2 in defenses:
            if d.modelName == d2.modelName:
                continue
            x,y = DMP.DataLoaderToTensor(finalLoader)
            x = transforms.Resize((d2.imgSizeH, d2.imgSizeW))(x)
            finalfinalLoader = DMP.TensorToDataLoader(x,y)
            acc = d2.validateD(finalfinalLoader)
            results_i[d2.modelName] = acc
            print(d2.modelName, acc)

            torch.cuda.empty_cache()
        results[d.modelName] = results_i
        torch.save(results, "FinalResults-Transfer3")

    print(best)
    try:ray.shutdown()
    except:pass
    
    
    results = torch.load("FinalResults-Transfer3")
    #print(results)

    #torch.save(results, "FinalResults-Transfer3")
    keys = list(results.keys())
    keys.sort()
    print(keys)

    mat = np.zeros(shape = (len(keys), len(keys)))
    #i = 0
    #for key in keys:
    #    mat[0,i+1] = key
    #    mat[i+1, 0] = key
    #    i += 1

    i = 0
    for key in keys:
        data = results[key]
        j = 0
        keys2 = list(data.keys())
        keys2.sort()
        for key2 in keys2:
            if i == j:
                j += 1
                
                #mat[i,j] = best[key][1] #This is where min robust accuracy is stored
            mat[i,j] = data[key2]
            j += 1
        i += 1
    
    np.savetxt("TransferResults.csv", mat, delimiter = ",")
    np.savetxt("TransferResults2.csv", 1-mat, delimiter = ",")

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(mat[i,j], end = " ")
        print()


if __name__ == "__main__":
    dataset = sys.argv[1]
    n = sys.argv[2]
    mv = sys.argv[3]
    uniform = sys.argv[4]
    
    if dataset == "tiny":
        files = GetAllPaths("SavedModels/tiny/")
        files = [file for file in files if "Vanilla" not in file]

    else:
        files = GetAllPaths("SavedModels/")
        files = [file for file in files if "tiny" not in file]

    #Change dataset below for cifar10
    print(files)
    defenses = Load_Defenses(files, dataset = dataset)
    todo = [[False, False], [True, False]]
    #sys.exit()

    res = []
    n = 1 #max ensemble size
    uniform = uniform == "true" #whether or not to use uniform ensemble distribution
    mv = mv == "fh" #True to use fh False to use fs voting scheme
    temp = GaMEn(defenses, n, uniform, mv, dataset = dataset)
    





    

    


