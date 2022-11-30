# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 11:38:34 2021

@author: ethan
"""
from Defenses.BaRT.DefenseBarrageNetwork_TIN import DefenseBarrageNetwork as DBN_TIN #modified version for tiny imagenet 64x64
from Defenses.BaRT.DefenseBarrageNetwork import DefenseBarrageNetwork as DBN #modified version for CIFAR-10 128x128 
from Defenses.BaRT.DefenseBarrageNetworkOrig import DefenseBarrageNetwork as DBN_Orig #original version for CIFAR-10 32x32 images
import torch

import ray
import numpy as np
from matplotlib import pyplot as plt
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#Takes in a bart class file and images (np array) to generate the transformations on
def Generate_BART_Data(bart, xtensor, init = False):
    #try:
        x = xtensor.cpu().numpy()
        x = np.moveaxis(x, 1, -1)
        sha = x.shape
        if init:
            try:
                num_cpus = min( (psutil.cpu_count(logical=False)//2) + 2, 8)
                print("CPUs To Use: ", num_cpus)
                ray.init(num_cpus=num_cpus, log_to_driver = False)
            except:
                pass
        n = x.shape[0]
        xT = ray.get([IntermediateRay.remote(bart, x[i], i) for i in range(n)])
        if init:
            try:
                ray.shutdown()
            except:
                pass
        out = np.zeros(shape = sha)

        for i in range(n):
            out[i] = xT[i]
        out = np.moveaxis(out, -1, 1)
        return torch.tensor(out).float()
    #except:
    #    return Generate_BART_Data(bart, x)

#Wrapper function used by ray to parllelize the transformation generation
@ray.remote
def IntermediateRay(bart, x, j):
	print("Processing Sample: ", j)
	currentTransformNumber = bart.randUnifI(0, bart.TotalTransformNumber)
	if currentTransformNumber > 0 : #Only do transformations if the number is greater than 0, otherwise use original data
		if bart.ColorChannelNum == 3: #Color dataset
			out = bart.BarrageTransformColor(x, currentTransformNumber)
		else: #Grayscale
			out = bart.BarrageTransformGrayscale(x, currentTransformNumber)
		return out
	else:
		return x


#Creates a BaRT defense based upon the number of transformations, dataset, and image size (image size is most important)
def CreateBart(ntransforms,  dataset = "cifar10", imagesize = 32):
    classes = {"cifar10": 10, "cifar100": 100, "tiny":  200}
    if imagesize == 32:
        bart = DBN_Orig(None, ntransforms, classes[dataset], 3, n_cores = 10)
    elif imagesize == 64:
        bart = DBN_TIN(None, ntransforms, classes[dataset], 3, n_cores = 10)
    elif imagesize == 128:
        bart = DBN(None, ntransforms, classes[dataset], 3, n_cores = 10)
    else:
        raise ValueError("Invalid Image Size")
    return bart

class BaRTWrapper:

    def __init__(self, model, ntransforms, imagesize, dataset):
        self.bart = CreateBart(ntransforms,dataset, imagesize)
        self.model = model

    def generate(self, x, init = True):
        return Generate_BART_Data(self.bart, x, init)

    def predict(self, x):
        xT = Generate_BART_Data(self.bart, x)
        return(self.model(xT))

    def __call__(self, x):
        return self.predict(x)

    def to(self, device):
        self.model = self.model.to(device)
    
    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()

"""
#Model validation wrapper, takes in either a dataLoader (valLoader) or x,y tensors
def Validate_Bart(bart, model, valLoader = None, x = [], y = []):
    if len(x) == 0:
        x, y = PDM.DataLoaderToTensor(valLoader)
    x=x.numpy()
    xT = Generate_BART_Data(bart, x)

    xT = torch.tensor(xT).float()
    valLoader = PDM.TensorToDataLoader(xT, y, batchSize = 128)
    score = PDM.validateD(valLoader, model, device)
    return score

def LoadBiT(path, model_name, dataset = "cifar10"):
    classes = {"cifar10": 10, "cifar100": 100, "tinyimagenet":  200}
    try:
        model = models.KNOWN_MODELS[model_name](head_size=classes[dataset], zero_head=True)
    except:
        traceback.print_exc()
        print("Something went wrong with model creation, perhaps the dataset name or model name were incorrect")
        input("Press enter to continue: ")
        return
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(Fix_Dict(checkpoint))

#Generic Model Evaluation Function, doesn't use BaRT
def TestPrediction(model, valLoader):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            print("Processing up to sample=", batchTracker)
            if device == None: #assume CUDA by default
                print("cuda")
                inputVar = input.cuda()
            else:
                print("device")
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            print("Getting Output")
            output = model(inputVar)
            print("Finished Output")
            output = output.float()

#This function is for testing the Expectation over Transformation attack I mentioned in the meeting with Nuo.
#Could be useful later when we attack the defense
def mainEOT():
    print(os.listdir())
    try:
        ray.shutdown()
    except:
        pass
    ray.init(num_cpus=7, log_to_driver=False)
    dataset = "cifar10"
    model_name = "BiT-M-R101x3"
    modelpath = wd + "\\Models\\BaRT-10-101x3"
    ntransforms = 10

    print("Making BaRT")
    bart, model = CreateBart(modelpath, ntransforms, dataset, model_name)
    print("On Cuda")
    model = model.to(device)
    torch.cuda.empty_cache()

    modelplus = ModelPlus("BiT", model, device, 128, 128, 128)
    print("Loading Data")
    valLoader =  PDM.GetCIFAR10Validation(128, 128)

    print("Running Transformations")
    x, y = PDM.DataLoaderToTensor(valLoader)

    data = torch.load(wd + "\\Clean_10")
    x = data["x"]
    y = data["y"]
    del(data)

    model = model.to("cpu")
    func = lambda x: Generate_BART_Data(bart, x)
    xShape = (x.shape[1], x.shape[2], x.shape[3])
    xadv = torch.load("xadv_temp_10")
    #xadv = torch.zeros(len(x), xShape[0], xShape[1], xShape[2])
    bs = 10
    for i in range(10,len(x)//bs):
        print("Outer Iteration: ", i)
        xadv[i*bs:(i+1)*bs] = MIM_EOT_Batch(device, x[i*bs:(i+1)*bs], y[i*bs:(i+1)*bs], model, .5, .031, .0031, 10, 0, 1, False, func, 50)
        #xadv[i] = MIM_EOT(device, x[i:i+1].to(device), y[i].to(device), model, .5, .031, .0031, 10, 0, 1, False, func)
        print(torch.max(xadv[i].to("cpu") - x[i]), torch.min(xadv[i].to("cpu") - x[i]))
        torch.save(xadv,"xadv_temp_10")
    #xAdv = MIM_EOT("cuda", x, y, model, .5, .031, .0031, 10, 0, 1, False, func)
    score = 0
    advLoader = PDM.TensorToDataLoader(xadv, y, batchSize = 128)
    model = model.to(device)
    for i in range(5):
        score += Validate_Bart(bart, model, advLoader)
    print(score/5)
"""

if __name__ == "__main__":
    pass
    """#Use this as a template for sandy checking the code
    model_path = "C:\\Dev\\GameOfRings\\BART_GameOfRings\\Models\\BaRT-5-101x3" #path to BiT Model
    #101x3 model takes a lot of memory, but it is what I have available at the moment
    model_name = "BiT-M-R101x3" #name of model you're using e.g. BiT-M-R50x1 (check models.py KNOWN_MODELS for more) 
    dataset = "cifar10"
    model = LoadBiT(model_path, model_name, dataset)

    valLoader = PDM.GetCIFAR10Validation(128, 128) #image size for BiT is 128x128

    bart = CreateBart(5, "cifar10", 128)

    Validate_Bart(bart, model, valLoader)"""
