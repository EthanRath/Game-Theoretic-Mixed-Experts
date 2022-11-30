#This is the model "plus" class
#It wraps a Pytorch model, string name, and transforms together 
import torch 
import torchvision
import Utilities.DataManagerPytorch as DMP
import numpy as np
from torch import flatten
import torch
from torchvision import transforms
from spikingjelly.clock_driven import functional
import math
import itertools

def comb(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

class Ensemble():
    def __init__(self, models, nclasses = 10):
        self.models = models
        self.nclasses = nclasses

    def predict(self, x):
        out = torch.zeros(size = (len(x), self.nclasses)).to("cuda")
        r = np.random.randint(0, len(self.models), len(x))
        for i in range(len(self.models)):
            out[r == i] = self.models[i](x[r == i])
        return out
    
    def __call__(self, x):
        return self.predict(x)

    def eval(self):
        for model in self.models:
            model.eval()
    
    def train(self):
        for model in self.models:
            model.train()

class Ensemble_Efficient():
    def __init__(self, models, trans = [], jelly = [], nclasses = 10, rays = False, device = "cuda", shufflesize = 1, shufflemode = "avg", p = [], subsets = []):
        self.models = models
        self.nclasses = nclasses
        if len(trans) == 0:
            self.trans = [None]*len(models)
        else:
            self.trans = trans

        if len(jelly) == 0:
            self.jelly = [False]*len(models)
        else:
            self.jelly = jelly
        self.device = device
        self.rays = rays
        self.imgSizeH = 32
        self.imgSizeW = 32
        self.batchSize = 32
        self.shufflesize = shufflesize
        self.shufflemode = shufflemode
        if len(p) == 0:
            combs = comb(len(models), shufflesize)
            self.p = np.ones(combs) / combs
        else:
            self.p = p
        
        self.subsets = subsets
        self.modelName = ""
        for model in self.models:
            try:
                self.modelName += model.modelName + "-"
            except:
                pass
        self.modelName = self.modelName[:-1]

    def predict_single(self, dataLoader):
        out = torch.zeros((len(dataLoader.dataset), self.nclasses)).to("cpu")
        r = np.random.choice(np.arange(0, len(self.models)), size = (len(dataLoader.dataset)), p=self.p)
        x,y = DMP.DataLoaderToTensor(dataLoader)
        for i in range(len(self.models)):
            if np.sum(r == i) == 0:
                continue
            xIn = x[r == i]
            yIn = y[r == i]
            dlIn = DMP.TensorToDataLoader(xIn, yIn)
            if "BaRT" or "TiT" in self.models[i].modelName:
                for j in range(5):
                    if j == 0:
                        pred = torch.nn.functional.softmax(self.models[i].predictD(dlIn, self.nclasses))
                    else:
                        pred += torch.nn.functional.softmax(self.models[i].predictD(dlIn, self.nclasses))
                pred /= 5
            else:
                pred = self.models[i].predictD(dlIn, self.nclasses)
                pred = torch.nn.functional.softmax(pred)

            out[r == i] = pred.cpu()

        if self.rays:
            return out.detach().cuda()
        else:
            return out.detach().cpu()

    def predict_mv(self, models, dataLoader):
        yout = torch.zeros(len(dataLoader.dataset), self.nclasses)
        for model in models:
            pred = model.predictD(dataLoader, self.nclasses)
            #pred = torch.nn.functional.softmax(pred)
            for j in range(len(pred)):
                yout[j, pred[j].argmax(axis = 0)] += 1
            #print(pred.mean(), model.modelName)
        #yout /= len(models)
        #print(yout)
        return yout


    def predict_avg(self, models, dataLoader):
        yout = torch.zeros(len(dataLoader.dataset), self.nclasses)
        for model in models:
            pred = model.predictD(dataLoader, self.nclasses)
            pred = torch.nn.functional.softmax(pred)
            #print(pred.mean(), model.modelName)
            yout += pred
        yout /= len(models)
        return yout

    def predict_multi(self, dataLoader):
        r = np.random.choice(len(self.subsets), size = (len(dataLoader.dataset)), p=self.p) #r tells index of which subset is to be used
        #r = [self.subsets[rT[i]] for i in range(len(rT))]
        yout = torch.ones(len(dataLoader.dataset), self.nclasses)
        x, y = DMP.DataLoaderToTensor(dataLoader)
        tracker = 0
        for i in range(len(self.subsets)):
            group = (r == i)
            count = np.sum(group)
            if count == 0:
                continue
            models = [self.models[j] for j in range(len(self.models)) if j in self.subsets[i]]
            tracker += count
            xIn = x[group]
            yIn = y[group]
            dlIn = DMP.TensorToDataLoader(xIn, yIn)
            if self.shufflemode == "avg":
                yout[group] = self.predict_avg(models, dlIn)
            else:
                yout[group] = self.predict_mv(models, dlIn)
        return yout

    def predict(self, dataLoader):
        if self.shufflesize == 1:
            return self.predict_single(dataLoader)
        else:
            return self.predict_multi(dataLoader)

    def validateD(self, dataLoader, adv = False):
        #acc = DMP.validateD(dataLoader, self)
        pred = self.predictD(dataLoader)
        _, target = DMP.DataLoaderToTensor(dataLoader)
        acc = 0
        for j in range(0, len(pred)):
            #ind = pred[j].argmax(axis=0)
            m = pred[j].max(axis = 0).values
            #if torch.sum(pred[j] == m) > 1:
            #    if adv:
            #        acc += 1
            if pred[j].argmax(axis=0) == target[j]:
                acc = acc +1
        #print([pred[i].argmax(axis = 0) for i in range(len(pred))])
        #print(target)
        acc /= len(pred)
        return acc

    def validateDA(self, dataLoader):
        accArray, acc = DMP.validateDA(dataLoader, self, self.device)
        torch.cuda.empty_cache()
        return accArray, acc

    def predictD(self, dataLoader):
        return self.predict(dataLoader)

    def __call__(self, x):
        return self.predictT(x)

    def predictT(self, x):
        yfake = torch.ones(len(x))
        dlFake = DMP.TensorToDataLoader(x, yfake)
        return self.predict(dlFake)

    def eval(self):
        for model in self.models:
            model.eval()
    
    def train(self):
        for model in self.models:
            model.train()

class ModelPlus():
    #Constuctor arguements are self explanatory 
    def __init__(self, modelName, model, device, imgSizeH, imgSizeW, batchSize, n_classes = 10, rays = True, normalize = None, jelly = False, TiT = False):
        self.modelName = modelName
        self.model = model
        self.imgSizeH = imgSizeH 
        self.imgSizeW = imgSizeW
        self.batchSize = batchSize
        self.resizeTransform = torchvision.transforms.Resize((imgSizeH, imgSizeW))
        self.device = device
        self.n_classes = n_classes
        self.rays = rays
        self.normalize = normalize
        self.jelly = jelly
        self.TiT = TiT

    #Validate a dataset, makes sure that the dataset is the right size before processing
    def validateD(self, dataLoader, mem = True):
        if self.normalize != None:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            x = self.normalize(x)
            dataLoaderFinal = DMP.TensorToDataLoader(x,y, batchSize = self.batchSize)
        else:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            dataLoaderFinal = DMP.TensorToDataLoader(x,y, batchSize = self.batchSize)
        #Put the images in the right size if they are not already
        #dataLoaderFinal = self.formatDataLoader(dataLoader)
        #Make a copy of the model and put it on the GPU
        if mem:
            currentModel = self.model
            currentModel.to(self.device)
            acc = DMP.validateD(dataLoaderFinal, currentModel, jelly = self.jelly, TiT = self.TiT)
            currentModel = currentModel.to("cpu")
            self.model = self.model.to("cpu")
            del currentModel
        else:
            acc = DMP.validateD(dataLoaderFinal, self.model, jelly = self.jelly, TiT = self.TiT)
        #Clean up the GPU memory
        torch.cuda.empty_cache()
        return acc

    #Predict on a dataset, makes sure that the dataset is the right size before processing
    def predictD(self, dataLoader, numClasses = -1, mem = True):
        if numClasses == -1:
            numClasses = self.n_classes
        if self.normalize != None:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            x = self.normalize(x)
            dataLoaderFinal = DMP.TensorToDataLoader(x,y, batchSize = self.batchSize)
        else:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            dataLoaderFinal = DMP.TensorToDataLoader(x,y, batchSize = self.batchSize)
        if mem:
            currentModel = self.model
            currentModel.to(self.device)
            yPred = DMP.predictD(dataLoaderFinal, numClasses, currentModel, jelly = self.jelly, TiT = self.TiT)
            del currentModel
        else:
            yPred = DMP.predictD(dataLoaderFinal, numClasses, self.model, jelly = self.jelly, TiT = self.TiT)
        torch.cuda.empty_cache()
        return yPred

    #Takes in tensor, not data loader
    def predictT(self, xFinal):
        if self.normalize != None:
            xFinal = self.normalize(xFinal)
        bs = self.batchSize
        batches = int(np.ceil(len(xFinal) / bs))
        yPred = torch.ones((len(xFinal), self.n_classes))
        for i in range(batches):
            if self.jelly:
                functional.reset_net(self.model)
            if i == batches-1:
                inx = xFinal[i*bs:]
            else:
                inx = xFinal[i*bs:(i+1)*bs]
            inx = inx.to(self.device)
            pred = self.model(inx)
            if self.jelly:
                pred = pred.mean(0)
            if i == batches-1:
                yPred[i*bs:] = pred.detach()
            else:
                yPred[i*bs:(i+1)*bs] = pred.detach()
        del xFinal
        torch.cuda.empty_cache()
        if self.rays:
            return yPred.detach().cuda()
        else:
            return yPred.detach().cpu()

    #Validate AND generate a model array 
    def validateDA(self, dataLoader, mem = True):
        if self.normalize != None:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            x = self.normalize(x)
            dataLoaderFinal = DMP.TensorToDataLoader(x,y, batchSize = self.batchSize)
        else:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            dataLoaderFinal = DMP.TensorToDataLoader(x,y, batchSize = self.batchSize)
        #Put the images in the right size if they are not already
        #dataLoaderFinal = self.formatDataLoader(dataLoader)
        #Make a copy of the model and put it on the GPU
        if mem:
            currentModel = self.model
            currentModel.to(self.device)
            accArray, acc = DMP.validateDA(dataLoaderFinal, currentModel, jelly = self.jelly, TiT = self.TiT)
            currentModel = currentModel.to("cpu")
            self.model = self.model.to("cpu")
            del currentModel
        else:
            accArray, acc = DMP.validateDA(dataLoaderFinal, self.model, jelly = self.jelly, TiT = self.TiT)
        torch.cuda.empty_cache()
        return accArray, acc

    #Makes sure the inputs are the right size 
    def formatDataLoader(self, dataLoader):
        if self.normalize != None:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            newx = self.normalize(x) #transforms.Resize((self.imgSizeH, self.imgSizeW))(x)
            return DMP.TensorToDataLoader(newx, y, batchSize= self.batchSize)
        else:
            x,y = DMP.DataLoaderToTensor(dataLoader)
            return DMP.TensorToDataLoader(x, y, batchSize= self.batchSize)

    #Go through and delete the main parts that might take up GPU memory
    def clearModel(self):
        print("Warning, model "+self.modelName+" is being deleted and should not be called again!")
        del self.model
        torch.cuda.empty_cache() 

    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self, device):
        self.model.to(device)
    def zero_grad(self):
        self.model.zero_grad()

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        yfake = torch.ones(len(x))
        dataLoader = DMP.TensorToDataLoader(x, yfake)
        dataLoader = self.formatDataLoader(dataLoader)

        numSamples = len(dataLoader.dataset)
        yPred = torch.zeros(numSamples, self.n_classes)
        #switch to evaluate mode
        currentModel = self.model
        currentModel.to(self.device)
        currentModel.eval()
        device = self.device

        indexer = 0
        batchTracker = 0
        for i, (input, target) in enumerate(dataLoader):
            if self.jelly:
                functional.reset_net(currentModel)
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            inputVar = input.to(device)
            output = currentModel(inputVar)
            if self.jelly:
                output = output.mean(0)
            output = output.float()
            yPred[batchTracker - sampleSize : batchTracker] = output

        return yPred.to("cuda")


