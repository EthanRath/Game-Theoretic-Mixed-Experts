#DataManagerPytorch = All special Pytorch functions here. Needs torch, torchvision, math, matplotlib, random, os and PIL
#Current Version Number = 1.1 (July 15, 2022), Please do not remove this comment
#Current supported datasets = CIFAR-10, CIFAR-100, Tiny ImageNet (requires image files and path)
import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
import random 
import matplotlib.pyplot as plt
import os 
import PIL
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional

def Fix_Dict(sd):
    out = {}
    for key in sd.keys():
        temp = key
        if "module." in key:
            temp = key[7:]
        out[temp] = sd[key]
    return out

def get_CIFAR10_loaders_test(img_size_H = 32, img_size_W = 32, train = False, bs = 32, norm = True): #transforms used in adversarial work - RM
    if norm:
        transform_test = transforms.Compose([
            transforms.Resize((img_size_H, img_size_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize((img_size_H, img_size_W)),
            transforms.ToTensor(),
        ])
    cifar_test = datasets.CIFAR10("./data", train=train, download=True, transform=transform_test)
    test_loader = DataLoader(cifar_test, batch_size = bs, shuffle=False)
    return test_loader

def get_CIFAR100_loaders_test(img_size_H = 32, img_size_W = 32, train = False, bs = 32): #transforms used in adversarial work - RM
    transform_test = transforms.Compose([
        transforms.Resize((img_size_H, img_size_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ])
    cifar_test = datasets.CIFAR100("./data", train=train, download=True, transform=transform_test)
    test_loader = DataLoader(cifar_test, batch_size = bs, shuffle=False)
    return  test_loader

def get_tiny_loaders_test(img_size_H = 64, img_size_W = 64):
	norm_mean = 0
	norm_var = 1
	transform_test = transforms.Compose([
		transforms.Resize((img_size_H, img_size_W)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_var)])
	test_loader = LoadTinyImageNetValidationData("data\\archive\\tiny-imagenet-200", 64, 8, transform_test)
	return test_loader


#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderIToDataLoaderRE(dataLoaderI, length):
    #First convert the image dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderI)
    #Create memory for the new tensor with repeat encoding 
    xTensorRepeat = torch.zeros(xTensor.shape + (length,))
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        for j in range(0, length):
            xTensorRepeat[i, :, :, :, j] = xTensor[i]
    #New tensor is filled in, convert back to dataloader
    dataLoaderRE = TensorToDataLoader(xTensorRepeat, yTensor, transforms=None, batchSize =dataLoaderI.batch_size, randomizer = None)
    return dataLoaderRE

def TensorIToTensorRE(xTensor, length):

    return xTensor.unsqueeze(4).repeat(1,1,1,1,length)

    xTensorRepeat = torch.zeros(xTensor.shape + (length,))
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        for j in range(0, length):
            xTensorRepeat[i, :, :, :, j] = xTensor[i]
    return xTensorRepeat

def CheckCudaMem():
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Unfree Memory=", a)

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None, jelly = False, TiT = False):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    if not TiT:
        with torch.no_grad():
            #Go through and process the data in batches 
            for i, (input, target) in enumerate(valLoader):
                if jelly:
                    functional.reset_net(model)
                sampleSize = input.shape[0] #Get the number of samples used in each batch
                batchTracker = batchTracker + sampleSize
                if i % 10 == 0:
                    print("Processing up to sample=", batchTracker, end = "\r")
                if device == None: #assume cuda
                    inputVar = input.cuda()
                else:
                    inputVar = input.to(device)
                #compute output
                output = model(inputVar)
                if jelly:
                    output = output.mean(0)
                output = output.float()
                #Go through and check how many samples correctly identified
                for j in range(0, sampleSize):
                    if output[j].argmax(axis=0) == target[j]:
                        acc = acc +1
    else:
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            if jelly:
                functional.reset_net(model)
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            print("Processing up to sample=", batchTracker, end = "\r")
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            if jelly:
                output = output.mean(0)
            output = output.float().detach()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    print()
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None, jelly = False, TiT = False):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    if not TiT:
        with torch.no_grad():
            #Go through and process the data in batches 
            for i, (input, target) in enumerate(valLoader):
                if jelly:
                    functional.reset_net(model)
                sampleSize = input.shape[0] #Get the number of samples used in each batch
                batchTracker = batchTracker + sampleSize
                if i % 100 == 0:
                    print("Processing up to sample=", batchTracker, end = "\r")
                if device == None: #assume CUDA by default
                    inputVar = input.cuda()
                else:
                    inputVar = input.to(device) #use the prefered device if one is specified
                #compute output
                output = model(inputVar)
                if jelly:
                    output = output.mean(0)
                output = output.float()
                #Go through and check how many samples correctly identified
                for j in range(0, sampleSize):
                    if output[j].argmax(axis=0) == target[j]:
                        accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                        accuracy = accuracy + 1
                    indexer = indexer + 1 #update the indexer regardless 
    else:
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            if jelly:
                functional.reset_net(model)
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            if i % 100 == 0:
                print("Processing up to sample=", batchTracker, end = "\r")
            if device == None: #assume CUDA by default
                inputVar = input.cuda()
            else:
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            if jelly:
                output = output.mean(0)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    print()
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray, accuracy

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None, jelly = False, TiT = False):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    if TiT:
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            if jelly:
                functional.reset_net(model)
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            if jelly:
                output = output.mean(0)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    else:
        with torch.no_grad():
            #Go through and process the data in batches 
            for i, (input, target) in enumerate(dataLoader):
                if jelly:
                    functional.reset_net(model)
                sampleSize = input.shape[0] #Get the number of samples used in each batch
                batchTracker = batchTracker + sampleSize
                #print("Processing up to sample=", batchTracker)
                if device == None:
                    inputVar = input.cuda()
                else:
                    inputVar = input.to(device)
                #compute output
                output = model(inputVar)
                if jelly:
                    output = output.mean(0)
                output = output.float()
                for j in range(0, sampleSize):
                    yPred[indexer] = output[j]
                    indexer = indexer + 1 #update the indexer regardless 
        
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Returns the train and val loaders  
def LoadFashionMNISTAsPseudoRGB(batchSize):
    #First transformation, just convert to tensor so we can add in the color channels 
    transformA= transforms.Compose([
        transforms.ToTensor(),
    ])
    #Make the train loader 
    trainLoader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformA), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    numSamplesTrain = len(trainLoader.dataset) 
    sampleIndex = 0
    #This part hard coded for Fashion-MNIST
    xTrain = torch.zeros(numSamplesTrain, 3, 28, 28)
    yTrain = torch.zeros((numSamplesTrain), dtype=torch.long)
    #Go through and process the data in batches 
    for i,(input, target) in enumerate(trainLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xTrain[sampleIndex,0] = input[batchIndex]
            xTrain[sampleIndex,1] = input[batchIndex]
            xTrain[sampleIndex,2] = input[batchIndex]
            yTrain[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    #Make the validation loader 
    valLoader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformA), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    numSamplesTest = len(valLoader.dataset) 
    sampleIndex = 0 #reset the sample index to use with the validation loader 
    #This part hard coded for Fashion-MNIST
    xTest = torch.zeros(numSamplesTest, 3, 28, 28)
    yTest = torch.zeros((numSamplesTest),dtype=torch.long)
    #Go through and process the data in batches 
    for i,(input, target) in enumerate(valLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xTest[sampleIndex,0] = input[batchIndex]
            xTest[sampleIndex,1] = input[batchIndex]
            xTest[sampleIndex,2] = input[batchIndex]
            yTest[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    transform_train = torch.nn.Sequential(
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform_test = torch.nn.Sequential(
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    trainLoaderFinal = TensorToDataLoader(xTrain, yTrain, transform_train, batchSize, True)
    testLoaderFinal = TensorToDataLoader(xTest, yTest, transform_test, batchSize)
    return trainLoaderFinal, testLoaderFinal

#Show 20 images, 10 in first and row and 10 in second row 
def ShowImages(xFirst, xSecond):
    n = 10  # how many digits we will display
    plt.figure(figsize=(5, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xFirst[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(xSecond[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xClean = torch.zeros((numSamples,) + sampleShape)
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation224(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation160(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((160, 128)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR100Validation(imgSize=224, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR100Training(imgSize=224, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='./data', train=True, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader


def GetCIFAR100Validation160(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((160, 128)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation(imgSize = 32, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Training(imgSize = 32, batchSize=128):
    toTensorTransform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    #correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    #xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense(defense, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    #correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    #xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Takes the Tiny ImageNet main dir (as string) as input, imgSize, batchSize and shuffle (true/false)
#Returns the train loader as output 
def LoadTinyImageNetTrainingData(mainDir, imgSize, batchSize, shuffle):
    tinyImageNetTrainDir = mainDir + "//train"
    if imgSize != 64:
        print("Warning: The default size of Tiny ImageNet is 64x64. You are not using this image size for the dataloader.")
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((imgSize, imgSize)),
    transforms.ToTensor()])
    dataset = datasets.ImageFolder(tinyImageNetTrainDir, transform=transform)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return trainLoader


#Takes the Tiny ImageNet main dir (as string) as input, imgSize, batchSize and shuffle (true/false)
#Returns the test loader as output 
def LoadTinyImageNetValidationData(mainDir, imgSize, batchSize):
    #mainDir = "C://Users//kaleel//Desktop//Tiny ImageNet//tiny-imagenet-200"
    #Create the dictionary to get the class labels 
    imgNum = 10000 #This part hardcoded for Tiny ImageNet
    wnidsDir = mainDir + "//tiny-imagenet-200//wnids.txt"
    file1 = open(wnidsDir)
    Lines = file1.readlines() 
    classDict = {} #Start a dictionary for the classes 
    classIndex = 0
    for i in range(0, len(Lines)):
        classDict[Lines[i][0:len(Lines[i])-1]] = classIndex
        classIndex = classIndex + 1 
    #Match validation data with the corresponding labels 
    valLabelDir = mainDir + "//val//val_annotations.txt"
    file2 = open(valLabelDir)
    LinesV = file2.readlines() 
    yData = torch.zeros(imgNum, dtype=torch.long) #Without long type cannot train with cross entropy and PyTorch will throw an error
    #Debugging code
    dirTrainList = mainDir + "//tiny-imagenet-200//train//"
    trainClassArrayStrings = os.listdir(dirTrainList)

    yData = torch.load(mainDir + "//yDataValidation")
    xData = torch.zeros(imgNum, 3, imgSize[0], imgSize[1])
    t = transforms.ToTensor()
    rs = transforms.Resize(imgSize)
    valImageDir = mainDir + "//val//images//"
    for i in range(0, imgNum):
        imgName = valImageDir + "val_"+str(i)+".JPEG"
        #currentImage = cv2.imread(imgName)
        currentImage = PIL.Image.open(imgName)
        xData[i] = t(rs(currentImage))
    finalLoader = TensorToDataLoader(xData, yData, transforms = None, batchSize = batchSize, randomizer = None)
    return finalLoader