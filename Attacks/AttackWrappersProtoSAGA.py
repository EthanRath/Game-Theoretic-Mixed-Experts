import torch
import Utilities.DataManagerPytorch as DMP
import numpy as np
from spikingjelly.clock_driven import functional
import cv2
import copy
import ray
import psutil



def MIMNativePytorch_cnn(device, dataLoader, model, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, mean, std, targeted):
    model.eval() #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample: ", tracker, end = "\r")
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        xOridata = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        #Initalize memory for the gradient momentum
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample
            if targeted == True:
                advTemp = (xAdvCurrent * std + mean) - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = (xAdvCurrent * std + mean) + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Adding clipping to maintain the range
            # delta = torch.clamp(advTemp - (xOridata * std + mean), min=-epsilonMax, max=epsilonMax)
            # xAdvCurrent = torch.clamp((xOridata * std + mean) + delta, min=clipMin, max=clipMax).detach_()
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
            xAdvCurrent = (xAdvCurrent - mean) / std
        #Save the adversarial images from the batch
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - mean) / std
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def ComputeAutoAttackCheckpoints(nIter):
    # P list according to the paper
    p = []
    p.append(0)
    p.append(0.22)
    for j in range(1, nIter):
        pCurrent = p[j] + max(p[j] - p[j - 1] - 0.03, 0.06)
        if np.ceil(pCurrent * nIter) <= nIter:
            p.append(pCurrent)
        else:
            break
    # After we make p list can compute the actual checkpoints w[j]
    w = []
    for j in range(0, len(p)):
        w.append(int(np.ceil(p[j] * nIter)))
    # return checkpoints w
    return w

# Projection operation
def ProjectionS(xAdv, x, epsilonMax, clipMin, clipMax):
    return torch.clamp(torch.min(torch.max(xAdv, x - epsilonMax), x + epsilonMax), clipMin, clipMax)

#Main function to run MIME attack
def MIM_EOT_Wrapper(device, dataLoader, model, decayFactor, epsilonMax, numSteps, clipMin, clipMax, targeted, tfunc, numSamples, bs, BaRT = False):
    xShape = [len(dataLoader.dataset)] + list(DMP.GetOutputShape(dataLoader))
    xAdv = torch.zeros(size = xShape)
    print(xAdv.shape)
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = len(xData)
        tracker += batchSize
        print("Processing Up To: ", tracker)
        xAdv[tracker - batchSize: tracker] = MIM_EOT_Batch(device, xData, yData, model, decayFactor, epsilonMax, numSteps, clipMin, clipMax, targeted, tfunc, numSamples, bs)
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    return DMP.TensorToDataLoader(xAdv, yClean)

def MIM_EOT_Batch(device, x,y, model, decayFactor, epsilonMax, numSteps, clipMin, clipMax, targeted, tfunc, numSamples = 100, bs = 8):
    model.eval()
    epsilonStep = epsilonMax / numSteps
    xShape = list(x.shape) #(x.shape[1], x.shape[2], x.shape[3]) #Get the shape of the input (there may be easier way to do this)
    xAdv = copy.copy(x)
    yClean = torch.zeros(numSamples * len(x))
    for i in range(len(y)):
        yClean[i*numSamples: (i+1) * numSamples] = y[i].repeat(numSamples)
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    #Go through each sample 
    gMomentum = torch.zeros(size = xShape)
    for attackStep in range(0, numSteps):
        print("   Inner Iteration: ", attackStep)
        tracker = 0
        gradients = torch.zeros(numSamples * xShape[0], xShape[1], xShape[2], xShape[3])

        xStack = torch.ones(size = [xShape[0] * numSamples, xShape[1], xShape[2], xShape[3]])
        for i in range(len(xAdv)):
            xStack[i*numSamples: (i+1) * numSamples] = xAdv[i].cpu().repeat(numSamples, 1, 1, 1)
        xt = tfunc(xStack, False)
        del(xStack)

        dataLoader = DMP.TensorToDataLoader(xt, yClean, transforms= None, batchSize= bs, randomizer=None)
        model_gpu = model.to("cuda")
        for xData, yData in dataLoader:
            batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
            tracker = tracker + batchSize

            xAdvCurrent = xData.to(device)
            yCurrent = yData.type(torch.LongTensor).to(device)
            xAdvCurrent.requires_grad = True
            outputs = model_gpu(xAdvCurrent)
            model_gpu.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term 
            gradients[tracker-batchSize:tracker] = GradientNormalizedByL1(xAdvCurrent.grad)
            del(xAdvCurrent)
            torch.cuda.empty_cache()

        del(model_gpu)
        torch.cuda.empty_cache()

        tempgrad = torch.zeros(size = gMomentum.shape)
        tracker = 0
        for i in range(len(x)):
            tracker += numSamples
            tempgrad[i] = torch.mean(gradients[tracker-numSamples : tracker], dim = 0, keepdim=True)

        gMomentum = decayFactor*gMomentum + tempgrad
        if targeted == True:
            xAdv = xAdv.to(device) - (epsilonStep*torch.sign(gMomentum)).to(device)
        else:
            xAdv = xAdv.to(device) + (epsilonStep*torch.sign(gMomentum)).to(device)
        xAdv = torch.clamp(xAdv, min=clipMin, max=clipMax).detach_()
    return xAdv

def GradientNormalizedByL1(gradient):
    #Do some basic error checking first
    if gradient.shape[1] != 3:
        raise ValueError("Shape of gradient is not consistent with an RGB image.")
    #basic variable setup
    batchSize = gradient.shape[0]
    colorChannelNum = gradient.shape[1]
    imgRows = gradient.shape[2]
    imgCols = gradient.shape[3]
    gradientNormalized = torch.zeros(batchSize, colorChannelNum, imgRows, imgCols)
    #Compute the L1 gradient for each color channel and normalize by the gradient 
    #Go through each color channel and compute the L1 norm
    for i in range(0, batchSize):
        for c in range(0, colorChannelNum):
           norm = torch.linalg.norm(gradient[i,c], ord=1)
           gradientNormalized[i,c] = gradient[i,c]/norm #divide the color channel by the norm
    return gradientNormalized

# Try to replicate part of the AutoAttack
def PGDNativeAttack(device, dataLoader, model, epsilonMax, numSteps, clipMin, clipMax, targeted):
    model.eval()  # Change model to evaluation mode for the attack
    # Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    epsilonStep = epsilonMax / float(numSteps)
    # Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        # Initalize memory for the gradient momentum
        # Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)
            # Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    # All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                       randomizer=None)  # use the same batch size as the original loader
    return advLoader

# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
#Main implementation of the AE-SAGA attack
def SelfAttentionGradientAttack_EOT(device, epsMax, epsStep, numSteps, modelListPlus, dataLoader, clipMin, clipMax, alphaLearningRate, fittingFactor, advLoader=None, numClasses=10, decay = 0, samples = 4):
    #samples = 2
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    if advLoader is not None:
        xAdv, _ = DMP.DataLoaderToTensor(advLoader)
        dataLoaderCurrent = advLoader
    else:
        xAdv = xClean  # Set the initial adversarial samples
        dataLoaderCurrent = dataLoader
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xOridata = xClean.detach()
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    print('input size: ', xClean.shape)
    confidence = 0
    nClasses = numClasses
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],xShape[2])  # alpha for every model and every sample
    xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])

    for i in range(0, numSteps):
        print("Running step", i)
        print("---------------------------------------------")
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],xShape[2])  # Change to the math here to take in account all objecitve functions
        for m in range(0, len(modelListPlus)):
            if "BaRT" in modelListPlus[m].modelName or "TiT" in modelListPlus[m].modelName:
                nsamp = samples
            else:
                nsamp = 1
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m], samples = nsamp)
            if "ViT" in modelListPlus[m].modelName:
                attmap = GetAttention(dataLoaderCurrent, modelListPlus[m])
                attmap = torch.nn.functional.interpolate(attmap, size=(xShape[1], xShape[2]))
                dCdX[m] = dCdX[m] * attmap
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
        
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        xAdvStepOne = torch.min(xOriMax, xAdvStepOne)
        xAdvStepOne = torch.max(xOriMin, xAdvStepOne)
        xAdvStepOne = torch.clamp(xAdvStepOne, clipMin, clipMax)
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,batchSize=dataLoader.batch_size, randomizer=None)
        for m in range(0, len(modelListPlus)):
            if "BaRT" in modelListPlus[m].modelName or "TiT" in modelListPlus[m].modelName:
                nsamp = samples
            else:
                nsamp = 1
            dFdX = dFdX + torch.nn.functional.interpolate(dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses, samples = nsamp),size=(xShape[1], xShape[2]))
        
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        alpha = alpha - dFdAlpha * alphaLearningRate
        
        xGradientCumulativeTemp = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeTemp = xGradientCumulativeTemp + alpha[m] * dCdX[m]
        if decay == 0:
            xGradientCumulativeB = xGradientCumulativeTemp
        else:
            xGradientCumulativeB = (decay * xGradientCumulativeB) + xGradientCumulativeTemp
        
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        xAdv = torch.min(xOriMax, xAdv)
        xAdv = torch.max(xOriMin, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,randomizer=None)
    
    return dataLoaderCurrent

# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttackProto_Old(device, epsMax, epsStep, numSteps, modelListPlus, dataLoader, clipMin, clipMax, alphaLearningRate, fittingFactor, advLoader=None, numClasses=10, decay = 0):
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    if advLoader is not None:
        xAdv, _ = DMP.DataLoaderToTensor(advLoader)
        dataLoaderCurrent = advLoader
    else:
        xAdv = xClean  # Set the initial adversarial samples
        dataLoaderCurrent = dataLoader
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xOridata = xClean.detach()
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    print('input size: ', xClean.shape)
    confidence = 0
    nClasses = numClasses
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],xShape[2])  # alpha for every model and every sample
    xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])

    for i in range(0, numSteps):
        print("Running step", i)
        print("---------------------------------------------")
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],xShape[2])  # Change to the math here to take in account all objecitve functions
        for m in range(0, len(modelListPlus)):
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            if "ViT" in modelListPlus[m].modelName:
                attmap = GetAttention(dataLoaderCurrent, modelListPlus[m])
                attmap = torch.nn.functional.interpolate(attmap, size=(xShape[1], xShape[2]))
                dCdX[m] = dCdX[m] * attmap
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
        
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        xAdvStepOne = torch.min(xOriMax, xAdvStepOne)
        xAdvStepOne = torch.max(xOriMin, xAdvStepOne)
        xAdvStepOne = torch.clamp(xAdvStepOne, clipMin, clipMax)
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,batchSize=dataLoader.batch_size, randomizer=None)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),size=(xShape[1], xShape[2]))
        
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        alpha = alpha - dFdAlpha * alphaLearningRate

        
        xGradientCumulativeTemp = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeTemp = xGradientCumulativeTemp + alpha[m] * dCdX[m]
        xGradientCumulativeB = (decay * xGradientCumulativeB) + xGradientCumulativeTemp
        
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        xAdv = torch.min(xOriMax, xAdv)
        xAdv = torch.max(xOriMin, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,randomizer=None)
    return dataLoaderCurrent

def ComputePList(pList, startIndex, decrement):
    #p(j+1) = p(j) + max( p(j) - p(j-1) -0.03, 0.06))
    nextP = pList[startIndex] + max(pList[startIndex] - pList[startIndex-1] - decrement, 0.06)
    #Check for base case
    if nextP>= 1.0:
        return pList
    else:
        #Need to further recur
        pList.append(nextP)
        ComputePList(pList, startIndex+1, decrement)

def ComputeCheckPoints(Niter, decrement):
    #First compute the pList based on the decrement amount
    pList = [0, 0.22] #Starting pList based on AutoAttack paper
    ComputePList(pList, 1, decrement)
    #Second compute the checkpoints from the pList
    wList = []
    for i in range(0, len(pList)):
        wList.append(int(np.ceil(pList[i]*Niter)))
    #There may duplicates in the list due to rounding so finally we remove duplicates
    wListFinal = []
    for i in wList:
        if i not in wListFinal:
            wListFinal.append(i)
    #Return the final list
    return wListFinal

def GetAttention(dLoader, modelPlus):
    dLoader = modelPlus.formatDataLoader(dLoader)
    numSamples = len(dLoader.dataset)
    attentionMaps = torch.zeros(numSamples, modelPlus.imgSizeH, modelPlus.imgSizeW, 3)
    currentIndexer = 0
    model = modelPlus.model.to(modelPlus.device)
    with torch.no_grad():
        for ii, (x, y) in enumerate(dLoader):
            x = x.to(modelPlus.device)
            y = y.to(modelPlus.device)
            bsize = x.size()[0]
            currentIndexer += bsize
            attentionMapBatch = get_attention_map(model, x, bsize)

            attentionMaps[currentIndexer - bsize : currentIndexer] = attentionMapBatch
 
    del model
    torch.cuda.empty_cache()
    attentionMaps = attentionMaps.permute(0, 3, 1, 2)
    return attentionMaps

def get_attention_map(model, xbatch, batch_size, img_size=224):
    attentionMaps = torch.zeros(batch_size, img_size, img_size, 3)
    index = 0
    for i in range(0, batch_size):
        ximg = xbatch[i].cpu().numpy().reshape(1, 3, img_size, img_size)
        ximg = torch.tensor(ximg).cuda()
        model.eval()
        _, att_mat = model.forward2(ximg)
        #print(att_mat)
        # model should return attention_list for all attention_heads
        # each element in the list contains attention for each layer

        att_mat = torch.stack(att_mat).squeeze(1)
        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat.cpu().detach() + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (img_size, img_size))[..., np.newaxis]
        mask = np.concatenate((mask,) * 3, axis=-1)
        # print(mask.shape)
        attentionMaps[index] = torch.from_numpy(mask)
        index = index + 1
    return attentionMaps

# Compute dX/dAlpha for each model m
def dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, numModels, numSamples, xShape, super = False):
    # Allocate memory for the solution
    dXdAlpha = torch.zeros(numModels, numSamples, xShape[0], xShape[1], xShape[2])
    innerSum = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    # First compute the inner summation sum m=1,...M: a_{m}*dC/dX_{m}
    for m in range(0, numModels):
        innerSum = innerSum + alpha[m] * dCdX[m]
    # Multiply inner sum by the fitting factor to approximate the sign(.) function
    innerSum = innerSum * fittingFactor
    # Now compute the sech^2 of the inner sum
    innerSumSecSquare = SechSquared(innerSum)
    # Now do the final computation to get dX/dAlpha (may not actually need for loop)
    for m in range(0, numModels):
        if super:
            dXdAlpha[m] = fittingFactor * epsStep[:,None][:,None][:,None] * dCdX[m] * innerSumSecSquare
        else:
            dXdAlpha[m] = fittingFactor * epsStep * dCdX[m] * innerSumSecSquare
    # All done so return
    return dXdAlpha

# Compute sech^2(x) using torch functions
def SechSquared(x):
    y = 4 * torch.exp(2 * x) / ((torch.exp(2 * x) + 1) * (torch.exp(2 * x) + 1))
    return y

# Custom loss function for updating alpha
def UntargetedCarliniLoss(logits, targets, confidence, nClasses, device):
    # This converts the normal target labels to one hot vectors e.g. y=1 will become [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    yOnehot = torch.nn.functional.one_hot(targets, nClasses).to(torch.float)
    zC = torch.max(yOnehot * logits, 1).values  # Need to use .values to get the Tensor because PyTorch max function doesn't want to give us a tensor
    zOther = torch.max((1 - yOnehot) * logits, 1).values
    loss = torch.max(zC - zOther + confidence, torch.tensor(0.0).to(device))
    return loss

# Native (no attack library) implementation of the FGSM attack in Pytorch
def FGSMNativeGradient(device, dataLoader, modelPlus, samples = 1):
    samples = 2
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    for sample in range(samples):
        sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
        numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
        xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
        xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        batchSize = 0
        tracker = 0
        i = 0
        for xData, yData in sizeCorrectedLoader:
            i += 1
            batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
            tracker = tracker + batchSize
            if i % 5 == 0:
                print("Processing up to sample=", tracker, end = "\r")
            
            xDataTemp = xData.detach().to(device)
            yData = yData.type(torch.LongTensor).to(device)
            xDataTemp.requires_grad = True
            
            if "jelly" in modelPlus.modelName: #For spiking jelly
                functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
                output = model(xDataTemp).mean(0)
            else:
                output = model(xDataTemp)
            
            model.zero_grad()
            cost = loss(output, yData).to(device)
        
            cost.backward()
            if modelPlus.modelName == 'SNN VGG-16 Backprop': #for haowen backprop
                xDataTempGrad = xDataTemp.grad.data.sum(-1)
            else:
                xDataTempGrad = xDataTemp.grad.data
            
            xGradient[tracker - batchSize: tracker] += xDataTempGrad.cpu()            
            del xDataTemp
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    print()
    return xGradient/samples

def dFdXCompute(device, dataLoader, modelPlus, confidence, nClasses, samples = 1):
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device) 
    samples = 1 # set samples to 1 because we don't have time
    for sample in range(samples):
        sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
        numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
        xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
        xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        batchSize = 0
        tracker = 0
        i = 0
        for xData, yData in sizeCorrectedLoader:
            batchSize = xData.shape[0] 
            tracker = tracker + batchSize
            i += 1
            if i % 5 == 0:
                print("Processing up to sample=", tracker, end = "\r")
            
            xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
            yData = yData.type(torch.LongTensor).to(device)
            xDataTemp.requires_grad = True
            
            if "jelly" in modelPlus.modelName:
                functional.reset_net(model)
                outputLogits = model(xDataTemp).mean(0)
            else:
                outputLogits = model(xDataTemp)
            
            model.zero_grad()
            cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device)
            cost = cost.sum().to(device)

            cost.backward()
            
            if modelPlus.modelName == 'SNN VGG-16 Backprop':
                xDataTempGrad = xDataTemp.grad.data.sum(-1)
            else:
                xDataTempGrad = xDataTemp.grad.data
            
            xGradient[tracker - batchSize: tracker] += xDataTempGrad.cpu()
            
            del xDataTemp
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    print()
    return xGradient/samples


def CheckCarliniLoss(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    cumulativeCost = 0
    i = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        i += 1
        if i % 5 == 0:
            print("Processing up to sample=", tracker, end = "\r")
        # print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Forward pass the data through the model
        if "jelly" in modelPlus.modelName:
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device).sum()  # Not sure about the sum
        cumulativeCost = cumulativeCost + cost.to("cpu")
        cost.backward()
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return cumulativeCost

# Get the loss associated with single samples
def CarliniSingleSampleLoss(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    # Variables to store the associated costs values
    costValues = torch.zeros(numSamples)
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    i = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        if "jelly" in modelPlus.modelName:
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device)
        cost.sum().backward()
        # Store the current cost values
        costValues[tracker:tracker + batchSize] = cost.to("cpu")
        tracker = tracker + batchSize
        i += 1
        if i % 5 == 0:
            print("Processing up to sample=", tracker, end = "\r")
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return costValues

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#Function for computing the model gradient
def GetModelGradient(device, model, xK, yK):
    #Define the loss function
    loss = torch.nn.CrossEntropyLoss()
    xK.requires_grad = True
    #Pass the inputs through the model 
    outputs = model(xK.to(device))
    model.zero_grad()
    #Compute the loss 
    cost = loss(outputs, yK)
    cost.backward()
    xKGrad = xK.grad
    ##Do GPU memory clean up (important)
    #del xK
    #del cost
    #del outputs
    #del loss
    return xKGrad

def ComputePList(pList, startIndex, decrement):
    #p(j+1) = p(j) + max( p(j) - p(j-1) -0.03, 0.06))
    nextP = pList[startIndex] + max(pList[startIndex] - pList[startIndex-1] - decrement, 0.06)
    #Check for base case 
    if nextP>= 1.0:
        return pList
    else:
        #Need to further recur
        pList.append(nextP)
        ComputePList(pList, startIndex+1, decrement)

def ComputeCheckPoints(Niter, decrement):
    #First compute the pList based on the decrement amount
    pList = [0, 0.22] #Starting pList based on AutoAttack paper
    ComputePList(pList, 1, decrement)
    #Second compute the checkpoints from the pList
    wList = []
    for i in range(0, len(pList)):
        wList.append(int(np.ceil(pList[i]*Niter)))
    #There may duplicates in the list due to rounding so finally we remove duplicates
    wListFinal = []
    for i in wList:
        if i not in wListFinal:
            wListFinal.append(i)
    #Return the final list
    return wListFinal

#Condition two checks if the objective function and step size previously changed
def CheckConditionTwo(f, eta, checkPointIndex, checkPoints):
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex-1] #Get the previous checkpoint
    if eta[previousCheckPoint] == eta[currentCheckPoint] and f[previousCheckPoint] == f[currentCheckPoint]:
        return True
    else:
        return False

#Condition one checks the summation of objective function
def CheckConditionOne(f, checkPointIndex, checkPoints, targeted):
    sum = 0
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex-1] #Get the previous checkpoint
    #See how many times the objective function was growing bigger 
    for i in range(previousCheckPoint, currentCheckPoint): #Goes from w_(j-1) to w_(j) - 1
        if f[i+1] > f[i] :
            sum = sum + 1
    ratio = 0.75 * (currentCheckPoint - previousCheckPoint)
    #For untargeted attack we want the objective function to increase
    if targeted == False and sum < ratio: #This is condition 1 from the Autoattack paper
        return True
    elif targeted == True and sum > ratio: #This is my interpretation of how the targeted attack would work (not 100% sure)
        return True
    else:
        return False

#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
#This implementaiton is very GPU memory intensive
def AutoAttackNativePytorch(device, dataLoader, model, epsilonMax, etaStart, numSteps, clipMin, clipMax, targeted):
    #Setup attack variables:
    decrement = 0.03
    wList = ComputeCheckPoints(numSteps, decrement) #Get the list of checkpoints based on the number of iterations 
    alpha = 0.75 #Weighting factor for momentum 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 #Indexing variable for saving the adversarial example 
    batchSize = 0 #just do dummy initalization, will be filled in later
    tracker = 0
    #lossIndividual = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval() #Change model to evaluation mode for the attack 
    #Go through each batch and run the attack
    for xData, yData in dataLoader:
        #Initialize the AutoAttack variables
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize #Update the tracking variable 
        print(tracker, end = "\r")
        yK = yData.type(torch.LongTensor).to(device) #Correct class labels which don't change in the iterations
        eta = torch.zeros(numSteps + 1, batchSize) #Keep track of the step size for each sample
        eta[0, :] = etaStart #Initalize eta values as the starting eta for each sample in the batch 
        f = torch.zeros(numSteps + 1 , batchSize) #Keep track of the function value for every sample at every step
        z = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        x = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        x[0] = xData #Initalize the starting adversarial example as the clean example 
        xBest = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2]) #Best adversarial example thus far
        fBest = torch.zeros(batchSize) #Best value of the objective function thus far
        #Do the attack for a number of steps
        for k in range(0, numSteps):
            lossIndividual = torch.nn.CrossEntropyLoss(reduction='none')
            #First attack step handled slightly differently
            if k == 0:
                xKGrad = GetModelGradient(device, model, x[k], yK) #Get the model gradient
                if targeted == True:
                    raise ValueError("Targeted Auto-Attack not yet implemented.")
                else: #targeted is false
                    for b in range(0, batchSize):
                        x[1, b] = x[0, b] + eta[k, b] * torch.sign(xKGrad[b]).cpu() #here we use index 1 because the 0th index is the clean sample
                #Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                for b in range(0, batchSize):
                    x[1, b] = torch.clamp(ProjectionOperation(x[1, b], x[0, b], epsilonMax), min=clipMin, max=clipMax)
                #Check which adversarial x is better, the clean x or the new adversarial x 
                outputsOriginal = model(x[k].to(device))
                model.zero_grad()
                f[0] = lossIndividual(outputsOriginal, yK).cpu().detach() #Store the value in the objective function array
                outputs = model(x[k+1].to(device))
                model.zero_grad()
                f[1] = lossIndividual(outputs, yK).cpu().detach() #Store the value in the objective function array
                for b in range(0, batchSize):
                    #In the untargeted case we want the cost to increase
                    if f[k+1, b] >= f[k, b] and targeted == False: 
                        xBest[b] = x[k + 1, b]
                        fBest[b] = f[k + 1, b]
                    #In the targeted case we want the cost to decrease
                    elif f[k+1, b] <= f[k, b] and targeted == True:
                         xBest[b] = x[k + 1, b]
                         fBest[b] = f[k + 1, b]
                    #Give a non-zero step size for the next iteration
                    eta[k + 1, b] = eta[k, b]
            #Not the first iteration of the attack
            else:
                xKGrad = GetModelGradient(device, model, x[k], yK)
                if targeted == True:
                    raise ValueError("Didn't implement targeted auto attack yet.")
                else:
                    for b in range(0, batchSize):
                        #Compute zk
                        z[k, b] = x[k, b] + eta[k, b] * torch.sign(xKGrad[b]).cpu()
                        z[k, b] = ProjectionOperation(z[k, b], x[0, b], epsilonMax)
                        #Compute x(k+1) using momentum
                        x[k + 1, b] = x[k, b] + alpha *(z[k, b]-x[k, b]) + (1-alpha)*(x[k, b]-x[k-1, b])
                        x[k + 1, b] =  ProjectionOperation(x[k + 1, b], x[0, b], epsilonMax)          
                        #Apply the clipping operation to make sure xAdv remains in the valid image range
                        x[k + 1, b] = torch.clamp(x[k + 1, b], min=clipMin, max=clipMax)
                #Check which x is better
                outputs = model(x[k+1].to(device))
                model.zero_grad()
                f[k + 1] = lossIndividual(outputs, yK).cpu().detach()
                for b in range(0, batchSize):
                    #In the untargeted case we want the cost to increase
                    if f[k+1, b] >= fBest[b] and targeted == False: 
                        xBest[b] = x[k + 1, b]
                        fBest[b] = f[k + 1, b]
                #Now time to do the conditional check to possibly update the step size 
                if k in wList: 
                    #print(k) #For debugging
                    checkPointIndex = wList.index(k) #Get the index of the currentCheckpoint
                    #Go through each element in the batch 
                    for b in range(0, batchSize):
                        conditionOneBoolean = CheckConditionOne(f[:,b], checkPointIndex, wList, targeted)
                        conditionTwoBoolean = CheckConditionTwo(f[:,b], eta[:,b], checkPointIndex, wList)
                        #If either condition is true halve the step size, else use the step size of the last iteration
                        if conditionOneBoolean == True or conditionTwoBoolean == True:           
                            eta[k + 1, b] = eta[k, b] / 2.0
                        else:
                            eta[k + 1, b] = eta[k, b]
                #If we don't need to check the conditions, just repeat the previous iteration's step size
                else:
                    for b in range(0, batchSize):
                        eta[k + 1, b] = eta[k, b] 
            #Memory clean up
            del lossIndividual
            del outputs
            torch.cuda.empty_cache() 
        #Save the adversarial images from the batch 
        for i in range(0, batchSize):
            #print("==========")
            #print(eta[:,i])
            #print("==========")
            xAdv[advSampleIndex] = xBest[i]
            yClean[advSampleIndex] = yData[i]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader