import torch
import Utilities.DataManagerPytorch as DMP
import numpy as np
from spikingjelly.clock_driven import functional
import cv2

# This is terribly coded
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


# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttackProtoAuto(device, epsMax, numSteps, modelListPlus, dataLoader, clipMin, clipMax, alphaLearningRate, fittingFactor):
    print("Using hard coded experimental function not advisable.")
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean  # Set the initial adversarial samples
    xOridata = xClean.to("cpu").detach()
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    epsStep = .005 #from SNN paper #epsMax / numSteps
    dataLoaderCurrent = dataLoader
    confidence = 0
    nClasses = 10
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],xShape[2])  

    for i in range(0, numSteps):
        print("Running step", i)
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],xShape[2])
        
        #--- To Combine ---
        for m in range(0, len(modelListPlus)):
            dataLoaderCurrent = modelListPlus[m].formatDataLoader(dataLoaderCurrent)
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
        
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,batchSize=dataLoader.batch_size, randomizer=None)
        print("===Pre-Alpha Optimization===")
        costMultiplier = torch.zeros(len(modelListPlus), numSamples)
        
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses)
            costMultiplier[m] = CarliniSingleSampleLoss(device, dataLoaderStepOne, modelListPlus[m], confidence,nClasses)
            print("For model", m, "the Carlini value is", cost)
        
        # Compute dF/dX (cumulative)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(
                dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),
                size=(xShape[1], xShape[2]))
        #---To Combnine ---
        
        # Compute dX/dAlpha
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        # Compute dF/dAlpha = dF/dx * dX/dAlpha
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # dFdAlpha = dFdX * dXdAlpha[m]
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        
        # Now time to update alpha
        alpha = alpha - dFdAlpha * alphaLearningRate
        # Compute final adversarial example using best alpha
        xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]
        
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        xAdv = torch.min(xOridata + epsMax, xAdv)
        xAdv = torch.max(xOridata - epsMax, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the current xAdv to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)
        # Debug HERE
        print("===Post-Alpha Optimization===")
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderCurrent, modelListPlus[m], confidence, nClasses)
            print("For model", m, "the Carlini value is", cost)
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
    for ii, (x, y) in enumerate(dLoader):
        x = x.to(modelPlus.device)
        y = y.to(modelPlus.device)
        bsize = x.size()[0]
        attentionMapBatch = get_attention_map(model, x, bsize)
        # for i in range(0, dLoader.batch_size):
        for i in range(0, bsize):
            attentionMaps[currentIndexer] = attentionMapBatch[i]
            currentIndexer = currentIndexer + 1
    del model
    torch.cuda.empty_cache()
    print("attention maps generated")
    # change order
    attentionMaps = attentionMaps.permute(0, 3, 1, 2)
    return attentionMaps

def get_attention_map(model, xbatch, batch_size, img_size=224):
    attentionMaps = torch.zeros(batch_size, img_size, img_size, 3)
    index = 0
    for i in range(0, batch_size):
        ximg = xbatch[i].cpu().numpy().reshape(1, 3, img_size, img_size)
        ximg = torch.tensor(ximg).cuda()
        model.eval()
        res, att_mat = model.forward2(ximg)
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
def dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, numModels, numSamples, xShape):
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
def FGSMNativeGradient(device, dataLoader, modelPlus, getcost = False):
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
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
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
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
        if "jelly" in modelPlus.modelName: #For spiking jelly
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            output = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            output = model(xDataTemp)
        
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        if getcost:
            cost_mat = loss(output, yData, reduction = "none").item().to(device)
        cost.backward()
        if modelPlus.modelName == 'SNN VGG-16 Backprop': #for haowen backprop
            xDataTempGrad = xDataTemp.grad.data.sum(-1)
        else:
            xDataTempGrad = xDataTemp.grad.data

        # Save the adversarial images from the batch
        xGradient[tracker - batchSize: tracker] = xDataTempGrad
        yClean[tracker - batchSize: tracker] = yData
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    if getcost:
        return xGradient, cost_mat
    return xGradient

def dFdXCompute(device, dataLoader, modelPlus, confidence, nClasses):
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
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
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
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
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
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device).sum().to(
            device)  # Not sure about the sum
        cost.backward()
        if modelPlus.modelName == 'SNN VGG-16 Backprop':
            xDataTempGrad = xDataTemp.grad.data.sum(-1)
        else:
            xDataTempGrad = xDataTemp.grad.data
        # Save the adversarial images from the batch
        xGradient[tracker - batchSize: tracker] = xDataTempGrad
        yClean[tracker - batchSize: tracker] = yData
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return xGradient


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

def AutoAutoSAGA(device, epsMax, numSteps, modelListPlus, dataLoader, clipMin, clipMax, alphaLearningRate, fittingFactor):
    print("Using hard coded experimental function not advisable.")
    # Basic graident variable setup
    wlist = ComputeCheckPoints(numSteps, .03)
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean  # Set the initial adversarial samples

    xBest = xClean #Best X so far
    costBest = 0#torch.zeros(len(xClean)) - 1 #best X cost so far
    prevcost = 0
    costmat = -1
    improvedcount = -1

    xOridata = xClean.to(device).detach()
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    # Compute eps step
    epsStep = torch.ones(len(xClean))*(epsMax / numSteps)
    dataLoaderCurrent = dataLoader
    # Hardcoded for alpha right now, put in the method later
    confidence = 0
    nClasses = 10
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],
                       xShape[2])  # alpha for every model and every sample
    # End alpha setup
    numSteps = 10
    for i in range(0, numSteps):
        print("Running step", i)

        if i in wlist:
            ind = wlist.index(i)
            #condition 1
            cond1 = improvedcount >= .75*(wlist[ind] - wlist[ind - 1])

        # Keep track of dC/dX for each model where C is the cross entropy function
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        # Keep track of dF/dX for each model where F, is the Carlini-Wagner loss function (for updating alpha)
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],
                           xShape[2])  # Change to the math here to take in account all objecitve functions
        # Go through each model and compute dC/dX


        for m in range(0, len(modelListPlus)):
            dataLoaderCurrent = modelListPlus[m].formatDataLoader(dataLoaderCurrent)
            dCdXTemp, tepcostmat = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m], True)
            if m == 0:
                costmat = tempcostmat.detach()
            else:
                costmat += tempcostmat.detach()
            del(tempcostmat)
            # Resize the graident to be the correct size and save it
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
            # Now compute the inital adversarial example with the base alpha
        costmat /= len(modelListPlus) #cost mat weights each model evenly

        if i == 0:
            costBest = costmat
            improved = costBest <= costmat
            prevcost = costmat
            improvedcount = prevcost == costmat
        else:
            improved = costBest <= costmat
            cotBest = torch.maximum(costBest, costmat)
            improvedcount += prevcost < costmat
            #costBest = (costBest * (1 - improved)) + (costmat* improved) #replace costBest with improvements
            xBest = (xBest * (1 - improved)) + (xAdv * improved)

        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        # Convert the current xAdv to dataloader
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,
                                                   batchSize=dataLoader.batch_size, randomizer=None)
        print("===Pre-Alpha Optimization===")
        costMultiplier = torch.zeros(len(modelListPlus), numSamples)
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses)
            costMultiplier[m] = CarliniSingleSampleLoss(device, dataLoaderStepOne, modelListPlus[m], confidence,
                                                        nClasses)
            print("For model", m, "the Carlini value is", cost)
        # Compute dF/dX (cumulative)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(
                dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),
                size=(xShape[1], xShape[2]))
        # Compute dX/dAlpha
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        # Compute dF/dAlpha = dF/dx * dX/dAlpha
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # dFdAlpha = dFdX * dXdAlpha[m]
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        # Now time to update alpha
        alpha = alpha - dFdAlpha * alphaLearningRate
        # Compute final adversarial example using best alpha
        xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        xAdv = torch.min(xOridata + epsMax, xAdv)
        xAdv = torch.max(xOridata - epsMax, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the current xAdv to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
        # Debug HERE
        print("===Post-Alpha Optimization===")
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderCurrent, modelListPlus[m], confidence, nClasses)
            print("For model", m, "the Carlini value is", cost)
    return dataLoaderCurrent
