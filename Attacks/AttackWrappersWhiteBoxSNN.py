#This is code for the SNN attack taken from other sources by K
import torch
import Utilities.DataManagerPytorch as DMP

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    snnLengthParam = xShape[3]
    #xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    xAdv = torch.zeros((numSamples,) + xShape)
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # Collect datagrad
        #xDataGrad = xDataTemp.grad.data
        ###Here we actual compute the adversarial sample 
        # Collect the element-wise sign of the data gradient
        #signDataGrad = xDataTemp.grad.data.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        #print("xData:", xData.is_cuda)
        #print("SignGrad:", signDataGrad.is_cuda)
        if targeted == True:
            raise ValueError("Targeted attack has not been tested.")
            #perturbedImage = xData - epsilonMax*signDataGrad.cpu().detach() #Go negative of gradient
        else:
            snnGradSign = xDataTemp.grad.data.sum(-1).repeat((snnLengthParam, 1, 1, 1, 1)).permute((1,2,3,4,0)).sign()
            perturbedImage = xData.cpu().detach() + epsilonMax*snnGradSign.cpu().detach()
        #Make sure it does not go outside the bounds of epsilon
        perturbedImage = ProjectionOperation(perturbedImage, xData.cpu().detach(), epsilonMax)
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturbedImage[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xDataTemp
        del snnGradSign
        torch.cuda.empty_cache()
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def PGDNativePytorch(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    snnLengthParam = 10#xShape[3]
    #xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    xAdv = torch.zeros((numSamples,) + xShape)
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #loss = torch.nn.CrossEntropyLoss(reduction = 'none')
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        #Initalize memory for the gradient momentum
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):   
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            if targeted == True:
                raise ValueError("Targeted attack not implemented yet.")
            else:
                snnGrad = xAdvCurrent.grad#.sum(-1)#.repeat((snnLengthParam, 1, 1, 1, 1)).permute((1,2,3,4,0))
                #print(snnGrad.shape, xAdvCurrent.shape)
                advTemp = xAdvCurrent + (epsilonStep*torch.sign(snnGrad)).to(device)
                #advTemp = xAdvCurrent + (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            #Adding clipping to maintain the range
            xAdvCurrent = ProjectionOperation(advTemp, xData.to(device), epsilonMax).detach_()
            xAdvCurrent = torch.clamp(xAdvCurrent, min=clipMin, max=clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def MIMNativePytorch(device, dataLoader, model, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    snnLengthParam = 10#xShape[3]
    xAdv = torch.zeros((numSamples,) + xShape)
    #xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        xClean = xData.to(device)
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
            #gMomentum = decayFactor*gMomentum + GradientNormalizedByL1((1.0/snnLengthParam)*xAdvCurrent.grad.sum(-1)).repeat((snnLengthParam, 1, 1, 1, 1)).permute((1,2,3,4,0))
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample 
            if targeted == True:
                raise ValueError("Targeted attack not implemented yet.")
                #advTemp = xAdvCurrent - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = xAdvCurrent + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Make sure the adversarial example does not exceed the epsilon bounds 
            xAdvCurrent = ProjectionOperation(advTemp, xClean, epsilonMax)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(xAdvCurrent, min=clipMin, max=clipMax).detach_()
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

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

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv
