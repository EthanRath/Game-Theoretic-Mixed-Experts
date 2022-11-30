import torch
import numpy
import Utilities.DataManagerPytorch as DMP

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
        wList.append(int(numpy.ceil(pList[i]*Niter)))
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
        print(tracker)
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

#Autoattack from June 2022, takes up too much GPU memory
#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def AutoAttackNativePytorchOLD(device, dataLoader, model, epsilonMax, etaStart, numSteps, clipMin, clipMax, targeted):
    #Get the list of checkpoints based on the number of iterations 
    decrement = 0.03
    wList = ComputeCheckPoints(numSteps, decrement)
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    alpha = 0.75 #Weighting factor for momentum 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    tracker = 0
    lossIndividual = torch.nn.CrossEntropyLoss(reduction='none')
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        yK = yData.type(torch.LongTensor).to(device) #Correct class labels which don't change in the iterations
        print("Processing up to sample=", tracker)
        #Initialize the AutoAttack variables
        eta = torch.zeros(numSteps + 1, batchSize) #Keep track of the step size for each sample
        f = torch.zeros(numSteps + 1 , batchSize) #Keep track of the function value for every sample at every step
        z = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        x = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        xBest = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2]) #Best adversarial example thus far
        fBest = torch.zeros(batchSize)
        #Initalize the 0th x and eta
        x[0] = xData 
        eta[0, :] = etaStart
        #Do the attack for a number of steps
        for k in range(0, numSteps):
            xCurrent = x[k].clone().to(device)#.copy()
            #xNext = x[k+1].to(device)
            if k % 25 == 0:
                print("Running step:", k)
            #First attack step handled slightly differently
            if k == 0:
                #Get the model gradient
                xKGrad = GetModelGradient(device, model, xCurrent, yK)
                #xKGrad = GetModelGradient(device, model, x[k], yK)
                #Compute the next adversarial example
                if targeted == True:
                    for i in range(0, batchSize):
                        x[1, i] = x[0, i] - eta[k, i] * torch.sign(xKGrad[i]).cpu()
                else: #targeted is false 
                   for i in range(0, batchSize):
                        x[1, i] = x[0, i] + eta[k, i] * torch.sign(xKGrad[i]).cpu()
                #Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                for i in range(0, batchSize):
                    x[1, i] = torch.clamp(ProjectionOperation(x[1, i], x[0, i], epsilonMax), min=clipMin, max=clipMax)
                #Check which adversarial x is better
                #Original sample
                #outputs = model(x[k].to(device))
                outputs = model(xCurrent)
                model.zero_grad()
                f[k] = lossIndividual(outputs, yK)#.to(device)
                #Next sample
                xNext = x[k+1].clone().to(device)
                outputs = model(xNext)
                #outputs = model(x[k+1].to(device))
                model.zero_grad()
                f[k+1] = lossIndividual(outputs, yK)#.to(device)
                for i in range(0, batchSize):
                    #In the untargeted case we want the cost to increase
                    if f[k+1, i] >= f[k, i] and targeted == False: 
                        xBest[i] = x[k + 1, i]
                        fBest[i] = f[k + 1, i]
                    #In the targeted case we want the cost to decrease
                    elif f[k+1, i] <= f[k, i] and targeted == True:
                         xBest[i] = x[k + 1, i]
                         fBest[i] = f[k + 1, i]
                    #Give a non-zero step size for the next iteration
                    eta[k + 1, i] = eta[k, i] 
            #Not the first iteration of the attack
            else:
                #Get the current x we are working with
                xKGrad = GetModelGradient(device, model, xCurrent, yK)
                #xKGrad = GetModelGradient(device, model, x[k], yK)
                #TODO: Fix the ugly for loops
                if targeted == True:
                    raise ValueError("Didn't implement targeted auto attack yet.")
                else:
                    for i in range(0, batchSize):
                        #Compute zk
                        z[k, i] = x[k, i] + eta[k, i] * torch.sign(xKGrad[i]).cpu()
                        z[k, i] = ProjectionOperation(z[k, i], x[0][i], epsilonMax)
                        #Compute x(k+1) using momentum
                        x[k + 1, i] = x[k, i] + alpha *(z[k, i]-x[k, i]) + (1-alpha)*(x[k, i]-x[k-1, i])
                        x[k + 1, i] =  ProjectionOperation(x[k + 1, i], x[0][i], epsilonMax)          
                        #Apply the clipping operation to make sure xAdv remains in the valid image range
                        x[k + 1, i] = torch.clamp(x[k + 1, i], min=clipMin, max=clipMax)
                #Check which x is better
                xNext = x[k+1].clone().to(device)
                outputs = model(xNext)
                #outputs = model(x[k + 1].to(device))
                model.zero_grad()
                f[k + 1] = lossIndividual(outputs, yK)#.to(device)
                for i in range(0, batchSize):
                    #In the untargeted case we want the cost to increase
                    if f[k+1, i] >= fBest[i] and targeted == False: 
                        xBest[i] = x[k + 1, i]
                        fBest[i] = f[k + 1, i]
                #Now time to do the conditional check to possibly update the step size 
                if k in wList: 
                    #print(k) #For debugging
                    checkPointIndex = wList.index(k) #Get the index of the currentCheckpoint
                    #Go through each element in the batch 
                    for i in range(0, batchSize):
                        conditionOneBoolean = CheckConditionOne(f[:,i], checkPointIndex, wList, targeted)
                        conditionTwoBoolean = CheckConditionTwo(f[:,i], eta[:,i], checkPointIndex, wList)
                        #If either condition is true halve the step size, else use the step size of the last iteration
                        if conditionOneBoolean == True or conditionTwoBoolean == True:           
                            eta[k + 1, i] = eta[k, i] / 2.0
                        else:
                            eta[k + 1, i] = eta[k, i]
                #If we don't need to check the conditions, just repeat the previous iteration's step size
                else:
                    for i in range(0, batchSize):
                        eta[k + 1, i] = eta[k, i] 
            del xCurrent, xNext, xKGrad #Free up the GPU memory for the next run
            torch.cuda.empty_cache() 
            print(torch.cuda.memory_summary())
            ok = 5
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

#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def PGDNativePytorch(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
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
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        #Initalize memory for the gradient momentum
        #gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):   
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term 
            #gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample 
            if targeted == True:
                advTemp = xAdvCurrent - (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            else:
                advTemp = xAdvCurrent + (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

#This one has memory problems but works for the 10 step attack on vanilla ResNet at least
def AutoAttackNativePytorchBACKUP(device, dataLoader, model, epsilonMax, etaStart, numSteps, clipMin, clipMax, targeted):
    print("Don't call can't go above 10 step attack.")
    #Get the list of checkpoints based on the number of iterations 
    decrement = 0.03
    wList = ComputeCheckPoints(numSteps, decrement)
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    alpha = 0.75 #Weighting factor for momentum 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    tracker = 0
    lossIndividual = torch.nn.CrossEntropyLoss(reduction='none')
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        yK = yData.type(torch.LongTensor).to(device) #Correct class labels which don't change in the iterations
        print("Processing up to sample=", tracker)
        #Initialize the AutoAttack variables
        eta = torch.zeros(numSteps + 1, batchSize) #Keep track of the step size for each sample
        f = torch.zeros(numSteps + 1 , batchSize) #Keep track of the function value for every sample at every step
        z = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        x = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        xBest = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2]) #Best adversarial example thus far
        fBest = torch.zeros(batchSize)
        #Initalize the 0th x and eta
        x[0] = xData 
        eta[0, :] = etaStart
        #Do the attack for a number of steps
        for k in range(0, numSteps):
            if k % 25 == 0:
                print("Running step:", k)
            #First attack step handled slightly differently
            if k == 0:
                #Get the model gradient
                xKGrad = GetModelGradient(device, model, x[k], yK)
                #Compute the next adversarial example
                if targeted == True:
                    for i in range(0, batchSize):
                        x[1, i] = x[0, i] - eta[k, i] * torch.sign(xKGrad[i])
                else: #targeted is false 
                   for i in range(0, batchSize):
                        x[1, i] = x[0, i] + eta[k, i] * torch.sign(xKGrad[i])
                #Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                for i in range(0, batchSize):
                    x[1, i] = torch.clamp(ProjectionOperation(x[1, i], x[0, i], epsilonMax), min=clipMin, max=clipMax)
                #Check which adversarial x is better
                #Original sample
                outputs = model(x[k].to(device))
                model.zero_grad()
                f[k] = lossIndividual(outputs, yK)#.to(device)
                #Next sample
                outputs = model(x[k+1].to(device))
                model.zero_grad()
                f[k+1] = lossIndividual(outputs, yK)#.to(device)
                for i in range(0, batchSize):
                    #In the untargeted case we want the cost to increase
                    if f[k+1, i] >= f[k, i] and targeted == False: 
                        xBest[i] = x[k + 1, i]
                        fBest[i] = f[k + 1, i]
                    #In the targeted case we want the cost to decrease
                    elif f[k+1, i] <= f[k, i] and targeted == True:
                         xBest[i] = x[k + 1, i]
                         fBest[i] = f[k + 1, i]
                    #Give a non-zero step size for the next iteration
                    eta[k + 1, i] = eta[k, i] 
            #Not the first iteration of the attack
            else:
                #Get the current x we are working with
                xKGrad = GetModelGradient(device, model, x[k], yK)
                #TODO: Fix the ugly for loops
                if targeted == True:
                    raise ValueError("Didn't implement targeted auto attack yet.")
                else:
                    for i in range(0, batchSize):
                        #Compute zk
                        z[k, i] = x[k, i] + eta[k, i] * torch.sign(xKGrad[i])
                        z[k, i] = ProjectionOperation(z[k, i], x[0][i], epsilonMax)
                        #Compute x(k+1) using momentum
                        x[k + 1, i] = x[k, i] + alpha *(z[k, i]-x[k, i]) + (1-alpha)*(x[k, i]-x[k-1, i])
                        x[k + 1, i] =  ProjectionOperation(x[k + 1, i], x[0][i], epsilonMax)          
                        #Apply the clipping operation to make sure xAdv remains in the valid image range
                        x[k + 1, i] = torch.clamp(x[k + 1, i], min=clipMin, max=clipMax)
                #Check which x is better
                outputs = model(x[k + 1].to(device))
                model.zero_grad()
                f[k + 1] = lossIndividual(outputs, yK)#.to(device)
                for i in range(0, batchSize):
                    #In the untargeted case we want the cost to increase
                    if f[k+1, i] >= fBest[i] and targeted == False: 
                        xBest[i] = x[k + 1, i]
                        fBest[i] = f[k + 1, i]
                #Now time to do the conditional check to possibly update the step size 
                if k in wList: 
                    #print(k) #For debugging
                    checkPointIndex = wList.index(k) #Get the index of the currentCheckpoint
                    #Go through each element in the batch 
                    for i in range(0, batchSize):
                        conditionOneBoolean = CheckConditionOne(f[:,i], checkPointIndex, wList, targeted)
                        conditionTwoBoolean = CheckConditionTwo(f[:,i], eta[:,i], checkPointIndex, wList)
                        #If either condition is true halve the step size, else use the step size of the last iteration
                        if conditionOneBoolean == True or conditionTwoBoolean == True:           
                            eta[k + 1, i] = eta[k, i] / 2.0
                        else:
                            eta[k + 1, i] = eta[k, i]
                #If we don't need to check the conditions, just repeat the previous iteration's step size
                else:
                    for i in range(0, batchSize):
                        eta[k + 1, i] = eta[k, i] 
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
