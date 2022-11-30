# %%
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Utilities import DataManagerPytorch as DMP

# %%

def get_CIFAR10_loaders_test(r): #transforms used in adversarial work - RM
    img_size_H = 32
    img_size_W = 32
    transform_test = transforms.Compose([
        transforms.Resize((img_size_H, img_size_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
    ])
    cifar_test = datasets.CIFAR10(r, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(cifar_test, batch_size = 64, shuffle=False)
    return test_loader


class threshold_rect(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - 1.0) < 0.5
        return grad_input * temp.float()

class threshold_logistic(torch.autograd.Function):
    """
    heaviside step threshold function
    """

    @staticmethod
    def forward(ctx, input):
        """

        """
        # a asjusts the max value of gradient and sharpness
        # b moves gradient to left (positive) or right (negative)
        a = 4 # set to 4 as it sets max value to 1
        b = -1
        ctx.save_for_backward(input)
        ctx.a = a
        ctx.b = b

        output = input.gt(1.0).float()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """

        """

        # a = time.time()
        input, = ctx.saved_tensors
        a = ctx.a
        b = ctx.b

        x = input
        logictic_ = a * torch.exp(-a * (b + x))/((torch.exp(-a*(b+x))+1)**2)

        grad = logictic_ * grad_output

        return grad, None

class if_encoder(torch.nn.Module):
    def __init__(self, step_num, max_rate = 1.0, threshold=1.0, reset_mode='soft', forward_choice=1):
        '''
        :param step_num:
        :param max_rate:
        :param threshold:
        :param reset_mode: 'soft' or 'hard'
        :param forward_choice: select different forward functions
        '''

        super().__init__()
        self.step_num = step_num
        self.reset_mode = reset_mode
        self.max_rate = max_rate
        self.threshold = threshold

        self.threshold = torch.nn.Parameter(torch.tensor(1.0))

        self.threshold.requires_grad = False

        # 1 does not support bp
        # 2 and 3 support bp
        if forward_choice == 1:
            self.forward_func = self.forward_1
        elif forward_choice == 2:
            self.forward_func = self.forward_2
        elif forward_choice == 3:
            self.forward_func = self.forward_3
    
    def forward(self, x):

        return self.forward_func(x)

    def forward_1(self, x):
        """
        no gradient approximation
        :param x: [batch, c, h, w], assume image is scaled in range [0,1]
        :return: shape [b,c,h,w,step_num]
        """

        spikes = []

        v = torch.zeros_like(x)
        for i in range(self.step_num):

            v = v + x*self.max_rate

            spike = v.clone()

            spike[spike < self.threshold] = 0.0
            spike[spike >= self.threshold] = 1.0

            if self.reset_mode == 'soft':
                v[v >= self.threshold] = v[v >= self.threshold] - self.threshold
            else:
                v[v >= self.threshold] = 0.0

            spikes += [spike]

        return torch.stack(spikes,dim=-1)
    
    def forward_2(self, x):
        """
        gradient approximation same as stbp
        """
        spikes = []

        v = torch.zeros_like(x)
        spike = torch.zeros_like(x)

        for i in range(self.step_num):

            if self.reset_mode == 'soft':
                v = v + x*self.max_rate - spike
            else:
                v = (1-spike) * v + x*self.max_rate
            
            threshold_function = threshold_rect.apply
            spike = threshold_function(v)

            spikes += [spike]

        return torch.stack(spikes,dim=-1)

    def forward_3(self, x):
        """
        use logistic function to approximate gradient
        :param x: [batch, c, h, w], assume image is scaled in range [0,1]
        :return: shape [b,c,h,w,step_num]
        """

        spikes = []

        v = torch.zeros_like(x)
        for i in range(self.step_num):

            v = v + x*self.max_rate

            threshold_function = threshold_logistic.apply
            spike = threshold_function(v)

            if self.reset_mode == 'soft':
                v = v - spike
            else:
                v = (1-spike)*v

            spikes += [spike]

        return torch.stack(spikes,dim=-1)


class signed_if_encoder(torch.nn.Module):
    def __init__(self, step_num, max_rate = 1.0, threshold=1.0, reset_mode='soft'):
        '''
        :param step_num:
        :param max_rate:
        :param threshold:
        :param reset_mode: 'soft' or 'hard'
        '''

        super().__init__()
        self.step_num = step_num
        self.reset_mode = reset_mode
        self.max_rate = max_rate
        self.threshold = threshold

        self.threshold = torch.nn.Parameter(torch.tensor(1.0))

        self.threshold.requires_grad = False

    def forward(self, x):
        """
        :param x: [batch, c, h, w], assume image is scaled in range [0,1]
        :return: shape [b,c,h,w,step_num]
        """

        spikes = []

        v = torch.zeros_like(x)
        for i in range(self.step_num):

            v = v + x*self.max_rate

            spike = torch.zeros_like(v)

            positive = v >= self.threshold
            negative = v <= -self.threshold

            spike[positive] = 1.0
            spike[negative] = -1.0

            if self.reset_mode == 'soft':
                v[positive] = v[positive] - self.threshold
                v[negative] = v[negative] + self.threshold
            else:
                v[positive] = 0.0
                v[negative] = 0.0

            spikes += [spike]

        return torch.stack(spikes,dim=-1)

class PoissonGenerator(torch.nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*0.9).float(),torch.sign(input))
		return out
    
def GetFirstCorrectlyOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusList, size, bs, inc = False):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(dataLoader)[0]
        accArrayCumulative = accArrayCumulative + accArray
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, size[0], size[1])
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    # Go through all the samples
    included = torch.zeros(totalSampleNum)
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArrayCumulative[i] == numModels and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer = sampleIndexer + 1  # update the indexer
            samplePerClassCount[currentClass] = samplePerClassCount[currentClass] + 1  # Update the number of samples for this class
            included[i] = 1
    # Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    # Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            print(samplePerClassCount[i])
            raise ValueError("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms=None, batchSize=bs,
                                             randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    """for i in range(0, numModels):
        cleanAcc = modelPlusList[i].validateD(cleanDataLoader)
        if cleanAcc != 1.0:
            print("Clean Acc " + modelPlusList[i].modelName + ":", cleanAcc)
            print("The clean accuracy is not 1.0")"""
    # All error checking done, return the clean balanced loader
    if inc:
        return cleanDataLoader, included
    else:
        return cleanDataLoader

def GetFirstCorrectlyOverlappingSamplesBalancedSingle(device, sampleNum, numClasses, dataLoader, modelPlusList, bound = 1):
    
    totalSampleNum = len(dataLoader.dataset)
    accArrayCumulative = torch.zeros(totalSampleNum) #Create an array with one entry for ever sample in the dataset
    for j in range(1):
        accArray , acc= modelPlusList.validateDA(dataLoader)
        accArrayCumulative = accArrayCumulative + accArray
    accArrayCumulative /= 1
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    #Basic variable setup 
    samplePerClassCount = torch.zeros(numClasses) #keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses) #Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader) #Get all the data as tensors 
    #Memory for the solution 
    xClean = torch.zeros(sampleNum, 3, modelPlusList.imgSizeH, modelPlusList.imgSizeW)
    yClean = torch.zeros(sampleNum)
    #print(xClean.shape)
    sampleIndexer = 0
    #Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        #Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArrayCumulative[i] > .4 and samplePerClassCount[currentClass]<maxRequireSamplesPerClass:
            #xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            #print(xTest[i].shape)
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            #print(accArrayCumulative[i])
            sampleIndexer = sampleIndexer +1 #update the indexer 
            samplePerClassCount[currentClass] = samplePerClassCount[currentClass] + 1 #Update the number of samples for this class
    #Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    #Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            print("Didn't find enough of some class: ", samplePerClassCount[i], i)
            #raise ValueError("We didn't find enough of class: "+str(i))
    #Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean[:sampleIndexer], yClean[:sampleIndexer], transforms = None, batchSize = modelPlusList.batchSize, randomizer = None)

    return cleanDataLoader, acc
    

def get_n_correct_samples(model, dataloader, n, device, balance=True):
    cnt = 0
    correct_total = 0

    correct_samples = []
    correct_labels = []

    model.eval()

    dataloader.dataset.return_original = True

    outputs = []

    class_num = 0
    for idx, item in enumerate(dataloader):
        samples, labels, original = item
        max_class_id = labels.max()
        class_num = max(class_num, int(max_class_id))
        if idx > 5:
            break
    
    class_num = class_num + 1
    num_per_class = n / class_num

    if balance:
        assert(n % class_num == 0)

    counter = torch.zeros(class_num)

    with torch.no_grad():
        for item in dataloader:
            #print(len(item), item[0].shape)
            if counter.sum() >= n:
                break

            samples, labels, original = item

            input = samples.to(device)
            target = labels.long().to(device)

            out_spike,spike_count, filtered_output,filter_sum = model(input)

            # calculate acc
            _, idx = torch.max(spike_count, dim=1)

            cnt += len(labels)

            correct_idx = torch.where(idx == target)
            # correct_total += correct

            # correct_samples += list(original[correct_idx])
            # correct_labels += list(labels[correct_idx])
            outputs += list(spike_count.detach()[correct_idx])

            assert((target[correct_idx] != idx[correct_idx]).sum() == 0)

            for i in correct_idx[0]:
                if counter.sum() >= n:
                    break
                correct_label = labels[i]
                correct_sample = original[i]
                if counter[correct_label] >= num_per_class and balance == True:
                    continue
                correct_samples += [correct_sample]
                correct_labels += [correct_label]
                counter[labels[i]] += 1

    acc = counter.sum() / cnt

    return acc, correct_samples, correct_labels, outputs
# %%
if __name__ == "__main__":

    from torchvision import datasets
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    def plot_raster_dot(spike_mat):
        '''
        another function to plot spikes
        :param spike_mat: [row, length/time]
        :return:
        '''
        h,w = spike_mat.shape
        plt.figure()
        point_coordinate = np.where(spike_mat != 0)
        plt.scatter(point_coordinate[1], point_coordinate[0], s=1.5)
        plt.gca().invert_yaxis()
        plt.gca().set_xlim([0, w])
        plt.gca().set_ylim([0, h])

    def plot_rate_as_image(x):
        '''
        x: tensor [c,h,w,t]
        '''
        plt.figure()

        _,_,_,length = x.shape
        cnt = x.sum(-1) # sum along last dimension, shape [c,h,w]
        # cnt = cnt.abs()
        cnt = cnt.permute(1,2,0).detach().cpu().numpy() # shape [h,w,c], swich channel to last dimesion to plot
        
        rate = cnt / length
        if (-1 in x):
            rate = rate /2 + 0.5
        else:
            rate 
        plt.imshow(rate)


    cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms.ToTensor())

    test_dataloader = DataLoader(cifar10_testset, batch_size=64, shuffle=False, drop_last=True)

    T = 100

    class mysnn(torch.nn.Module):

        def __init__(self):

            super().__init__()

            self.encoder = if_encoder(T,0.9,1.0,'soft')
            self.encoder_2 = PoissonGenerator()

        def forward(self,x):
            """
            x is a batch of input image, shape [b,c,h,w]
            """

            # imaged encoded as spike trains, shape [batch, c, h, w, length]
            input_spike_train = self.encoder(x)

            # unbind along last dimension (the time dimension)
            input_spike_train_unbind = input_spike_train.unbind(dim=-1)

            outputs = []
            outputs_2 = []

            for t in range(T):
                
                #inp shape [b, c, h, w]
                inp = input_spike_train_unbind[t]

                inp_2 = self.encoder_2(x)

                outputs += [inp]
                outputs_2 += [inp_2]
            
            # return shape [b,c,h,w,length]
            return  torch.stack(outputs,dim=-1), torch.stack(outputs_2,dim=-1)

    snn = mysnn()

    one_batch = next(iter(test_dataloader))

    out,out_2 = snn(one_batch[0]) # shape

    # get one encoded image sample
    img = out[0] # shape [c,h,w,length]

    plot_rate_as_image(img)

    img_2 = out_2[0] # shape [c,h,w,length]

    plot_rate_as_image(img_2)

    print((img.sum(-1)/T).mean(), (img_2.sum(-1)/T).mean())

    plt.figure()

    img_2 = out_2[0] # shape [c,h,w,length]

    img_spike_count_2 = img_2.sum(-1) # sum along last dimension, shape [c,h,w]
    img_spike_count_2 = img_spike_count_2.permute(1,2,0).detach().cpu().numpy() # shape [h,w,c], swich channel to last dimesion to plot
    img_spike_rate_2 = img_spike_count_2 / T

    # plot spike rate to check if we can recover original image
    plt.imshow(img_spike_rate_2)

    print(img_spike_rate_2.mean(), img_spike_rate_2.mean())

    # plot raster
    # reshape to [c*h*w, T]
    img_reshape = img.reshape((-1,T)).detach().cpu().numpy()
    plot_raster_dot(img_reshape[1550:1560,:]) # there are 3072 spike trains, only plot part of it
    plt.figure()
    img_reshape_2 = img_2.reshape((-1,T)).detach().cpu().numpy()
    plot_raster_dot(img_reshape_2[1550:1560,:])


# %% test signed spike coding

    class mysnn_2(torch.nn.Module):

        def __init__(self):

            super().__init__()

            self.encoder_1 = if_encoder(T,0.9,1.0,'soft')
            self.encoder_2 = signed_if_encoder(T,0.9,1.0,'soft')
            self.encoder_3 = PoissonGenerator()
            

        def forward(self,x):
            """
            x is a batch of input image, shape [b,c,h,w]
            """

            # imaged encoded as spike trains, shape [batch, c, h, w, length]
            input_spike_train_1 = self.encoder_1(x)
            input_spike_train_2 = self.encoder_2(x)

            input_spike_train_3 = []

            for t in range(T):

                inp = self.encoder_3(x)

                input_spike_train_3 += [inp]
            
            # return shape [b,c,h,w,length]
            return  input_spike_train_1, input_spike_train_2, torch.stack(input_spike_train_3,dim=-1)
    
    normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)
    test_dataloader = DataLoader(cifar10_testset, batch_size=64, shuffle=True, drop_last=True)

    snn_2 = mysnn_2()
    one_batch = next(iter(test_dataloader))
    out_1,out_2,out_3 = snn_2(one_batch[0]) # shape

    idx = 0
    # original image
    plt.figure()
    plt.imshow(one_batch[0][idx].permute(1,2,0).cpu().numpy()/2+0.5)
    # recover image from spike rate
    plot_rate_as_image(out_1[idx])
    plot_rate_as_image(out_2[idx])
    plot_rate_as_image(out_3[idx])

    print((out_1[idx].sum(-1)/T).mean(),(out_2[idx].sum(-1)/T).mean(),(out_3[idx].sum(-1)/T).mean())

    plot_raster_dot(out_1[idx].reshape((-1,T)).detach().cpu().numpy()[1550:1560,:])
    plot_raster_dot(out_2[idx].reshape((-1,T)).detach().cpu().numpy()[1550:1560,:])
    plot_raster_dot(out_3[idx].reshape((-1,T)).detach().cpu().numpy()[1550:1560,:])



    
