import numpy as np
import skimage
import skimage.morphology
import PIL
from io import BytesIO
import scipy
import random
import warnings

class DefenseBarrageNetwork():
    def __init__(self, model, totalTransformNumber, classNum, colorChannelNum, n_cores = 10):
        #num_cpus = psutil.cpu_count(logical=False)
        #ray.init(num_cpus=num_cpus)
        self.model = model
        self.TotalTransformNumber = totalTransformNumber
        self.ClassNum = classNum
        self.ColorChannelNum = colorChannelNum
        self.NumCores = n_cores #Number of cores for parallel transformations
        #print("Warning: This defense is currently configured to use at least "+str(self.NumCores)+" CPU cores.")
        if self.ColorChannelNum != 3 and self.ColorChannelNum != 1: #Do some basic error checking
            raise ValueError("The color channel number must either be 1 (grayscale) or 3 (RGB).")

    #This does the transformations slightly differently than predict
    #If predict has TotalTransformNumber=5, then 5 transformations always applied
    #In Barrage paper a different amount (uniformly randomly selected) will be applied
    #Make sure input data is in range -0.5 to 0.5 for the image transformations to work, code will automatically correct it
    def GenerateTransformedDataForTraining(self, xData):
        totalSampleNum = xData.shape[0]
        xTransformed = np.copy(xData)+0.5  #Make a copy of the data so that the original does not get overwritten
        for i in range(0, totalSampleNum):
            print("Generating Data: " + str(i), end = "\r")
            currentTransformNumber = self.randUnifI(0, self.TotalTransformNumber) #pick a number between 0 and the total transform number
            if currentTransformNumber > 0 : #Only do transformations if the number is greater than 0, otherwise use original data
                if self.ColorChannelNum == 3: #Color dataset
                    xTransformed[i] = self.BarrageTransformColor(xTransformed[i], currentTransformNumber)
                else: #Grayscale
                    xTransformed[i] = self.BarrageTransformGrayscale(xTransformed[i], currentTransformNumber)
        xTransformed = xTransformed-0.5 #To do transformations had to mean shift +0.5 to get in 0-1 range, now time to go back to -0.5 to 0.5 range
        return xTransformed

    def SeriesTransform(self, xDataTransformed):
        sampleNumber = xDataTransformed.shape[0]
        for i in range(0, sampleNumber):
            if self.ColorChannelNum == 3: #RGB can use all the transformations
                xDataTransformed[i] = self.BarrageTransformColor(xDataTransformed[i], self.TotalTransformNumber)
            elif self.ColorChannelNum == 1: #Grayscale cannot use every transform
                xDataTransformed[i] = self.BarrageTransformGrayscale(xDataTransformed[i], self.TotalTransformNumber)

    def ParallelTransform(self, xDataTransformed):
        sampleNumber = xDataTransformed.shape[0]
        result = np.empty(xDataTransformed.shape)
        for i in range(0, sampleNumber):
            if self.ColorChannelNum == 3: #RGB can use all the transformations
                result[i] = self.BarrageTransformColor(xDataTransformed[i], self.TotalTransformNumber)
            elif self.ColorChannelNum == 1: #Grayscale cannot use every transform
                result[i] = self.BarrageTransformGrayscale(xDataTransformed[i], self.TotalTransformNumber)
        return result

    #This applies a number of group transforms for an RGB color dataset i.e. CIFAR-10
    #In the paper transformNumber = 5
    #If we get NAN value we reshuffle the transformations and try again
    def BarrageTransformColor(self, img, totalTransformNumber):
        with warnings.catch_warnings():
            originalImg = np.copy(img)
            groupIndex = [1,2,3,4,5,6,7,8,9,10]
            random.shuffle(groupIndex)
            nanFlag = False #Have not detected any NAN problems currently
            #Do the transformations in a random order
            for i in range(0, totalTransformNumber):
                #print("Transform:", groupIndex[i])
                img = self.SelectGroupTransform(img, groupIndex[i])
                nanFlag = self.ClipImage0to1RangeColor(img, groupIndex[i])
                if nanFlag == True: #We have encounted an NAN and must redo the transformations
                    imgSecondTry= self.BarrageTransformColor(originalImg, totalTransformNumber)
                    #imgplot = plt.imshow(imgSecondTry)
                    #plt.show()
                    return imgSecondTry
            return img

    #This applies a number of group transforms for an grayscale dataset i.e. Fashion-MNIST
    #In the paper transformNumber = 5 (for ImageNet)
    def BarrageTransformGrayscale(self, img, totalTransformNumber):
        originalImg = np.copy(img)
        groupIndex = [1,2,3,4,5,6,8,10]
        random.shuffle(groupIndex)
        nanFlag = False #Have not detected any NAN problems currently
        #Do the transformations in a random order
        for i in range(0, totalTransformNumber):
            img = self.SelectGroupTransform(img, groupIndex[i])
            nanFlag = self.ClipImage0to1RangeColor(img, groupIndex[i])
            if nanFlag == True: #We have encounted an NAN and must redo the transformations
                imgSecondTry= self.BarrageTransformGrayscale(originalImg, totalTransformNumber)
                return imgSecondTry
        return img

    #Selects the transform from among the 10 groups
    def SelectGroupTransform(self, img, index):
        if index==0 or index>10: #Do some basic error checking
            raise ValueError("Transformation index out of bounds.")
        #Do 1 of the 10 transformations
        if index == 1:
            img = self.Group1ColorPrecisionReduction(img)
        if index == 2:
            img = self.Group2JPEGNoise(img)
        if index == 3:
            img = self.Group3Swirl(img)
        if index == 4:
            img = self.Group4NoiseInjection(img)
        if index == 5:
            img = self.Group5FFTPerturbation(img)
        if index == 6:
            img = self.Group6Transformation(img)
        if index == 7:
            img = self.Group7Transformations(img)
        if index == 8:
            img = self.Group8Transformations(img)
        if index == 9:
            img = self.Group9Transformations(img)
        if index ==10:
            img = self.Group10Transformations(img)
        return img

    def randUnifI(self, low, high, params=None):
        p = np.random.uniform()
        if params is not None:
            params.append(p)
        return round((high-low)*p + low)

    def randUnifC(self, low, high, params=None):
        p = np.random.uniform()
        if params is not None:
            params.append(p)
        return (high-low)*p + low

    def randLogUniform(self, low, high, base=np.exp(1)):
        div = np.log(base)
        return base**np.random.uniform(np.log(low)/div, np.log(high)/div)

    #Transformation 14 causes image to sometimes go slightly above 1
    #Clip method here prevents that
    def ClipImage0to1RangeColor(self, img, groupIndex):
        imgRows = img.shape[0]
        imgCols = img.shape[1]
        colorChannelNum = img.shape[2]
        nanFlag = False #Flag for debugging
        for i in range(0, imgRows):
            for j in range(0, imgCols):
                for k in range(0, colorChannelNum):
                    if(img[i,j,k]>1.0):
                       img[i,j,k] = 1.0
                    if(img[i,j,k]<0.0):
                        img[i,j,k] = 0.0
                    if(np.isnan(img[i,j,k])==True):
                        #img[i,j,k] = 0.0
                        nanFlag = True
        if nanFlag == True:
            print("Warning NAN value detected. Reshuffling the transformations...")
        return nanFlag

    #Transformation 14 causes image to sometimes go slightly above 1
    #Clip method here prevents that
    def ClipImage0to1RangeGray(self, img, groupIndex):
        imgRows = img.shape[0]
        imgCols = img.shape[1]
        colorChannelNum = img.shape[2]
        nanFlag = False #Flag for debugging
        for i in range(0, imgRows):
            for j in range(0, imgCols):
                if(img[i,j,0]>1.0):
                    img[i,j,0] = 1.0
                if(img[i,j,k]<0.0):
                    img[i,j,0] = 0.0
                if(np.isnan(img[i,j,0])==True):
                    #img[i,j,k] = 0.0
                    nanFlag = True
        if nanFlag == True:
            print("Warning NAN value detected. Reshuffling the transformations...")
        return nanFlag

    #Group 1: Color Precision Reduction
    def Group1ColorPrecisionReduction(self, img):
        scales = [np.asscalar(np.random.random_integers(8,200)) for x in range(3)] #pick the max range from 8 to 200
        multi_channel = np.random.choice(2) == 0 #select if every color channel will have same range OR each color channel has it owns max range
        params = [multi_channel] + [s/200.0 for s in scales] #Creating the scaling factor
        if self.ColorChannelNum == 3: #Color image
            if multi_channel:
                img2 = np.round(img*scales[0])/scales[0] #If each channel doesn't have its own scale, just use the 0th scale
                #img = np.round(img*scales[0])/scales[0] #If each channel doesn't have its own scale, just use the 0th scale
            else:
                for i in range(3):
                    img2 = np.copy(img)
                    img2[:,:,i] = np.round(img[:,:,i]*scales[i]) / scales[i] #Each channel has its own scale
                    #img[:,:,i] = np.round(img[:,:,i]*scales[i]) / scales[i] #Each channel has its own scale
            return img2
        elif self.ColorChannelNum == 1: #Grayscale
            img2 = np.round(img[:,:,:]*scales[0])/scales[0]
            #img = np.round(img[:,:,:]*scales[0])/scales[0] #One channel, one scale
        return img2

    #Group 2: JPEG Noise
    #Images need to be in range [0,1]
    def Group2JPEGNoise(self, img):
        quality = np.asscalar(np.random.random_integers(55,95)) #Choose a random save quality
        params = [quality/100.0] #Divide quality by 100 to get a percent I guess
        if self.ColorChannelNum == 3:#RGB image
            pil_image = PIL.Image.fromarray((img*255.0).astype(np.uint8))
            f = BytesIO()
            pil_image.save(f, format='jpeg', quality=quality) #Save the image with a certain quality
            jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0 #Reload the image and normalize between 0 and 1
        elif self.ColorChannelNum == 1:#Grayscale image
            pil_image = PIL.Image.fromarray((img[:,:,0]*255.0).astype(np.uint8))
            f = BytesIO()
            pil_image.save(f, format='jpeg', quality=quality) #Save the image with a certain quality
            jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0 #Reload the image and normalize between 0 and 1
            jpeg_image = np.reshape(jpeg_image, (img.shape[0],img.shape[1],1)) #make the image a HxWx1 array, somehow PIL doesn't handle grayscale correctly
        return jpeg_image

    #Group 3: Swirl
    def Group3Swirl(self, img):
        imgRows = img.shape[0]
        imgCols = img.shape[1]
        strength = (3.0-0.01)*np.random.random(1)[0] + 0.01
        #c_x = np.random.random_integers(1, 256) #Original range
        #c_y = np.random.random_integers(1, 256)
        c_x = np.random.random_integers(1, imgRows)
        c_y = np.random.random_integers(1, imgCols)
        #radius = np.random.random_integers(10, 200)
        rMax = int(np.round(imgRows*(200.0/256.0)))
        rMin = int(np.round(imgRows*(20.0/256.0)))
        radius = np.random.random_integers(rMin, rMax)
        #params = [strength/2.0, c_x/256.0, c_y/256.0, radius/200.0]
        params = [strength/2.0, c_x/imgRows, c_y/imgCols, radius/rMax]
        img = skimage.transform.swirl(img, rotation=0, strength=strength, radius=radius, center=(c_x, c_y))
        return img

    #Group 4: Noise Injection
    def Group4NoiseInjection(self, img):
        params = []
        options = ['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle'] #The type of noise
        noise_type = np.random.choice(options, 1)[0] #Randomly select the type of noise
        params.append(options.index(noise_type)/6.0)
        per_channel = np.random.choice(2) == 0 #choose if each channel has a different noise OR the noise is the same for each image
        params.append(per_channel)
        if self.ColorChannelNum == 3: #Color image
            if per_channel: #Randomly choosen that each color channel has a different noise
                img2 = np.copy(img)
                for i in range(3):
                    img2[:,:,i] = skimage.util.random_noise(img[:,:,i], mode=noise_type)
                    #img[:,:,i] = skimage.util.random_noise(img[:,:,i], mode=noise_type)
                return img2
            else: #Each channel has the same type of noise
                img = skimage.util.random_noise(img, mode=noise_type)
        elif self.ColorChannelNum ==1:#Grayscale image
            img = skimage.util.random_noise(img, mode=noise_type)
        return img

    #Group 5: FFT Perturbation
    def Group5FFTPerturbation(self, img):
        r, c, _ = img.shape #Get the rows and columns
        point_factor = (1.02-0.98)*np.random.random((r,c)) + 0.98
        randomized_mask = [np.random.choice(2)==0 for x in range(3)] #Apply one mask for all channels or give each channel its own mask
        keep_fraction = [(0.95-0.0)*np.random.random(1)[0] + 0.0 for x in range(3)]
        params = randomized_mask + keep_fraction
        if self.ColorChannelNum == 3:
            img2 = np.copy(img)
            for i in range(3):
                im_fft = scipy.fft.fft2(img[:,:,i])
                # Set r and c to be the number of rows and columns of the array.
                r, c = im_fft.shape
                if randomized_mask[i]:
                    mask = np.ones(im_fft.shape[:2]) > 0
                    im_fft[int(r*keep_fraction[i]): int(r*(1-keep_fraction[i]))] = 0
                    im_fft[:, int(c*keep_fraction[i]): int(c*(1-keep_fraction[i]))] = 0
                    mask = ~mask
                    #Now things to keep = 0, things to remove = 1
                    mask = mask * ~(np.random.uniform(size=im_fft.shape[:2] ) < keep_fraction[i])
                    #Now switch back
                    mask = ~mask
                    im_fft = np.multiply(im_fft, mask)
                else:
                    im_fft[int(r*keep_fraction[i]): int(r*(1-keep_fraction[i]))] = 0
                    im_fft[:, int(c*keep_fraction[i]): int(c*(1-keep_fraction[i]))] = 0
                #Now, lets perturb all the rest of the non-zero values by a relative factor
                im_fft = np.multiply(im_fft, point_factor)
                im_new = scipy.fft.ifft2(im_fft).real

                #FFT inverse may no longer produce exact same range, so clip it back
                im_new = np.clip(im_new, 0, 1)
                img2[:,:,i] = im_new
                #img[:,:,i] = im_new
            return img2
        elif self.ColorChannelNum == 1:
            im_fft = scipy.fft.fft2(img[:,:,0])
            # Set r and c to be the number of rows and columns of the array.
            r, c = im_fft.shape
            if randomized_mask[0]:
                mask = np.ones(im_fft.shape[:2]) > 0
                im_fft[int(r*keep_fraction[0]): int(r*(1-keep_fraction[0]))] = 0
                im_fft[:, int(c*keep_fraction[0]): int(c*(1-keep_fraction[0]))] = 0
                mask = ~mask
                #Now things to keep = 0, things to remove = 1
                mask = mask * ~(np.random.uniform(size=im_fft.shape[:2] ) < keep_fraction[0])
                #Now switch back
                mask = ~mask
                im_fft = np.multiply(im_fft, mask)
            else:
                im_fft[int(r*keep_fraction[0]): int(r*(1-keep_fraction[0]))] = 0
                im_fft[:, int(c*keep_fraction[0]): int(c*(1-keep_fraction[0]))] = 0
            #Now, lets perturb all the rest of the non-zero values by a relative factor
            im_fft = np.multiply(im_fft, point_factor)
            im_new = scipy.fft.ifft2(im_fft).real
            #FFT inverse may no longer produce exact same range, so clip it back
            im_new = np.clip(im_new, 0, 1)
            img = np.reshape(im_new, (r, c, 1)) #Was doing operations on HxW, now need HxWx1
        return img

    #Select between the possible transformations in Group 7
    def Group6Transformation(self, img):
        img = self.Group6p1ZoomRandom(img)
        return img
        
    #Group 6 Zoom Group: Transformation 6.1 Random Zoom
    def Group6p1ZoomRandom(self, img):
        h, w, _ = img.shape
        #i_s = np.random.random_integers(10, 50)
        #i_e = np.random.random_integers(10, 50)
        #j_s = np.random.random_integers(10, 50)
        #j_e = np.random.random_integers(10, 50)
        minValue = int(np.round(h*(20.0/256.0)))
        maxValue = int(np.round(h*(50.0/256.0)))
        i_s = np.random.random_integers(minValue, maxValue)
        i_e = np.random.random_integers(minValue, maxValue)
        j_s = np.random.random_integers(minValue, maxValue)
        j_e = np.random.random_integers(minValue, maxValue)
        #params = [i_s/50, i_e/50, j_s/50, j_e/50]
        params = [i_s/maxValue, i_e/maxValue, j_s/maxValue, j_e/maxValue]
        i_e = h-i_e
        j_e = w-j_e
        #Crop the image...
        img = img[i_s:i_e,j_s:j_e,:]
        #...now scale it back up
        if self.ColorChannelNum == 3: #RGB
            img = skimage.transform.resize(img, (h, w, 3))
        elif self.ColorChannelNum == 1: #Grayscale
            img = skimage.transform.resize(img, (h, w, 1))
        return img

    #Group 6 Zoom Group: 6.2. Seam Carving Expansion
    def Group6p2SeamCarvingExpansion(self, img):
        h, w, _ = img.shape
        both_axis = np.random.choice(2) == 0
        #toRemove_1 = np.random.random_integers(10, 50)
        #toRemove_2 = np.random.random_integers(10, 50)
        minValue = int(np.round(h*(20.0/256.0)))
        maxValue = int(np.round(h*(50.0/256.0)))
        toRemove_1 = np.random.random_integers(minValue, maxValue)
        toRemove_2 = np.random.random_integers(minValue, maxValue)
        #params = [both_axis, toRemove_1/50, toRemove_2/50]
        params = [both_axis, toRemove_1/maxValue, toRemove_2/maxValue]

        cutOffValue = int(np.round(h*(30.0/256.0))) #Was 30 in the original code
        if both_axis:
            #First remove from vertical
            if self.ColorChannelNum == 3: #RGB
                eimg = skimage.filters.sobel(skimage.color.rgb2gray(img))
            elif self.ColorChannelNum == 1: #Gray
                eimg = skimage.filters.sobel(img[:,:,0])
            #Do some typecasting
            img = img.astype('double')
            eimg = eimg.astype('double')
            img = skimage.transform.seam_carve(img, eimg, 'vertical', toRemove_1)

            #Now from horizontal
            if self.ColorChannelNum == 3: #RGB
                eimg = skimage.filters.sobel(skimage.color.rgb2gray(img))
            elif self.ColorChannelNum == 1: #Gray
                eimg = skimage.filters.sobel(img)
            img = skimage.transform.seam_carve(img, eimg, 'horizontal', toRemove_2)
        else: #Only one axis
            if self.ColorChannelNum == 3: #RGB
                eimg = skimage.filters.sobel(skimage.color.rgb2gray(img))
            elif self.ColorChannelNum == 1: #Gray
                eimg = skimage.filters.sobel(img[:,:,0])
            direction = 'horizontal'
            #if toRemove_2 < 30:
            if toRemove_2 < cutOffValue:
                direction = 'vertical'
            #Do some typecasting
            img = img.astype('double')
            eimg = eimg.astype('double')
            img = skimage.transform.seam_carve(img, eimg, direction, toRemove_1)
        #Now scale it back up
        if self.ColorChannelNum == 3: #RGB
            img = skimage.transform.resize(img, (h, w, 3))
        elif self.ColorChannelNum == 1:
            img = skimage.transform.resize(img, (h, w, 1))
        return img

    #Select between the possible transformations in Group 7
    def Group7Transformations(self, img):
        transformIndex = int(self.randUnifI(0, 3)) #Choose between 0 and 3
        if transformIndex == 0:
            img = self.Group7p1AlterHSV(img)
            #print("8")
        elif transformIndex == 1:
            img = self.Group7p2AlterXYZ(img)
            #print("9")
        elif transformIndex == 2:
            img = self.Group7p3CIELAB(img)
            #print("10")
        else:
            img = self.Group7p4YUV(img)
            #print("11")
        return img

    #Group 7 Color Space Group: 7.1 Alter HSV
    def Group7p1AlterHSV(self, img):
        img = skimage.color.rgb2hsv(img)
        params = []
        #Hue
        img[:,:,0] += self.randUnifC(-0.05, 0.05, params=params)
        #Saturation
        img[:,:,1] += self.randUnifC(-0.25, 0.25, params=params)
        #Value
        img[:,:,2] += self.randUnifC(-0.25, 0.25, params=params)
        img = np.clip(img, 0, 1.0)
        img = skimage.color.hsv2rgb(img)
        img = np.clip(img, 0, 1.0)
        #return img, params
        return img

    #Group 7 Color Space Group: 7.2 CIE 1931 XYZ colorspace
    def Group7p2AlterXYZ(self, img):
        img = skimage.color.rgb2xyz(img)
        params = []
        #X
        img[:,:,0] += self.randUnifC(-0.05, 0.05, params=params)
        #Y
        img[:,:,1] += self.randUnifC(-0.05, 0.05, params=params)
        #Z
        img[:,:,2] += self.randUnifC(-0.05, 0.05, params=params)
        img = np.clip(img, 0, 1.0)
        img = skimage.color.xyz2rgb(img)
        img = np.clip(img, 0, 1.0)
        #return img, params
        return img

    #Group 7 Color Space Group: 7.3 CIELAB colorspace
    def Group7p3CIELAB(self, img):
        img = skimage.color.rgb2lab(img)
        params = []
        #L
        img[:,:,0] += self.randUnifC(-5.0, 5.0, params=params)
        #a
        img[:,:,1] += self.randUnifC(-2.0, 2.0, params=params)
        #b
        img[:,:,2] += self.randUnifC(-2.0, 2.0, params=params)
        # L 2 [0,100] so clip it; a & b channels can have,! negative values.
        img[:,:,0] = np.clip(img[:,:,0], 0, 100.0)
        img = skimage.color.lab2rgb(img)
        img = np.clip(img, 0, 1.0)
        return img

    #Group 7 Color Space Group: 7.4 YUV
    def Group7p4YUV(self, img):
        img = skimage.color.rgb2yuv(img)
        params = []
        #Y
        img[:,:,0] += self.randUnifC(-0.05, 0.05, params=params)
        #U
        img[:,:,1] += self.randUnifC(-0.02, 0.02, params=params)
        #V
        img[:,:,2] += self.randUnifC(-0.02, 0.02, params=params)
        # U & V channels can have negative values; clip only Y
        img[:,:,0] = np.clip(img[:,:,0], 0, 1.0)
        img = skimage.color.yuv2rgb(img)
        img = np.clip(img, 0, 1.0)
        return img

    #Group 8 transformations
    def Group8Transformations(self, img):
        transformIndex = int(self.randUnifI(0, 2)) #Choose between 0 and 2 (0,1,2)
        #transformIndex = 0 #DELLA debug
        if transformIndex == 0:
            img = self.Group8p1HistogramEqualization(img)
            #print("12")
        elif transformIndex == 1:
            img = self.Group8p2AdaptiveHistogramEqualization(img)
            #print("13")
        else:
            img = self.Group8p3ContrastStretching(img)
            #print("14")
        return img

    #Group 8 Contrast:8.1 Histogram Equalization
    def Group8p1HistogramEqualization(self, img):
        nbins = np.random.random_integers(40, 256)
        params = [ nbins/256.0 ]
        img2 = np.copy(img)
        if self.ColorChannelNum == 3:
            for i in range(3):
                img2[:,:,i] = skimage.exposure.equalize_hist(img[:,:,i], nbins=nbins)
                #img[:,:,i] = skimage.exposure.equalize_hist(img[:,:,i], nbins=nbins)
        elif self.ColorChannelNum == 1:
            img2[:,:,0] = skimage.exposure.equalize_hist(img[:,:,0], nbins=nbins)
            #img[:,:,0] = skimage.exposure.equalize_hist(img[:,:,0], nbins=nbins)
        return img2

    #Group 8 Contrast:8.2 Adaptive Histogram Equalization
    def Group8p2AdaptiveHistogramEqualization(self, img):
        min_size = min(img.shape[0], img.shape[1])/10
        max_size = min(img.shape[0], img.shape[1])/6
        per_channel = np.random.choice(2) == 0
        params = [ per_channel ]
        kernel_h = [ self.randUnifI(min_size, max_size, params=params) for x in range(3)]
        kernel_w = [ self.randUnifI(min_size, max_size, params=params) for x in range(3)]
        clip_lim = [ self.randUnifC(0.01, 0.04, params=params) for x in range(3)]
        if self.ColorChannelNum == 3: #RGB image
            if per_channel: #Different transformation for each channel
                img2 = np.copy(img)
                for i in range(3):
                    kern = (kernel_w[i], kernel_h[i])
                    #img[:,:,i] = skimage.exposure.equalize_adapthist(img[:,:,i], kernel_size=kern, clip_limit=clip_lim[i])
                    img2[:,:,i] = skimage.exposure.equalize_adapthist(img[:,:,i], kernel_size=kern, clip_limit=clip_lim[i])
                return img2
            else: #Same transformation for each channel
                kern = (kernel_w[0], kernel_h[0])
                img = skimage.exposure.equalize_adapthist(img, kernel_size=kern, clip_limit=clip_lim[0])
        elif self.ColorChannelNum == 1:
            imgRows = img.shape[0]
            imgCols = img.shape[1]
            kern = (kernel_w[0], kernel_h[0])
            img = skimage.exposure.equalize_adapthist(img[:,:,0], kernel_size=kern, clip_limit=clip_lim[0])
            img = np.reshape(img, (imgRows,imgCols,1)) #Add the last dummy dimension
        return img

    #Group 8 Contrast:8.3 Contrast Stretching
    def Group8p3ContrastStretching(self, img):
        per_channel = np.random.choice(2) == 0
        params = [ per_channel ]
        low_precentile = [ self.randUnifC(0.01, 0.04, params=params) for x in range(3)]
        hi_precentile = [ self.randUnifC(0.96, 0.99, params=params) for x in range(3)]
        if self.ColorChannelNum == 3: #RGB image
            if per_channel: #Apply different transformation to each channel
                img2 = np.copy(img)
                for i in range(3):
                    p2, p98 = np.percentile(img[:,:,i], (low_precentile[i]*100, hi_precentile[i]*100))
                    img2[:,:,i] = skimage.exposure.rescale_intensity(img[:,:,i], in_range=(p2, p98))
                    #img[:,:,i] = skimage.exposure.rescale_intensity(img[:,:,i], in_range=(p2, p98))
                return img2
            else: #Apply same transformation to each color channel of RGB image
                p2, p98 = np.percentile(img, (low_precentile[0] *100, hi_precentile[0]*100))
                img = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))
        elif self.ColorChannelNum == 1: #Grayscale image
            p2, p98 = np.percentile(img, (low_precentile[0] *100, hi_precentile[0]*100))
            img = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))
        return img

    #Group 9 transformations, note these actually use transformation 8.4, 8.5, 8.6 and 8.7 transformations
    #This is because the transformations were mislabeled in the Barrage appendix
    def Group9Transformations(self, img):
        transformIndex = int(self.randUnifI(0, 3)) #Choose between 0 and 3 (0,1,2,3)
        if transformIndex == 0:
            img = self.Group8p4GreyScaleMix(img)
            #print("15")
        elif transformIndex == 1:
            img = self.Group8p5GreyScalePartialMix(img)
            #print("16")
        elif transformIndex == 2:
            img = self.Group8p6TwoThirdsGreyScaleMix(img)
            #print("17")
        else:
            img = self.Group8p7OneChannelPartialGrey(img)
            #print("18")
        return img

    #Group 8 Contrast:8.4 Grey Scale Mix
    def Group8p4GreyScaleMix(self, img):
        # average of color channels, different contribution for each channel
        ratios = np.random.rand(3)
        ratios /= ratios.sum()
        params = [x for x in ratios]
        img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1] + img[:,:,2] * ratios[2]
        img2 = np.copy(img)
        for i in range(3):
            img2[:,:,i] = img_g
            #img[:,:,i] = img_g
        #return img
        return img2

    #Group 8 Contrast:8.5 Partial Grey Scale Partial Mix
    def Group8p5GreyScalePartialMix(self, img):
        ratios = np.random.rand(3)
        ratios/=ratios.sum()
        prop_ratios = np.random.rand(3)
        params = [x for x in ratios] + [x for x in prop_ratios]
        img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1] + img[:,:,2] * ratios[2]
        img2 = np.copy(img)
        for i in range(3):
            p = max(prop_ratios[i], 0.2)
            #img[:,:,i] = img[:,:,i]*p + img_g*(1.0-p)
            img2[:,:,i] = img[:,:,i]*p + img_g*(1.0-p)
        #return img
        return img2

    #Group 8 Contrast:8.6 2/3 Grey Scale Mix
    def Group8p6TwoThirdsGreyScaleMix(self, img):
        params = []
        # Pick a channel that will be left alone and remove it from the ones to be averaged
        channels = [0, 1, 2]
        remove_channel = np.random.choice(3)
        channels.remove( remove_channel)
        params.append( remove_channel )
        ratios = np.random.rand(2)
        ratios/=ratios.sum()
        params.append(ratios[0])
        #They sum to one, so first item fully specifies the group
        img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
        img2 = np.copy(img)
        for i in channels:
            img2[:,:,i] = img_g
            #img[:,:,i] = img_g
        return img2
        #return img

    #Group 8 Contrast: 8.7 One Channel Partial Grey
    def Group8p7OneChannelPartialGrey(self, img):
        params = []
        # Pick a channel that will be altered and remove it from the ones to be averaged
        channels = [0, 1, 2]
        to_alter = np.random.choice(3)
        channels.remove(to_alter)
        params.append(to_alter)
        ratios = np.random.rand(2)
        ratios/=ratios.sum()
        params.append(ratios[0]) #They sum to one, so first item fully specifies the group
        img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
        # Lets mix it back in with the original channel
        p = (0.9-0.1)*np.random.random(1)[0] + 0.1
        params.append(p)
        img2 = np.copy(img)
        img2[:,:,to_alter] = img_g*p + img[:,:,to_alter] *(1.0-p)
        #img[:,:,to_alter] = img_g*p + img[:,:,to_alter] *(1.0-p)
        return img2

    #Group 9 transformations, note these actually use transformation 9.1 through 9.7 transformations
    #This is because the transformations were mislabeled in the Barrage appendix
    def Group10Transformations(self, img):
        transformIndex = int(self.randUnifI(0, 6)) #Choose between 0 and 6 (0,1,2,3,4,5,6)
        #transformIndex = 5 #DELLA
        if transformIndex == 0:
            img = self.Group9p1GaussianBlur(img)
            #print("19")
        elif transformIndex == 1:
            img = self.Group9p2MedianFilter(img)
            #print("20")
        elif transformIndex == 2:
            img = self.Group9p3MeanFilter(img)
            #print("21")
        elif transformIndex == 3:
            img = self.Group9p4MeanBilateralFilter(img)
            #print("22")
        elif transformIndex == 4:
            img = self.Group9p5ChambolleDenoising(img)
            #print("23")
        elif transformIndex == 5:
            img = self.Group9p6WaveletDenoising(img)
            #print("24")
        else:
            img = self.Group9p7NonLocalMeansDenoising(img)
            #print("25")
        return img

    #Group 9: Denoising Group 9.1. Gaussian Blur
    def Group9p1GaussianBlur(self, img):
        img2 = np.copy(img)
        if self.ColorChannelNum == 3: #RGB image
            if self.randUnifC(0, 1) > 0.5:
                sigma = [self.randUnifC(0.1, 2)]*3 #Blur each channel with the same parameters
            else:
                sigma = [self.randUnifC(0.1, 2), self.randUnifC(0.1, 2), self.randUnifC(0.1, 2)] #Blur each channel with a different parameter
            img2[:,:,0] = skimage.filters.gaussian(img[:,:,0], sigma=sigma[0])
            img2[:,:,1] = skimage.filters.gaussian(img[:,:,1], sigma=sigma[1])
            img2[:,:,2] = skimage.filters.gaussian(img[:,:,2], sigma=sigma[2])
            #img[:,:,0] = skimage.filters.gaussian(img[:,:,0], sigma=sigma[0])
            #img[:,:,1] = skimage.filters.gaussian(img[:,:,1], sigma=sigma[1])
            #img[:,:,2] = skimage.filters.gaussian(img[:,:,2], sigma=sigma[2])
        if self.ColorChannelNum == 1: #Grayscale image
            sigma = [self.randUnifC(0.1, 2)]*3 #3 is extra here, don't actually need it
            #img[:,:,0] = skimage.filters.gaussian(img[:,:,0], sigma=sigma[0])
            img2[:,:,0] = skimage.filters.gaussian(img[:,:,0], sigma=sigma[0])
        return img2

    #Group 9: Denoising Group 9.2 Median Filter
    def Group9p2MedianFilter(self, img):
        if self.ColorChannelNum == 3: #RGB image
            if self.randUnifC(0, 1) > 0.5:
                radius = [self.randUnifI(2, 3)]*3
            else:
                radius = [self.randUnifI(2, 3), self.randUnifI(2, 3), self.randUnifI(2, 3)]
            # median blur - different sigma for each channel
            img2 = np.copy(img)
            for i in range(3):
                mask = skimage.morphology.disk(radius[i])
                img2[:,:,i] = skimage.filters.rank.median(img[:,:,i], mask) / 255.0
                #img[:,:,i] = skimage.filters.rank.median(img[:,:,i], mask) / 255.0
            return img2
        elif self.ColorChannelNum == 1: #Grayscale image
             radius = [self.randUnifI(2, 3), self.randUnifI(2, 3), self.randUnifI(2, 3)]
             img2 = np.copy(img) #For joblib
             mask = skimage.morphology.disk(radius[0])
             img2[:,:,0] = skimage.filters.rank.median(img[:,:,0], mask) / 255.0
             #img[:,:,0] = skimage.filters.rank.median(img[:,:,0], mask) / 255.0
        return img2

    #Group 9: Denoising Group 9.3 Mean Filter
    def Group9p3MeanFilter(self, img):
        if self.ColorChannelNum == 3: #RGB image
            if self.randUnifC(0, 1) > 0.5:
                radius = [self.randUnifI(2, 3)]*3
            else:
                radius = [self.randUnifI(2, 3), self.randUnifI(2, 3), self.randUnifI(2, 3)]
            # mean blur w/ different sigma for each channel
            img2 = np.copy(img)
            for i in range(3):

                mask = skimage.morphology.disk(radius[i])
                img2[:,:,i] = skimage.filters.rank.mean(img[:,:,i],mask)/255.0
                #img[:,:,i] = skimage.filters.rank.mean(img[:,:,i],mask)/255.0
            return img2
        elif self.ColorChannelNum == 1: #Grayscale image
            radius = [self.randUnifI(2, 3)]*3
            mask = skimage.morphology.disk(radius[0])
            img2 = np.copy(img) #Make a copy to have it work with joblib
            img2[:,:,0] =  skimage.filters.rank.mean(img[:,:,0],mask)/255.0
            #img[:,:,0] = skimage.filters.rank.mean(img[:,:,0],mask)/255.0
        return img2
        #return img

    #Group 9: Denoising Group 9.4 Mean Bilateral Filter
    def Group9p4MeanBilateralFilter(self, img):
        params = []
        radius = []
        ss = []
        if self.ColorChannelNum == 3: #RGB image
            img2 = np.copy(img)
            for i in range(3):
                radius.append(self.randUnifI(2, 30, params=params))
                ss.append(self.randUnifI(5, 30, params=params) )
                ss.append(self.randUnifI(5, 30, params=params) )
            for i in range(3):
                mask = skimage.morphology.disk(radius[i])
                #img[:,:,i] = skimage.filters.rank.mean_bilateral(img[:,:,i], mask, s0=ss[i], s1=ss[3+i])/255.0
                img2[:,:,i] = skimage.filters.rank.mean_bilateral(img[:,:,i], mask, s0=ss[i], s1=ss[3+i])/255.0
            return img2
        elif self.ColorChannelNum == 1: #Grayscale image
            img2 = np.copy(img) #do this for joblib
            radius.append(self.randUnifI(2, 20, params=params))
            ss.append(self.randUnifI(5, 20, params=params) )
            ss.append(self.randUnifI(5, 20, params=params) )
            mask = skimage.morphology.disk(radius[0])
            img2[:,:,0] = skimage.filters.rank.mean_bilateral(img[:,:,0], mask, s0=ss[0], s1=ss[1])/255.0
            #img[:,:,0] = skimage.filters.rank.mean_bilateral(img[:,:,0], mask, s0=ss[0], s1=ss[1])/255.0
        #return img
        return img2

    #Group 9: Denoising Group 9.5 Chambolle Denoising
    def Group9p5ChambolleDenoising(self, img):
        params = []
        weight = (0.3-0.05)*np.random.random(1)[0] + 0.05
        params.append( weight )
        multi_channel = np.random.choice(2) == 0
        params.append( multi_channel )
        img = skimage.restoration.denoise_tv_chambolle( img, weight=weight, multichannel=multi_channel)
        return img

    #Group 9: Denoising Group 9.6 Wavelet Denoising
    def Group9p6WaveletDenoising(self, img):
        convert2ycbcr = np.random.choice(2) == 0
        #convert2ycbcr = False #DEBUG DELLA
        #wavelet = np.random.choice(self.wavelets)
        wavelet = 'db1'
        #wavelets = np.arange(1,31) #Goes from 1 to 30
        #wavelets =[1,2,3,4,5]
        #wavelet = np.random.choice(wavelets)
        mode_ = np.random.choice(["soft", "hard"])
        #mode_ = "soft" #DEBUG DELLA
        if self.ColorChannelNum ==3: #RGB
            #denoise_kwargs = dict(multichannel=True, convert2ycbcr=convert2ycbcr, wavelet=wavelets, mode=mode_)
            denoise_kwargs = dict(multichannel=True, convert2ycbcr=convert2ycbcr, wavelet=wavelet, mode=mode_)
        elif self.ColorChannelNum ==1: #Grayscale
            mode_ = "hard" #Soft does not seem to work with Fasion-MNIST
            #denoise_kwargs = dict(multichannel=False, convert2ycbcr=convert2ycbcr, wavelet=wavelets, mode=mode_)
            denoise_kwargs = dict(multichannel=False, convert2ycbcr=convert2ycbcr, wavelet=wavelet, mode=mode_)
        max_shifts = np.random.choice([0, 1])
        if self.ColorChannelNum ==3: #RGB
            img = skimage.restoration.cycle_spin(img, func=skimage.restoration.denoise_wavelet, max_shifts=max_shifts, func_kw=denoise_kwargs, multichannel=True, num_workers=1)
        elif self.ColorChannelNum ==1: #Grayscale
            img = skimage.restoration.cycle_spin(img, func=skimage.restoration.denoise_wavelet, max_shifts=max_shifts, func_kw=denoise_kwargs, multichannel=False, num_workers=1)
        return img

    #Group 9: Denoising Group 9.7 Non-Local Means Denoising
    def Group9p7NonLocalMeansDenoising(self, img):
        h_1 = self.randUnifC(0, 1)
        params = [h_1]
        sigma_est = np.mean(skimage.restoration.estimate_sigma(img,multichannel=True))
        h = (1.15-0.6)*sigma_est*h_1 + 0.6*sigma_est
        #If false, it assumes some weird 3D stuff
        multi_channel = np.random.choice(2) == 0
        params.append( multi_channel )
        #Takes too long to run without fast mode.
        fast_mode = True
        patch_size = np.random.random_integers(5, 7)
        params.append(patch_size)
        patch_distance = np.random.random_integers(6, 11)
        params.append(patch_distance)
        if self.ColorChannelNum == 3: #RGB
            if multi_channel:
                img2 = skimage.restoration.denoise_nl_means( img, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode)
                #img = skimage.restoration.denoise_nl_means( img, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode)
            else:
                img2 = np.copy(img)
                for i in range(3):
                    sigma_est = np.mean(skimage.restoration.estimate_sigma(img[:,:,i], multichannel=True ) )
                    h = (1.15-0.6)*sigma_est*params[i] + 0.6*sigma_est
                    #img[:,:,i] = skimage.restoration.denoise_nl_means(img[:,:,i], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode )
                    img2[:,:,i] = skimage.restoration.denoise_nl_means(img[:,:,i], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode )
                return img2
        elif self.ColorChannelNum == 1: #Grayscale
            img2 = np.copy(img)
            #In grayscale we assume multi-channel equals false or else we get very poor results
            sigma_est = np.mean(skimage.restoration.estimate_sigma(img[:,:,0], multichannel=False ) )
            h = (1.15-0.6)*sigma_est*params[0] + 0.6*sigma_est
            img2[:,:,0] = skimage.restoration.denoise_nl_means(img[:,:,0], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode )
            #img[:,:,0] = skimage.restoration.denoise_nl_means(img[:,:,0], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode )
        return img2

    #the network is fooled if we don't have a noise class label AND it gets the wrong label
    #Returns attack success rate
    def evaluateAdversarialAttackSuccessRate(self, xAdv, yClean):
        sampleNum=xAdv.shape[0]
        yPred=self.predict(xAdv)
        advAcc=0
        for i in range(0, sampleNum):
            #The attack wins only if we don't correctly label the sample AND the sample isn't given the nosie class label
            if yPred[i].argmax(axis=0) != self.ClassNum and yPred[i].argmax(axis=0) != yClean[i].argmax(axis=0): #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/sampleNum
        return advAcc
