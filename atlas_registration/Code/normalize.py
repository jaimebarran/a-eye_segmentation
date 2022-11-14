import sys,os
import SimpleITK as sitk
import numpy as np
from medpy.io import load,save,get_pixel_spacing
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import interp1d
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage.filters import generic_gradient_magnitude, gaussian_gradient_magnitude, prewitt, sobel
from medpy.io import load, save, get_pixel_spacing
from pathlib import Path

def doGradient(fileName,mode):
    
    data_input, header_input = load(fileName)
  
    data_output = scipy.zeros(data_input.shape, dtype=scipy.float32)

    reader = sitk.ImageFileReader()
    
    if (mode ==1):
        gaussian_gradient_magnitude(data_input, 0.2, output=data_output)
        
        fileName = fileName+ "gradient.nii"
 
    if (mode ==2):
        generic_gradient_magnitude(data_input, prewitt, output=data_output)
        fileName = fileName+ "/prewitt.nii"

    if (mode ==3):
        generic_gradient_magnitude(data_input, sobel, output=data_output)
        fileName = fileName+ "/sobel.nii"
         
    save(data_output, fileName , header_input)       
    
    reader.SetFileName ( fileName)
    image = reader.Execute()
    
    return image

def doAnisoAndBiascorrection(binaryName,i):
            
            print("do anisotropic diffusion") 
            data_input, header_input = load(binaryName)
            
            data_output = anisotropic_diffusion(data_input, 10, 50, 0.1, get_pixel_spacing(header_input))


            print("do bias field correction")
            inputImage = sitk.ReadImage( binaryName )
            
            maskImage = sitk.OtsuThreshold( inputImage, 0, 1, 200 )
            
            inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )
            
            corrector = sitk.N4BiasFieldCorrectionImageFilter();

            numberFilltingLevels = 4

            output = corrector.Execute( inputImage, maskImage )    

            biasName =  "./Output/Normalization_Soenke/BiasField/biasfield"+str(i)+".nii"
            sitk.WriteImage( output,  biasName)
            
            return biasName
    
    
'''do image normalization includes 3 steps:
     S1: anisotropic
     S2: bias field correction
     S3: Nyul histogram equalization'''
def normalize(location, Modality):
    
        listFiles = []

        #do Anisotropic and bias correction for each image
        i = 0
        for dirpath, dirnames, filenames in os.walk(location):
            for filename in [f for f in filenames if ((Modality in f)and('.nii'in f)) ]:
                imageName= os.path.join(dirpath,filename)
                print(imageName)
                biasName=doAnisoAndBiascorrection(imageName,i)
                listFiles.append(biasName)
                i= i+1      
        print("do Nyul model")
        outputModel = './Output/Normalization_Soenke/nyulModel/nyulModel'+Modality

        nyul = NyulNormalizer()
        
        nyul.train(listOfImages= listFiles)
        nyul.saveTrainedModel(outputModel)

        for dirpath, dirnames, filenames in os.walk(location):
            for filename in [f for f in filenames if (('.nii' in f)and(Modality in f))]:
                imageName= os.path.join(dirpath,filename)
                p = Path(imageName)
                image = sitk.ReadImage(imageName)
                transformedImage = nyul.transform(image)
                
                directory = './Output/Normalization_Soenke/Normalization/'+p.parts[2]+'/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                outputfilename = os.path.join(directory,filename)
                
                print(outputfilename)
                sitk.WriteImage( transformedImage, outputfilename )

                
                   
# ********************************************************
# ************* Auxiliar functions ***********************
# ********************************************************

def getCdf(hist):
    """ 
        Given a histogram, it returns the cumulative distribution function.
    """
    aux = np.cumsum(hist)
    aux = aux / aux[-1] * 100
    return aux
    
def getPercentile(cdf, bins, perc):
    """ 
        Given a cumulative distribution function obtained from a histogram, 
        (where 'bins' are the x values of the histogram and 'cdf' is the 
        cumulative distribution function of the original histogram), it returns
        the x center value for the bin index corresponding to the given percentile,
        and the bin index itself.
        
        Example:
        
            import numpy as np
            hist = np.array([204., 1651., 2405., 1972., 872., 1455.])
            bins = np.array([0., 1., 2., 3., 4., 5., 6.])
            
            cumHist = getCdf(hist)
            print cumHist
            val, bin = getPercentile(cumHist, bins, 50)
            
            print "Val = " + str(val)
            print "Bin = " + str(bin)
        
    """
    b = len(bins[cdf <= perc])
    return bins[b] + ( (bins[1] - bins[0]) / 2), b

def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)



# ********************************************************
# ************* NyulNormalizer class *********************
# ********************************************************

class NyulNormalizer:
    """ 
    It implements the training and transform methods for Nyul's normalization.

    Attributes:
        trainingImages images used to learn the standard intensity landmarks 
    """
    nbins = 1024

    def __init__(self):
        self.meanLandmarks = None

    def __getLandmarks(self, image, mask = None, showLandmarks=False):
        """
            This Private function obtain the landmarks for a given image and returns them
            in a list like:
                [lm_pLow, lm_perc1, lm_perc2, ... lm_perc_(numPoints-1), lm_pHigh] (lm means landmark)

            :param image    SimpleITK image for which the landmarks are computed.
            :param mask     [OPTIONAL] SimpleITK image containing a mask. If provided, the histogram will be computed
                                    taking into account only the voxels where mask > 0.
            :param showLandmarks    Plot the landmarks using matplotlib on top of the histogram.

        """
            
        data = sitk.GetArrayFromImage(image)

        if mask is None:
            # Calculate useful statistics
            stats = sitk.StatisticsImageFilter()
            stats.Execute( image )
            mean = stats.GetMean()

            # Compute the image histogram
            histo, bins = np.histogram(data.flatten(), self.nbins, normed=True)

            # Calculate the cumulative distribution function of the original histogram
            cdfOriginal = getCdf(histo)

            # Truncate the histogram (put 0 to those values whose intensity is less than the mean)
            # so that only the foreground values are considered for the landmark learning process
            histo[bins[:-1] < mean] = 0.0
        else:
            # Calculate useful statistics
            dataMask = sitk.GetArrayFromImage(mask)

            # Compute the image histogram
            histo, bins = np.histogram(data[dataMask > 0].flatten(), self.nbins, normed=True)

            # Calculate the cumulative distribution function of the original histogram
            cdfOriginal = getCdf(histo)

        # Calculate the cumulative distribution function of the truncated histogram, where outliers are removed
        cdfTruncated = getCdf(histo)

        # Generate the percentile landmarks for  m_i          
        perc = [x for x in range(0, 100, 100//self.numPoints)]
        # Remove the first landmark that will always correspond to 0
        perc = perc[1:]        

        # Generate the landmarks. Note that those corresponding to pLow and pHigh (at the beginning and the
        # end of the list of landmarks) are generated from the cdfOriginal, while the ones
        # corresponding to the percentiles are generated from cdfTruncated, meaning that only foreground intensities
        # are considered.
        bins = bins[1:len(bins)]
        landmarks = [getPercentile(cdfOriginal, bins, self.pLow)[0]]  + [getPercentile(cdfTruncated, bins, x)[0] for x in perc] + [getPercentile(cdfOriginal, bins, self.pHigh)[0]]

        if showLandmarks:            
            yCoord = max(histo)
            
            # plt.figure(dpi=100)
            # plt.plot(bins[:-1],histo)
            # plt.plot([landmarks[0], landmarks[-1]], [yCoord, yCoord] ,'r^')    
            # plt.plot(landmarks[1:-1], np.ones(len(landmarks)-2) * yCoord,'g^')    
            # plt.show()
                
        return landmarks
    
    def __landmarksSanityCheck(self, landmarks):
        if not (np.unique(landmarks).size == len(landmarks)):
            for i in range(1, len(landmarks)-1):
                if landmarks[i] == landmarks[i+1]:
                    landmarks[i] = (landmarks[i-1] + landmarks[i+1]) / 2.0

                print ("WARNING: Fixing duplicate landmark.")

            if not (np.unique(landmarks).size == len(landmarks)):
                raise Exception('ERROR NyulNormalizer landmarks sanity check : One of the landmarks is duplicate. You can try increasing the number of bins in the histogram \
                (NyulNormalizer.nbins) to avoid this behaviour. Landmarks are: ' + str(landmarks))

        elif not (sorted(landmarks) == list(landmarks)):
            raise Exception('ERROR NyulNormalizer landmarks sanity check: Landmarks in the list are not sorted, while they should be. Landmarks are: ' + str(landmarks))
        
    def train(self, listOfImages, listOfMasks = [], pLow = 1, pHigh = 99, sMin = 1, sMax = 100, numPoints = 10, showLandmarks=False):
        """
            Train a new model for the given list of images (listOfImages is a list of strings, where every element is the full path to an image)

            Optionally, a list containing full paths to the corresponding masks (listOfMasks) can be provided. If masks are provided, the image will

            Note that the actual number of points is numPoints that will be generated (including
            the landmarks corresponding to pLow and pHigh) is numPoints + 1.
            
            Recommended values for numPoints are 10 and 4.
            
            Example 1: if pLow = 1, pHigh = 99, numPoints = 10, the landmarks will be:
            
                [lm_p1, lm_p10, lm_p20, lm_p30, lm_p40, lm_p50, lm_p60, lm_p70, lm_p90, lm_p99 ]
            
            Example 2: if pLow = 1, pHigh = 99, numPoints = 4, the landmarks will be:
            
                [lm_p1, lm_p25, lm_p50, lm_p75, lm_p99]

            :param listOfImages Images used to learn the distribution of landmarks. It's a list of strings, where every element is the full path to an image.
            :param listOfMasks  [OPTIONAL] Masks used to compute the histogram of the corresponding images. If masks are provided, the histograms will be computed
                                taking into account only those voxels where mask > 1. len(listOfMasks) == len(listOfImages). It is list of strings, where every element
                                is the full path to an image.
            :param pLow, pHigh  Percentiles corresponding to the positions where the first and last landmarks are going to be mapped.
            :param sMin, sMax   Minimum and maximum value of the target intensity range. The value corresponding to the percentile pLow will be mapped to sMin, while
                                the value corresponding to pHigh will be mapped to sMax.
            :param numPoints    Number of landmarks that will be learned
            :param showLandmarks Show the landmarks for every image, overlapped in the corresponding histogram, using matplotlib.

        """
        # Percentiles used to trunk the tails of the histogram
        if pLow > 10:
            raise("NyulNormalizer Error: pLow may be bigger than the first lm_pXX landmark.")
        if pHigh < 90:
            raise("NyulNormalizer Error: pHigh may be bigger than the first lm_pXX landmark.")
                    
        self.pLow = pLow
        self.pHigh = pHigh        
        self.numPoints = numPoints
        self.sMin = sMin
        self.sMax = sMax
        self.meanLandmarks = None
        
        allMappedLandmarks = []
        # For each image in the training set
        for fNameIndex in range(len(listOfImages)):
            fName = listOfImages[fNameIndex]
      
            # Read the image
            image = sitk.ReadImage(fName)

            if listOfMasks == []:
                # Generate the landmarks for the current image
                landmarks = self.__getLandmarks(image, showLandmarks=showLandmarks)
            else:
                mask = sitk.ReadImage(listOfMasks[fNameIndex])
                landmarks = self.__getLandmarks(image, mask, showLandmarks=showLandmarks)

            # Check the obtained landmarks ...
            self.__landmarksSanityCheck(landmarks)
            
            # Construct the linear mapping function
            mapping = interp1d([landmarks[0],landmarks[-1]],[sMin,sMax], fill_value="extrapolate")
            
            # Map the landmarks to the standard scale
            mappedLandmarks = mapping(landmarks)
            
            # Add the mapped landmarks to the working set
            allMappedLandmarks.append(mappedLandmarks)
                
       
        
        self.meanLandmarks = np.array(allMappedLandmarks).mean(axis=0)
        
        # Check the obtained landmarks ...
        self.__landmarksSanityCheck(self.meanLandmarks)

        
        
    def saveTrainedModel(self, location):
        """
            Saves the trained model in the specified file location (it adds '.npz' to the filename, so 
            do not specify the extension). To load it, you can use:
                
                nyulNormalizer.loadTrainedModel(outputModel)

            :param location Absolute path corresponding to the output file. It adds '.npz' to the filename, so
                            do not specify the extension (e.g. /path/to/file)
        """
        trainedModel = {
            'pLow' : self.pLow,
            'pHigh' : self.pHigh,
            'sMin' : self.sMin,
            'sMax' : self.sMax,
            'numPoints' : self.numPoints,
            'meanLandmarks' : self.meanLandmarks
        }
            
        np.savez(location, trainedModel = [trainedModel])
        
        
    def loadTrainedModel(self, savedModel):
        """
            Loads a trained model previously saved using: 
                
                nyulNormalizer.saveTrainedModel(outputModel)

            :param savedModel Absolute path (including extension) to the "npz" file with the corresponding learned landmarks.
        """
        f = np.load(savedModel)
        tModel = f['trainedModel'].all()
        
        self.pLow = tModel['pLow']
        self.pHigh = tModel['pHigh']
        self.numPoints = tModel['numPoints']
        self.sMin = tModel['sMin']
        self.sMax = tModel['sMax']
        self.meanLandmarks = tModel['meanLandmarks']       

        # Check the obtained landmarks ...
        self.__landmarksSanityCheck(self.meanLandmarks)

                
    def transform(self, image, mask=None):
        """
            It transforms the image to the learned standard scale
            and returns it as a SimpleITK image.

            If mask is provided, the histogram of the image to be normalized will be only calculated based on the pixels where mask>0.
            All pixels (independently of their mask value) will be transformed.

            The intensities between [minIntensity, intensity(pLow)) are linearly mapped using
            the same function that the first interval.
            
            The intensities between [intensity(pHigh), maxIntensity) are linearly mapped using
            the same function that the last interval.

            :param image    SimpleITK image that will be transformed.
            :param mask     [OPTIONAL] If provided, only voxels where mask > 0 will be considered to compute the histogram.
            :return Transoformed image
        """
        
        # Get the raw data of the image
        data = sitk.GetArrayFromImage( image )
        
        # Calculate useful statistics            
        stats = sitk.StatisticsImageFilter()
        stats.Execute( image )
        
        # Obtain the minimum
        origMin = stats.GetMinimum()
        origMax = stats.GetMaximum()
        origMean = stats.GetMean()
        origVariance= stats.GetVariance()
        

        
        # Get the landmarks for the current image
        landmarks = self.__getLandmarks(image, mask)
        landmarks = np.array(landmarks)

        # Check the obtained landmarks ...
        self.__landmarksSanityCheck(landmarks)
        
        # Recover the standard scale landmarks
        standardScale = self.meanLandmarks
        
   
        
        # Construct the piecewise linear interpolator to map the landmarks to the standard scale
        mapping = interp1d(landmarks, standardScale, fill_value="extrapolate")

        # Map the input image to the standard space using the piecewise linear function
        
        flatData = data.ravel()        
 
        mappedData = mapping(flatData)

        mappedData = mappedData.reshape(data.shape)
    
        # Save edited data    
        output = sitk.GetImageFromArray( mappedData )
        output.SetSpacing( image.GetSpacing() )
        output.SetOrigin( image.GetOrigin() )
        output.SetDirection( image.GetDirection() )
        
        # Calculate useful statistics            
        stats = sitk.StatisticsImageFilter()
        stats.Execute( output )
        


        # Obtain the minimum
        origMin = stats.GetMinimum()
        origMax = stats.GetMaximum()
        origMean = stats.GetMean()
        origVariance= stats.GetVariance()
        
             
        
        return output


if __name__ == "__main__":
    # To run: python 1-normalization/normalize.py Data/P_Soenke/ T1
    # sys.argv[1]: data location where .nii files
    # location = './Data/PSI_Nii/'
    # location = './Data/P_Soenke/'
    # sys.argv[2]: data sequence
    # sequence = 'T1'

    #here create output folder
    # directory = './Output'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
        
    directory = './Output/Normalization_Soenke'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = './Output/Normalization_Soenke/BiasField'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = './Output/Normalization_Soenke/nyulModel'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = './Output/Normalization_Soenke/Normalization'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    #normalisation
    normalize(sys.argv[1],sys.argv[2])
    # normalize(location, sequence)
    
