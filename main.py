# This is a sample Python script.

# Simple script and function to read RAW (.nef) file, convert to 3-channel RGB file, and view each as grayscale.

# Original Nikon RAW (.nef) images can be downloaded from Google Drive folder:
# https://drive.google.com/drive/folders/0Bwvtq_Wfkb6yR1h1NmU4bUJwck0?usp=sharing

# The following code sample assumes that at least one NEF (or nef) image file is in a folder named 'images'
# in the current working directory.

import os
import shutil
import rawpy
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math
from scipy.stats import *
import scipy.stats
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

#%%
# Local function definitions:

# Get a list of image files ending with a specified extension in a named directory
def getImgFnames(imgDir, extension='nef'):

    imgList = []

    for fname in os.listdir(imgDir):
        if fname.endswith(extension.lower()) or fname.endswith(extension.upper()):
            imgList.append(os.path.join(imgDir, fname))
        else:
            continue

    return imgList

#%%
# Read a specified raw image and return the rawImg object
def readRawImage(absImgFname, verbose=False):
    rawImg = rawpy.imread(absImgFname)  # Read raw image class

    if verbose:
        print(f'np.unique(rawImg.raw_colors) = {np.unique(rawImg.raw_colors)}   ', end='')
        print(f'[Number of colors = {len(np.unique(rawImg.raw_colors))}]')

    return rawImg

#%%
# Extract one of the four image arrays from the raw (Bayer) image
def extractOneBayerChannel(rawArr, rawColors, channelNumber, verbose=False):
    # Function accepts as input rawImage representing a Bayer-CFA
    # and the channel number (0-3) and returns the appropriate (1/4) subimage
    if verbose:
        print(f'===================   extractOneBayerChannel  ========================')
        print(f'type(rawArr) = {type(rawArr)}   ', end='')
        print(f'   [Each element: type(rawArr[0,0]) = {type(rawArr[0,0])}]')
        print(f'channelNumber = {channelNumber}')

    if verbose:
        print(f'rawArr.shape = {rawArr.shape}\n')

    # Extract ONE of the subimages:
    # Create a boolean mask which is True for the target channel, False elsewhere
    assert(channelNumber in np.unique(rawColors)), "Channel not in list ..."

    chanMask = (rawColors == channelNumber)  # rawImg.raw_colors==colorChan

    # Determine nrows & ncols of subimage (1/4 size for Bayer array pattern)
    nrows = int(rawArr.shape[0] / 2)
    ncols = int(rawArr.shape[1] / 2)


    oneRawChanArr = rawArr[chanMask]  # extract the subpixels at the mask==True locations
    oneRawChanArr = oneRawChanArr.reshape(nrows, ncols)  # reshape 1D -> 2D

    if verbose:
        print(
            f'\n - - - - - - \n channelNumber = {channelNumber}:   chanMask = \n{chanMask}   chanMask.shape = {chanMask.shape}')
        print(f'\n - - - - - - \n rawArr/2: nrows, ncols = {nrows}, {ncols}')
        print(f'\n - - - - - - \n \nrawChanArr = \n{oneRawChanArr}   \nrawChanArr.shape = {oneRawChanArr.shape}')
        print(f'\n - - - - - - \n Reshaping rawChanArr from {oneRawChanArr.shape} to {nrows} x {ncols}')
        print(f'\n - - - - - - \n Completed calculating subArray # {channelNumber}: \n - - - - - - \n ')
        print(f'oneRawChanArr = \n{oneRawChanArr} ')
        print(f'===================  END   extractOneBayerChannel   END ========================')

    return oneRawChanArr

#%%
def imgShow(img, r=0, c=0, h=0, w=0, title='', cmap='gray', colorbar=True):
    plt.figure()
    if h==0 or w==0:  # optional height, width. If not supplied, use shape of image
        h,w = img.shape[0]-1, img.shape[1]-1

    plt.imshow(img[r:r+h,c:c+w], cmap='gray')

    if title is not '':
        plt.title(title)

    if colorbar:
        plt.colorbar()


    plt.savefig('gamma loops/' + title + '_image.png')
    plt.close()
    plt.show()


#%%
def standarise(array_values, pct, pct_lower):
    # sets up sc shortcut which subtracts mean and scales to units of variance (zscore)
    sc = StandardScaler()
    #sorts array values from smallest to largest
    array_values.sort()
    #len_av is the length of the array_values array
    len_av = len(array_values)
    #trimms array_values according to pct, pct_lower, and stores in array_vals variable
    array_vals = array_values[int(pct_lower * len_av):(int(len_av * pct) - 1)]
    #length of array after trim
    len_av = len(array_vals)
    #yy is a list of all the values in array_vals
    yy = ([[x] for x in array_vals])
    # computes the mean and std of yy - needed for sc.transform
    sc.fit(yy)
    #sc.transform: Performs standardization by centering and scaling
    # ( value - mean )/ standard deviation
    y_standard = sc.transform(yy)
    # returns the standardized (trimmed) data, the length of that data, the trimmed data
    return y_standard, len_av, array_vals

#%%
def fit_distribution(array_values, pct, pct_lower, row, column):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    #trims array_values, returns:
    #y_standard - list of the standardise values (zscores)
    #size - length of y - organized
    #y_organized - array of the trimmed array_values, sorted from least to greatest
    y_standard, size, y_organized = standarise(array_values, pct, pct_lower)
    dist_names = ['weibull_min', 'norm', 'weibull_max', 'beta',
                   'invgauss', 'uniform', 'gamma', 'expon', 'lognorm', 'pearson3', 'triang']
    chi_square_statistics = []  # to fill as the loop progresses

    #returns the intervals for 11 evenly spaced bins from 0 to 100
    #percentile_bins = np.linspace(0, 100, 11)
    #use bins instead of percentile_bins
    #returns the values in y_standard at the percentiles given by percentile_bins
    #percentile_cutoffs = np.percentile(y_standard, percentile_bins)
    #use bins instead of percentile_cutoffs
    #linear_bins = np.linspace(0, 4095, 11)
    #linear_cutoffs =

    #observed_frequency, bins = (np.histogram(y_standard, bins=percentile_cutoffs))
    try:
        #observed freq is the values of each of the bins of the histogram
        #bins is the bins edges of 10 evenly spaced bins in the given range of data
        observed_frequency, bins = (np.histogram(y_standard, bins=10))
    except ValueError as errMsg:
        observed_frequency, bins = None, None
        print(f'we got: {errMsg}')
        return 1

    print(f'bins = {bins}')

    #returns the cumulative sum of the observed frequency
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # retrieves attributes of the specified distribution from the scipy.stats package
        dist = getattr(scipy.stats, distribution)
        #returns maximum likelihood estimate for shape, location, and scale parameters from y_standard
        param = dist.fit(y_standard)

        # Get expected counts in percentile bins
        # cdf of fitted distribution across bins

        #returns cumulative distribution function evaluated at percentile_cutoffs <at 0, 10, 20 ect percentiles>
        cdf_fitted = dist.cdf(bins, *param)
        #cdf_fitted = dist.cdf(bins, *param)
        expected_frequency = []
        for bin in range(len(bins) - 1):  #bins replaced percentile_bins
            #taking the difference between cdf_fitted adjacent values
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            #appends values to array expected_frequency
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics

        #scales expected_frequency up by the size of y_organized (trimmed array)
        expected_frequency = np.array(expected_frequency) * size

        #expected_frequency can't be 0 because of standard error calculations, so if it is set it = 1
        for index, expFreq in enumerate(expected_frequency):
            if (expFreq == 0):
                expected_frequency[index] = 1

        # returns the cumulative sum of the expected frequency
        cum_expected_frequency = np.cumsum(expected_frequency)
        #standard squared error ->> (observed - expected)^2/expected
        ss = round(sum(((cum_observed_frequency - cum_expected_frequency) ** 2) / cum_expected_frequency), 0)
        chi_square_statistics.append(ss)

    # Sort by minimum chi-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)

    #print('\nDistributions listed by Betterment of fit:')
    #print('............................................')
    #print(results)
    #print(results.iat[0, 0])
    histogram(results.iat[0, 0], row, column, y_organized, pct, pct_lower)
    return results.iat[0, 0]

#%%
def histogram(dist_name, row, column, data, pct, pct_lower):
    plt.figure()
    plt.hist(data, bins=10, density=True, color='red')
    plt.title(f'{dist_name}, row {row}, column {column}')
    strFile = (f'histogram/trim_' + str(pct_lower) + '_' + str(pct) + '/' + str(row) + '_' + str(column) +
               '_' + str(dist_name) + '.png')
    plt.savefig(strFile)

    if os.path.isfile(strFile):
        os.remove(strFile)  # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile)

    plt.close()

#%%
def findMax(redVal, greenRVal, blueVal, greenBVal):
    values = [(redVal, 'redChan'),
              (greenRVal, 'green_rChan'),
              (blueVal, 'blueChan'),
              (greenBVal, 'green_bChan')]
    values = sorted(values, key=lambda values: values[0])

    return values[3], values[2]

#%%
if __name__ == '__main__':

    useImg0 = True  # If true, just use the first image and don't stop to ask which image to use

    relImgDir = "images"  # directory containing raw images relative to current working directory (cwd)
    absImgDir = os.path.join(os.getcwd(), relImgDir)  # absolute image directory

    rawImgList = getImgFnames(absImgDir, 'nef')  # Create list of all raw images in directory

    # Print enumerated list of all files in list
    for idx, imgFname in enumerate(rawImgList):
        print(f'imgFname[{idx}] = {imgFname}')




    # Allow user to select one of the images:
    if useImg0:
        selectedImgNum = 3
    else:
        selectedImgNum = eval(input('Select desired image: '))

    rawImg = readRawImage(rawImgList[selectedImgNum])  # read raw image object
    #_visible so that the mosiaced image will not include the borders around the image
    rawMosaic = rawImg.raw_image_visible # Extract raw arrays from the raw image class. Note: MOSAICED image
    rawCFA = rawImg.raw_colors_visible  # Extract the color-filter array mask; 0,1,2,3 for R, G_r, G_b, B



    print(f'type(rawImage) = {type(rawMosaic)}   rawImage.shape = {rawMosaic.shape}')


    # Display the *mosaiced* image as a grayscale image
    plt.imshow(rawMosaic, cmap='gray')
    plt.title("rawMosaic before splitting")
    plt.show()


    '''
    # Display a small portion of the *mosaiced* image as a grayscale image
    r,c = 0,0  # row,col of upper-left pixel
    h,w = 200,200  # height and width of sub-image to be displayed
    plt.imshow(rawMosaic[r:r+h,c:c+w], cmap='gray')
    plt.show()
    '''


    # Extract one channel at a time:
    redChan     = extractOneBayerChannel(rawMosaic, rawCFA, 0, verbose=False)  # First channel 0 (R)
    green_rChan = extractOneBayerChannel(rawMosaic, rawCFA, 1, verbose=False)  # First channel 1 (G_r)
    blueChan    = extractOneBayerChannel(rawMosaic, rawCFA, 2)  # First channel 2 (B)
    green_bChan = extractOneBayerChannel(rawMosaic, rawCFA, 3)  # First channel 3 (G_b)

    print(f' redChan shape is {redChan.shape}')
    print(f' green_rChan shape is {green_rChan.shape}')
    print(f' blueChan shape is {blueChan.shape}')
    print(f' green_bChan shape is {green_bChan.shape}')

    redChan_flat = redChan.flatten()
    green_rChan_flat = green_rChan.flatten()
    blueChan_flat = blueChan.flatten()
    green_bChan_flat = green_bChan.flatten()
    '''
    redChanMean = np.mean(redChan.flatten())
    redChanStdv = np.std(redChan.flatten())
    print(f'redChanMean = {redChanMean:0.2f} ({redChanStdv:0.2f})')
    green_rChanMean = np.mean(green_rChan.flatten())
    green_rChanStdv = np.std(green_rChan.flatten())
    print(f'green_rChan mean = {green_rChanMean:0.2f} ({green_rChanStdv:0.2f})')
    blueChanMean = np.mean(blueChan.flatten())
    blueChanStdv = np.std(blueChan.flatten())
    print(f'blueChan mean = {blueChanMean:0.2f} ({blueChanStdv:0.2f})')
    green_bChanMean = np.mean(green_bChan.flatten())
    green_bChanStdv = np.std(green_bChan.flatten())
    print(f'green_bChan mean = {green_bChanMean:0.2f} ({green_bChanStdv:0.2f})')
    '''


'''
print(f'redchan.shape = {redChan.shape}')
subImageRed = redChan[0:200, 0:200]
df = subImageRed.flatten()
print(f'redchan.shape = {redChan.shape}')
print(f'df.shape = {df.shape}')
'''


plt.imshow(redChan, cmap='gray')
plt.title("redChan")
plt.show()
# for partitioning parts of images
# calling fit_distribution for each image
# plotting the histogram for each image
imageH, imageW = redChan.shape
indexOfNoise = []
tolerance = 0.6
maxLimit = 70
'''
we will have to decide on the actual noise tolerance later.
This is the value which, once added and subtracted
from the value in an index of a channel, will determine the lower
and upper limit of what is considered 'equal'. Anything outside 
of that range will be considered unequal, and has the potential of being noise.
'''

for i in range(len(redChan_flat)): #since all channels are the same size, we just use red for the range
    noise = False
    max, max2 = findMax(redChan_flat[i], green_rChan_flat[i], blueChan_flat[i], green_bChan_flat[i])
    #tolerance = findTolerance(max)
    '''
    If the value at i passes all of these if statements without entering them,
    then the values are all roughly the same (within tolerance) and it is not
    noise so noise remains False. If any of these if statements are true, then
    noise becomes True.
    If noise is True but enters another if statement, that means that two of the 
    values in the channels are outside of tolerance and that value is not noise,
    so we terminate the current loop iteration.
    '''
    '''
    if (redChan[i] < (green_rChan[i] - tolerance)) or (redChan[i] > (green_rChan[i] + tolerance)):
        noise = not noise
    if (redChan[i] < (blueChan[i] - tolerance)) or (redChan[i] > (blueChan[i] + tolerance)):
        noise = not noise
    if (redChan[i] < (green_bChan[i] - tolerance)) or (redChan[i] > (green_bChan[i] + tolerance)):
        noise = not noise
    '''
    if (max2[0] < tolerance*max[0]) and (max[0] > maxLimit):
        noise = True

    if (noise == True):
        indexOfNoise.append([(i%imageW)*2, (i//imageW)*2, max[1], max[0], max2[0]])

print(indexOfNoise)
name = rawImgList[selectedImgNum].split('\\')
data = np.asarray(indexOfNoise)
pd.DataFrame(indexOfNoise).to_csv(f"new_noise_{name[-1]}_{tolerance}_maxLimit{maxLimit}.csv")
print('--------------------------------------------------------------------------------------')

count = 1

subImageRows, subImageCols = 20, 20
h, w = int((imageH / subImageRows)), int((imageW / subImageCols)) #height and width of the subimages

# for showing the image as subimages
rawImageH, rawImageW = rawMosaic.shape
# so that we don't alter the original data
rawMosaicSplit = rawMosaic
print(f'max of rawMosaicSplit is {np.max(rawMosaicSplit)}')
intervalH = rawImageH//subImageRows
intervalW = rawImageW//subImageCols

#plt.imshow(rawMosaicSplit, cmap='gray')

#list of the top distribution as determined by fit_distribution
chi2list = []
print(f'{len(redChan.flatten())}')


#trim to be used for fit_distribution
upper_pct = 0.999
lower_pct = 0
filepath = (f'histogram/trim_' + str(lower_pct) + '_' + str(upper_pct) + '/')
'''
try:
    os.mkdir(filepath)
    os.mkdir(filepath + 'weibull_min' + '/')
    os.mkdir(filepath + 'norm' + '/')
    os.mkdir(filepath + 'weibull_max' + '/')
    os.mkdir(filepath + 'beta' + '/')
    os.mkdir(filepath + 'invgauss' + '/')
    os.mkdir(filepath + 'uniform' + '/')
    os.mkdir(filepath + 'gamma' + '/')
    os.mkdir(filepath + 'expon' + '/')
    os.mkdir(filepath + 'lognorm' + '/')
    os.mkdir(filepath + 'pearson3' + '/')
    os.mkdir(filepath + 'triang' + '/')
except FileExistsError:
    shutil.rmtree(filepath)
    print('pre-existing directory removed')
    os.mkdir(filepath)
    os.mkdir(filepath + 'weibull_min' + '/')
    os.mkdir(filepath + 'norm' + '/')
    os.mkdir(filepath + 'weibull_max' + '/')
    os.mkdir(filepath + 'beta' + '/')
    os.mkdir(filepath + 'invgauss' + '/')
    os.mkdir(filepath + 'uniform' + '/')
    os.mkdir(filepath + 'gamma' + '/')
    os.mkdir(filepath + 'expon' + '/')
    os.mkdir(filepath + 'lognorm' + '/')
    os.mkdir(filepath + 'pearson3' + '/')
    os.mkdir(filepath + 'triang' + '/')
'''


for y in range(int(subImageRows)):
    rawMosaicSplit[y * intervalH] = 4000  # put in a horizontal line
    rawMosaicSplit[(y * intervalH) + 1] = 4000
    for x in range(int(subImageCols)):
        rawMosaicSplit[:,x*intervalW] = 4000 #put in a vertical line
        rawMosaicSplit[:, (x * intervalW) + 1] = 4000

        r, c = y*h, x*w

        subImageRed = redChan[r:r + h, c:c + w]

        print('\n')
        print(f'number {count}')
        subRedMean = np.mean(subImageRed.flatten())
        subRedStdv = np.std(subImageRed.flatten())
        print(f'subRedMean = {subRedMean:0.2f} ({subRedStdv:0.2f})')

        pixels_red = subImageRed.flatten()
        plt.figure()


        print('Calling fit_distribution ...')
        # fit_distribution('price', 0.99, 0.01)
        # do upper percent, lower percent
        distribution = fit_distribution(pixels_red, upper_pct, lower_pct, y, x)
        chi2list.append(distribution)
        print('Back from fit_distribution ...')
        #histogram will be called in fit_distribution so that it graphs with the trim

        x = x + 1
        count = count + 1

    y = y + 1

plt.imshow(rawMosaicSplit, cmap='gray')
plt.title("rawMosaicSplit after lines")
strFile = (f'histogram/trim_' + str(lower_pct) + '_' + str(upper_pct) + '/' + 'fullImageSplit' + '.png')
plt.savefig(strFile)

#imgShow(subImageRed, title='Red SubImage')

print(chi2list)

print('end of program')

sys.exit()