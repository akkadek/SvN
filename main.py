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
    values = [(redVal, 'red'),
              (greenRVal, 'green_r'),
              (blueVal, 'blue'),
              (greenBVal, 'green_b')]
    values = sorted(values, key=lambda values: values[0])

    return values[3], values[2]

#%%
class BayerChannel:
    def __init__(self, red, green_r, blue, green_b, column, row, width, height):
        self._red = red
        self._green_r = green_r
        self._blue = blue
        self._green_b = green_b
        self._column = column
        self._row = row
        self._width = width
        self._height = height

    def get_red(self):
        return self._red

    def get_green_r(self):
        return self._green_r

    def get_blue(self):
        return self._blue

    def get_green_b(self):
        return self._green_b

    def set_column_row(self, column, row):
        self._column = column
        self._row = row

    def set_all_channel_sub_images(self):
        red = self._red[self._row:self._row + self._height, self._column:self._column + self._width]
        green_r = self._green_r[self._row:self._row + self._height, self._column:self._column + self._width]
        blue = self._blue[self._row:self._row + self._height, self._column:self._column + self._width]
        green_b = self._green_b[self._row:self._row + self._height, self._column:self._column + self._width]

        return red, green_r, blue, green_b

    def get_red_subimage(self):
        channels = self.set_all_channel_sub_images()
        return channels[0]

    def get_green_r_subimage(self):
        channels = self.set_all_channel_sub_images()
        return channels[1]

    def get_blue_subimage(self):
        channels = self.set_all_channel_sub_images()
        return channels[2]

    def get_green_b_subimage(self):
        channels = self.set_all_channel_sub_images()
        return channels[3]

    def all_channel_flat(self):
        red, green_r, blue, green_b = self.set_all_channel_sub_images()
        return red.flatten(), green_r.flatten(), blue.flatten(), green_b.flatten()

    def all_channel_stats(self):
        red, green_r, blue, green_b = self.all_channel_flat()
        redStats = np.mean(red), np.std(red)
        green_rStats = np.mean(green_r), np.std(green_r)
        blueStats = np.mean(blue), np.std(blue)
        green_bStats = np.mean(green_b), np.std(green_b)
        return redStats, green_rStats, blueStats, green_bStats

    @staticmethod
    def one_channel_stats(channel, stats):
        if channel == 'red':
            return stats[0]
        if channel == 'green_r':
            return stats[1]
        if channel == 'blue':
            return stats[2]
        if channel == 'green_b':
            return stats[3]


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
        selectedImgNum = 2
    else:
        selectedImgNum = eval(input('Select desired image: '))

    rawImg = readRawImage(rawImgList[selectedImgNum])  # read raw image object
    #_visible so that the mosiaced image will not include the borders around the image
    rawMosaic = rawImg.raw_image_visible # Extract raw arrays from the raw image class. Note: MOSAICED image
    rawCFA = rawImg.raw_colors_visible  # Extract the color-filter array mask; 0,1,2,3 for R, G_r, G_b, B

    print(f'type(rawImage) = {type(rawMosaic)}   rawImage.shape = {rawMosaic.shape}')

    # Display the *mosaiced* image as a grayscale image
    '''
    plt.imshow(rawMosaic, cmap='gray')
    plt.title("rawMosaic before splitting")
    plt.show()
    '''

    # Extract one channel at a time:
    redChan     = extractOneBayerChannel(rawMosaic, rawCFA, 0, verbose=False)  # First channel 0 (R)
    green_rChan = extractOneBayerChannel(rawMosaic, rawCFA, 1, verbose=False)  # First channel 1 (G_r)
    blueChan    = extractOneBayerChannel(rawMosaic, rawCFA, 2)  # First channel 2 (B)
    green_bChan = extractOneBayerChannel(rawMosaic, rawCFA, 3)  # First channel 3 (G_b)

    #the name of the image
    name = rawImgList[selectedImgNum].split('\\')

    print(f'redChan shape is {redChan.shape}')
    print(f'green_rChan shape is {green_rChan.shape}')
    print(f'blueChan shape is {blueChan.shape}')
    print(f'green_bChan shape is {green_bChan.shape}')
    '''
    r, c = 2055, 825 #upper left corner: row, column coordinate
    w, h = 10, 10  #width and height of subimage

    #subimages of the four channels, original size: 2142 x 1422
    redSubimage = redChan[c:c+h, r:r+w]
    green_rSubimage = green_rChan[c:c+h, r:r+w]
    blueSubimage = blueChan[c:c+h, r:r+w]
    green_bSubimage = green_bChan[c:c+h, r:r+w]

    redChan_flat = redSubimage.flatten()
    green_rChan_flat = green_rSubimage.flatten()
    blueChan_flat = blueSubimage.flatten()
    green_bChan_flat = green_bSubimage.flatten()

    indexOfNoise = []
    tolerance = 0.6
    maxLimit = 70

    for i in range(len(redChan_flat)): #since all channels are the same size, we just use red for the range
        noise = False
        max, max2 = findMax(redChan_flat[i], green_rChan_flat[i], blueChan_flat[i], green_bChan_flat[i])

        if (max2[0] < tolerance*max[0]) and (max[0] > maxLimit):
            noise = True

        if (noise == True):
            #appends x coordinate, y coordinate, the channnel that contains the damage,
            #the damages channel value (the max), and the 2nd highest channel value
            indexOfNoise.append([(r + (i%w))*2, (c + (i//w))*2, max[1], max[0], max2[0]])
            #r+(i%w), c+(i//w)

    print(indexOfNoise)
    name = rawImgList[selectedImgNum].split('\\')
    data = np.asarray(indexOfNoise)
    pd.DataFrame(indexOfNoise).to_csv(f"{name[-1]}_tolerance{tolerance}_maxLimit{maxLimit}_xy{r*2}-{c*2}_wh{w*2}-{h*2}.csv")
    print('--------------------------------------------------------------------------------------')
    '''

    count = 1

    #the following calculations assume all of the four bayer channels have the same dimensions
    imageH, imageW = redChan.shape
    subImageRows, subImageCols = 25, 25
    # height and width of the subimages
    h, w = int((imageH / subImageRows)), int((imageW / subImageCols))

    #rawMosaic has different dimenstions than the 4 bayer channels (2x size)
    rawImageH, rawImageW = rawMosaic.shape
    print(f' imageH, imageW = {imageH}, {imageW} \n rawImageH, rawImageW = {rawImageH}, {rawImageW}')
    # for showing the image as subimages
    # so that we don't alter the original data
    rawMosaicSplit = rawMosaic
    intervalH = rawImageH//subImageRows
    intervalW = rawImageW//subImageCols

    c, r = 0, 0 #to initialize object - will be changed inside the loop
    image3 = BayerChannel(redChan, green_rChan, blueChan, green_bChan, c, r, w, h)
    # for partitioning parts of images

    for y in range(int(subImageRows)):
        rawMosaicSplit[y * intervalH] = 4000  # put in a horizontal line
        rawMosaicSplit[(y * intervalH) + 1] = 4000
        for x in range(int(subImageCols)):
            rawMosaicSplit[:,x*intervalW] = 4000 #put in a vertical line
            rawMosaicSplit[:, (x * intervalW) + 1] = 4000
    
            r, c = y*h, x*w #row, column of upper left corner of subimage
            #note to self: column is x coordinate, row is y coordinate
            image3.set_column_row(c, r)

            #subImage designation assigns height, then width of subimage
            redSubImage = image3.get_red_subimage()
            green_rSubImage = image3.get_green_r_subimage()
            blueSubImage = image3.get_blue_subimage()
            green_bSubImage = image3.get_green_b_subimage()

            print('\n')
            print(f'number {count}')

            redChan_flat = redSubImage.flatten()
            green_rChan_flat = green_rSubImage.flatten()
            blueChan_flat = blueSubImage.flatten()
            green_bChan_flat = green_bSubImage.flatten()

            stats = image3.all_channel_stats()
            print(f'all_channel_stats {stats}')

            plt.figure()

            indexOfNoise = []
            tolerance = 0.6


            for i in range(len(redChan_flat)):  # since all channels are the same size, we just use red for the range
                noise = False
                max, max2 = findMax(redChan_flat[i], green_rChan_flat[i], blueChan_flat[i], green_bChan_flat[i])
                mean, stdv = image3.one_channel_stats(max[1], stats)
                maxLimit = mean + stdv
                #maxLimit = 70

                if (max2[0] < tolerance * max[0]) and (max[0] > maxLimit) and (max[0] > 30):
                    noise = True

                if (noise == True):
                    # appends x coordinate, y coordinate, the channel that contains the damage,
                    # the damages channel value (the max), and the 2nd highest channel value
                    indexOfNoise.append([(c + (i % w)) * 2, (r + (i // w)) * 2, max[1], max[0], max2[0]])

            #print(indexOfNoise)
            data = np.asarray(indexOfNoise)
            pd.DataFrame(indexOfNoise).to_csv(
                f"visual_noise/variable maxLimit with minimum/row{y+1}_column{x+1}_{name[-1]}_tolerance{tolerance}_xy{c * 2}-{r * 2}_wh{w * 2}-{h * 2}.csv")
            print('--------------------------------------------------------------------------------------')

            x = x + 1
            count = count + 1
    
        y = y + 1
    
    plt.imshow(rawMosaicSplit, cmap='gray')
    plt.title("rawMosaicSplit after lines")
    strFile = (f'visual_noise' + '/' + str(name[-1]) + '_split' + '.png')
    plt.savefig(strFile)


    print('end of program')

    sys.exit()