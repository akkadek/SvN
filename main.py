# This is a sample Python script.

# Simple script and function to read RAW (.nef) file, convert to 3-channel RGB file, and view each as grayscale.

# Original Nikon RAW (.nef) images can be downloaded from Google Drive folder:
# https://drive.google.com/drive/folders/0Bwvtq_Wfkb6yR1h1NmU4bUJwck0?usp=sharing

# The following code sample assumes that at least one NEF (or nef) image file is in a folder named 'images'
# in the current working directory.

#This code cannot analyze pixels along the edge of images at this point in time, due to secondPass
#because this filter requires an examination window centered on the suspicious pixel



import os
import shutil
import rawpy
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math
from scipy.stats import *
import scipy.stats
import scipy.signal
import pandas as pd
import cv2 as cv
import colour_demosaicing

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
        print(f'rawColors is {rawColors}')
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
def imgShow(img, r=0, c=0, h=0, w=0, folder='', title='', cmap='gray', colorbar=True):
    plt.figure()
    if h==0 or w==0:  # optional height, width. If not supplied, use shape of image
        h,w = img.shape[0]-1, img.shape[1]-1

    plt.imshow(img[r:r+h,c:c+w], cmap='gray')

    if title is not '':
        plt.title(title)

    if colorbar:
        plt.colorbar()

    plt.savefig(f'visual_noise/{folder}/{title}_image.png')
    plt.close()
    plt.show()

#%%
# from work with the statistics
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
#from work with the statistics
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
#pass in four values, one from each of the four color channels
#returns the maximum value and the second highest value with their corresponding color values
def findMax(redVal, greenRVal, blueVal, greenBVal):
    values = [(redVal, 'red'),
              (greenRVal, 'green_r'),
              (blueVal, 'blue'),
              (greenBVal, 'green_b')]
    values = sorted(values, key=lambda values: values[0])

    return values[3], values[2]

#%%
#a class to make accessing subimages/examinationn windows of the full image easier
class BayerChannels:
    #initialize by passing in the full color channel image for each of the channels,
    #the column, row coordinate for the upper left pixel of the subimage
    #the width and height of the subimage
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

    #sets the column, row -> is the only way to change the region the code examines
    def set_column_row(self, column, row):
        self._column = column
        self._row = row

    #sets all the channel subimages according to the column, row and the width, height
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

    #using the size of the subimage, it calculates
    #the mean, stdv, median, and IQR for each color channel
    #it returns these values in a 4x4 array
    def all_channel_stats(self):
        red, green_r, blue, green_b = self.all_channel_flat()
        r_q75, r_q25 = np.percentile(red, [75, 25])
        red_iqr = r_q75 - r_q25
        redStats = np.mean(red), np.std(red), np.median(red), red_iqr
        g_r_q75, g_r_q25 = np.percentile(green_r, [75, 25])
        green_r_iqr = g_r_q75 - g_r_q25
        green_rStats = np.mean(green_r), np.std(green_r), np.median(green_r), green_r_iqr
        b_q75, b_q25 = np.percentile(blue, [75, 25])
        blue_iqr = b_q75 - b_q25
        blueStats = np.mean(blue), np.std(blue), np.median(blue), blue_iqr
        g_b_q75, g_b_q25 = np.percentile(green_b, [75, 25])
        green_b_iqr = g_b_q75 - g_b_q25
        green_bStats = np.mean(green_b), np.std(green_b), np.median(green_b), green_b_iqr
        return redStats, green_rStats, blueStats, green_bStats

    @staticmethod
    #return the stats for only the specified color channel
    #we pass in the stats because this is much faster
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
def firstPass(column, row):
    # set the row and column to the coordinates of this subimage
    imageBayerChannels.set_column_row(column, row)

    # retrieve only the subimage portions of each channel that we need
    redSubImage = imageBayerChannels.get_red_subimage()
    green_rSubImage = imageBayerChannels.get_green_r_subimage()
    blueSubImage = imageBayerChannels.get_blue_subimage()
    green_bSubImage = imageBayerChannels.get_green_b_subimage()

    #print the number of the subimage we are on - this is defined in main
    print('\n')
    print(f'number {count}')

    # flatten the subimages for calculating the stats and running through the visual noise loop
    redChan_flat = redSubImage.flatten()
    green_rChan_flat = green_rSubImage.flatten()
    blueChan_flat = blueSubImage.flatten()
    green_bChan_flat = green_bSubImage.flatten()

    # calculate the mean and std dev for each channel in the subimage and save them to a 4x4 array named 'stats'
    #each row has the mean, stdv, median, and IQR for the color channel
    #we create this array here because it significantly speeds up the code
    stats = imageBayerChannels.all_channel_stats()

    # create an array where we will record information
    pass1SuspiciousPixels = []
    # the max proportion the second highest channel value can be of the max value
    pctOfMax = 0.6
    #todo: should pctOfMax be a variable?

    # run through every pixel in the subimage
    for i in range(len(redChan_flat)):  # since all channels are the same size, we just use red for the range

        # find largest and 2nd largest values and their channels
        max, max2 = findMax(redChan_flat[i], green_rChan_flat[i], blueChan_flat[i], green_bChan_flat[i])

        # get the mean, stdv of the subimage for the channel that contains the max
        # pass in stats to decrease program run time
        mean, stdv, median, iqr = imageBayerChannels.one_channel_stats(max[1], stats)

        # the minimum value the max must be to be classified as noise
        maxLimit = mean + stdv
        #todo: maxLimit now is set by the mean and stdv of the subimage

        # calculate the average of the four channel values at this pixel location
        meanOfPixel = np.mean([redChan_flat[i], green_rChan_flat[i], blueChan_flat[i], green_bChan_flat[i]])
        # calculate the standard deviation of the four channel values at this pixel location
        stdvOfPixel = np.std([redChan_flat[i], green_rChan_flat[i], blueChan_flat[i], green_bChan_flat[i]])

        # check conditions to determine if there is noise at this pixel
        if (max2[0] < pctOfMax * max[0]) and (max[0] > maxLimit):
            columnCoordinate = c + (i % w)
            rowCoordinate = r + (i // w)

            # appends x coordinate, y coordinate, the channel that contains the damage,
            # the damaged channel value (the max), and the 2nd highest channel value,
            # the RGBG2 mean and standard deviations, the average and stdv of the four channels
            # at this pixel location, and the weighted stdv at this location
            pass1SuspiciousPixels.append([columnCoordinate * 2, rowCoordinate * 2, max[1], max[0], max2[0],
                                         stats[0][0], stats[0][1], stats[1][0], stats[1][1], stats[2][0],
                                         stats[2][1], stats[3][0], stats[3][1],
                                         meanOfPixel, stdvOfPixel, stdvOfPixel / meanOfPixel
                                         ])

    return pass1SuspiciousPixels

#%%
#secondPass takes the results of each subimage from firstPass as they are output
def secondPass(column, row):
    #todo: can we pass in the threshold for the zscore here?
    #secondRunBayerChannels will have a 'subimage' size of whatever the variable 'examinationWindowSize' is set to
    secondRunBayerChannels.set_column_row(column, row)

    # stats of 5x5 (or examinationWindowSize x examinationWindowSize) subimage
    try:
        subStats = secondRunBayerChannels.all_channel_stats()
    except IndexError:
        # if a suspicious pixel is along an edge, where the examination window is not completely within the image
        # it results in an error
        # this can probably be fixed later, but for now we bypass it
        print(f'edge of image')
        return pass2SuspiciousPixels, pass2ManualWindow

    # check specified pixel to see if it makes it through the second filter
    # pass1SuspiciousPixels[pixel][0] = the column the suspicious pixel is in (x - coordinate)
    # pass1SuspiciousPixels[pixel][1] = the row the suspicious pixel is in (y - coordinate)
    # pass1SuspiciousPixels[pixel][2] = channel that contains the suspicious pixel
    # pass1SuspiciousPixels[pixel][3] = the value of the max channel of the suspicious pixel
    # pass1SuspiciousPixels[pixel][4] = the second highest value of the suspicious pixel

    mean, stdv, median, iqr = secondRunBayerChannels.one_channel_stats(pass1SuspiciousPixels[pixel][2], subStats)
    #todo: where is pixel set? and what does it represent? pixel is created in main, it refers to the index of pass1SuspiciousPixels, or the coordinates and stats of the saved suspicious pixel

    zscore = (pass1SuspiciousPixels[pixel][3] - mean) / stdv
    mz_score = 0.6745*((pass1SuspiciousPixels[pixel][3] - median)/iqr)
    #0.6745 is to make mz_score comparable to zscore for a gaussian
    if  (zscore > 1.5):
        #todo: threshold zscore for secondPass is set to 1.5 in secondPass() - make it adjustable?

        # append column, row, channel of suspicious pixel,
        # max, 2nd highest, mean of 5x5, stdv of 5x5, zscore of max in 5x5
        pass2SuspiciousPixels.append(
            [pass1SuspiciousPixels[pixel][0], pass1SuspiciousPixels[pixel][1], pass1SuspiciousPixels[pixel][2],
             pass1SuspiciousPixels[pixel][3], pass1SuspiciousPixels[pixel][4],
             mean, stdv, zscore, median, iqr, mz_score
             ])

        #this will be the total hit rate and will be used to calculate the false alarms
        #within the manual labelling window
        #todo: convert 510 and 284 to variables dependent on subimage size and number that are being manually labelled
        if (pass1SuspiciousPixels[pixel][0] < 510) and (pass1SuspiciousPixels[pixel][1] < 284):
            pass2ManualWindow.append(
                [pass1SuspiciousPixels[pixel][0], pass1SuspiciousPixels[pixel][1], pass1SuspiciousPixels[pixel][2],
                 pass1SuspiciousPixels[pixel][3], pass1SuspiciousPixels[pixel][4],
                 mean, stdv, zscore, median, iqr, mz_score
                 ])


    return pass2SuspiciousPixels, pass2ManualWindow

#%%
def replaceNoise(noise, red, green_r, blue, green_b):
    for i in range(len(noise)):
        #divide by two because the coordinates are according to the full image
        c, r = noise[i][0]//2, noise[i][1]//2  #c, r because this is like an x, y coordinate
        colorChan = noise[i][2]

        #todo: make median filter size variable?
        #section is always 3x3
        if colorChan == 'red':
            section = red[r-1:r+2, c-1:c+2]
            med = int(np.median(section))
            red[r][c] = med
        elif colorChan == 'green_r':
            section = green_r[r - 1:r + 2, c - 1:c + 2]
            med = int(np.median(section))
            green_r[r][c] = med
        elif colorChan == 'blue':
            section = blue[r - 1:r + 2, c - 1:c + 2]
            med = int(np.median(section))
            blue[r][c] = med
        elif colorChan == 'green_b':
            section = green_b[r - 1:r + 2, c - 1:c + 2]
            med = int(np.median(section))
            green_b[r][c] = med

    return red, green_r, blue, green_b

#%%
def remosaicNoNoiseColorChans(rawMosaic_rows, rawMosaic_columns, sus_removed_red, sus_removed_green_r, sus_removed_blue, sus_removed_green_b):
    # make an array of zeros
    CFA = np.zeros((rawMosaic_rows, rawMosaic_columns))

    for idx in range(rawMosaic_rows * rawMosaic_columns):
        # determine the row, column coordinate this iteration is on
        row, column = idx // rawMosaic_columns, idx % rawMosaic_columns
        # save the value to use outside of the if statements
        value = -1

        # determine if red, green_r, green_b, or blue channel values go in this coordinate placement
        if (row % 2 == 0) and (column % 2 == 0):
            # if both the row and column are even
            # then this is the red channel value
            value = sus_removed_red[int(row / 2)][int(column / 2)]
        elif (row % 2 == 0) and (column % 2 == 1):
            # if the row is even but the column is odd
            # this is the green_r channel value
            value = sus_removed_green_r[int(row / 2)][int((column - 1) / 2)]
        elif (row % 2 == 1) and (column % 2 == 0):
            # if the row is odd but the column is even
            # this is the green_b channel value
            value = sus_removed_green_b[int((row - 1) / 2)][int(column / 2)]
        elif (row % 2 == 1) and (column % 2 == 1):
            # if both the row and the column are odd
            # this is the blue channel
            value = sus_removed_blue[int((row - 1) / 2)][int((column - 1) / 2)]

        CFA[row][column] = value

    return CFA

#%%
if __name__ == '__main__':

    defaultImgNum = 25 # If positive, just use this image number and don't stop to ask which image to use

    relImgDir = "images"  # directory containing raw images relative to current working directory (cwd)
    absImgDir = os.path.join(os.getcwd(), relImgDir)  # absolute image directory

    rawImgList = getImgFnames(absImgDir, 'nef')  # Create list of all raw images in directory

    # Print enumerated list of all files in list
    for idx, imgFname in enumerate(rawImgList):
        # the name of the image
        shortname = rawImgList[idx].split('\\')
        print(f'imgFname[{idx}] = {shortname[-1]}')

    # Allow user to select one of the images:
    if (defaultImgNum > 0):
        selectedImgNum = defaultImgNum
    else:
        selectedImgNum = eval(input('Select desired image: '))

    startTime = time.time()
    print(f'Clock started')

    rawImg = readRawImage(rawImgList[selectedImgNum], verbose=True)  # read raw image object

    #we are using _visible so that the mosiaced image will not include the borders around the image
    rawMosaic = rawImg.raw_image_visible # Extract raw arrays from the raw image class. Note: MOSAICED image
    rawCFA = rawImg.raw_colors_visible  # Extract the color-filter array mask; 0,1,2,3 for R, G_r, G_b, B
    print(f'rawCFA is {rawCFA}')

    #print the type and shape of the rawMosaic
    print(f'type(rawMosaic) = {type(rawMosaic)}   rawMosaic.shape = {rawMosaic.shape}')

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

    #the name of the image,  will be used later when we save files
    name = rawImgList[selectedImgNum].split('\\')

    print(f'redChan shape is {redChan.shape}')
    print(f'green_rChan shape is {green_rChan.shape}')
    print(f'blueChan shape is {blueChan.shape}')
    print(f'green_bChan shape is {green_bChan.shape}')

    #this directory is just for storing the color channel images
    try:
        os.mkdir(f'{name[-1]}')
    except FileExistsError:
        shutil.rmtree(f'{name[-1]}')
        os.mkdir(f'{name[-1]}')
    #apply histogram equalization to the images
    # the color channels are 16 bit so we have to convert them to 8 bit
    # todo: are all the images 16 bit? Make the conversion to 8 bit dependent on image bit size
    #todo: look at np.normalize instead of equalizeHist
    red_equalized = cv.equalizeHist((redChan/256).astype('uint8'))
    green_r_equalized = cv.equalizeHist((green_rChan/256).astype('uint8'))
    blue_equalized = cv.equalizeHist((blueChan/256).astype('uint8'))
    green_b_equalized = cv.equalizeHist((green_bChan/256).astype('uint8'))
    #save the equalized images for each color channel
    cv.imwrite(f'{name[-1]}/OG_red_equ.tif', red_equalized)
    cv.imwrite(f'{name[-1]}/OG_green_r_equ.tif', green_r_equalized)
    cv.imwrite(f'{name[-1]}/OG_blue_equ.tif', blue_equalized)
    cv.imwrite(f'{name[-1]}/OG_green_b_equ.tif', green_b_equalized)

    #a variable that will count the subimages that are being processed as they are processed
    count = 1

    #the following calculations assume all of the four bayer channels have the same dimensions
    imageH, imageW = redChan.shape

    # height and width of the subimages
    #h, w = int((imageH / subImageRows)), int((imageW / subImageCols))
    #todo: make subimage size a variable
    #due to the manual window code we want the windows to be 142x170 sized
    #because the single color channels are of smaller dimensions they have to be halved here
    h, w = 142//2, 170//2 #divide by two because 142x170 is for the mosaiced image size
    subImageRows = imageH//h #the number of rows that the full image will be divided into
    subImageCols = imageW//w #the number of columns that the full image will be divided into

    #rawMosaic has different dimenstions than the 4 bayer channels (2x size)
    rawImageH, rawImageW = rawMosaic.shape
    print(f' imageH, imageW = {imageH}, {imageW} \n rawImageH, rawImageW = {rawImageH}, {rawImageW}')

    # for showing the image with lines dividing it into subimages
    # so that we don't alter the original data
    rawMosaicSplit = rawMosaic.copy()
    lineConst = np.amax(rawMosaicSplit) - 1 #the value of the lines drawn on the output image
    #the intervals between the lines that are drawn
    intervalH = rawImageH//subImageRows
    intervalW = rawImageW//subImageCols

    c, r = 0, 0 #column, row to initialize object - will be changed inside the loop
    imageBayerChannels = BayerChannels(redChan, green_rChan, blueChan, green_bChan, c, r, w, h)
    #initialization of object that will be used in secondPass
    examinationWindowSize = 5 #the examination window is square so this is the width and height
    #todo: define examinationWindowSize variable at the top of main
    secondRunBayerChannels = BayerChannels(redChan, green_rChan, blueChan, green_bChan,
                                           c, r, examinationWindowSize, examinationWindowSize)
    #to make these two arrays accessible outside of the loop
    pass2SuspiciousPixels = []
    pass2ManualWindow = []

    # folder for where the firstPass csv files will be saved
    folder = f'{name[-1]}_filter1'
    # try - except will delete a preexisting file of the same name and replace it with the new one being created
    #todo: NOTE: this uses a directory named 'visual_noise' - either create that directory in the working one, or delete it from the code here
    try:
        os.mkdir(f'visual_noise/{folder}')
    except FileExistsError:
        shutil.rmtree(f'visual_noise/{folder}')
        os.mkdir(f'visual_noise/{folder}')

    timestamp1 = time.time() - startTime
    print(f'Time before entering the filters: {(timestamp1):.2f} seconds, {(timestamp1/60):.2f} minutes')

    #determine which pixels in the image are suspicious by running them through two filters
    #the loops run through the image in subimages, first by row then by column
    for y in range(int(subImageRows)):
        rawMosaicSplit[y * intervalH] = lineConst # put in a horizontal line
        rawMosaicSplit[(y * intervalH) + 1] = lineConst
        for x in range(int(subImageCols)):
            rawMosaicSplit[:,x*intervalW] = lineConst #put in a vertical line
            rawMosaicSplit[:, (x * intervalW) + 1] = lineConst

            r, c = y*h, x*w #row, column of upper left corner of subimage
            #note: column is x coordinate, row is y coordinate

            #run the subimage through the first pass filter
            #which finds suspicious pixels in the subimage and saves them in the returned array
            pass1SuspiciousPixels = firstPass(c, r)

            #todo: to save the csv files from firstPass, uncomment code here
            '''
            #save the recorded information from the subimage
            data = np.asarray(pass1SuspiciousPixels)
            title = f"visual_noise/{folder}/row{y+1}_column{x+1}_{name[-1]}_wRatioMin_pctOfMax0.6_xy{c * 2}-{r * 2}_wh{w * 2}-{h * 2}.csv"
            headers = ['column', 'row', 'max channel', 'maximum', '2nd highest', 'red mean', 'red stdev', 'green_r mean', 'green_r stdev',
                            'blue mean', 'blue stdev', 'green_b mean', 'green_b stdev', 'pixel mean', 'pixel stdev', 'pixel stdev/mean']
            # create a csv file with the information
            try:
                pd.DataFrame(pass1SuspiciousPixels).to_csv(title, header= headers)
            except FileExistsError:
                # if a csv file of the same name already exists, delete it and make a new one
                shutil.rmtree(title)
                pd.DataFrame(pass1SuspiciousPixels).to_csv(title, header=headers)
            except ValueError:
                #if there is no information to record (no noise in the subimage) then delete the created csv
                os.remove(title)
                print('No suspicious pixels')
                pass
            '''

            print('--------------------------------------------------------------------------------------')

            #SECOND FILTER
            #examine the 5x5 surrounding subimage of suspicious pixels
            #this is within the loop because it takes the results of each subimage as they are returned
            for pixel in range(len(pass1SuspiciousPixels)):
                # access the coordinates of the suspicious pixel by directly pulling them from the recorded values in pass1SuspiciousPixels
                # column and row of the suspicious pixel
                c_suspiciousPixel, r_suspiciousPixel = int(pass1SuspiciousPixels[pixel][0]/2), int(pass1SuspiciousPixels[pixel][1]/2)
                # integer division to find what the top left pixel of the examination window should be set to
                offset = examinationWindowSize//2

                # sets column, row to the upper left of the wanted sub-sub image
                column, row = c_suspiciousPixel - offset, r_suspiciousPixel - offset

                #dtermine if the pixels determined to be suspicious by the first pass filter
                # are still suspicious within their own square examination window
                # if they are they will be saved to an array and returned
                pass2SuspiciousPixels, pass2ManualWindow = secondPass(column, row)

                #pas2SuspiciousPixels is all the pixels in the image that make it through the second filter
                #pass2ManualWindow is all the pixels in the manual labelling window that make it though the second filter

            x = x + 1
            count = count + 1
    
        y = y + 1

    #make sure the data is in array form to be saved into csv files
    np.asarray(pass2SuspiciousPixels)
    np.asarray(pass2ManualWindow) #this saves only the suspcious pixels present in the image in the upper left 510x284 section
    #todo: visual_noise filename
    title = f"visual_noise/{name[-1]}_filter2_squareWindowSize{examinationWindowSize}.csv"
    titleForManual = f'visual_noise/{name[-1]}_filter2_size{examinationWindowSize}_manualList.csv'
    headers = ['column', 'row', 'max channel', 'maximum', '2nd highest', 'max channel average', 'max channel stdv',
               'zscore of max', 'median', 'IQR', 'MZ-score']
    # create a csv file with the pixels that have passed through both filters (suspicious pixels)
    try:
        pd.DataFrame(pass2SuspiciousPixels).to_csv(title, header=headers)
        pd.DataFrame(pass2ManualWindow).to_csv(titleForManual, header=headers)
    except FileExistsError:
        # if a csv file of the same name already exists, delete it and make a new one
        shutil.rmtree(title)
        shutil.rmtree(titleForManual)
        pd.DataFrame(pass2SuspiciousPixels).to_csv(title, header=headers)
        pd.DataFrame(pass2ManualWindow).to_csv(titleForManual, header=headers)
    except ValueError:
        # if there is no information to record (no noise in the subimage) then delete the created csv
        os.remove(title)
        os.remove(titleForManual)
        print('No noise')
        pass

    timestamp2 = time.time() - startTime
    print(f'Time after saving the results of secondPass: {(timestamp2):.2f} seconds, {(timestamp2/60):.2f} minutes')

    #display the image with lines dividing it into the subimages
    plt.imshow(rawMosaicSplit, cmap='gray')
    plt.title("rawMosaicSplit after lines")
    strFile = (f'visual_noise' + '/' + str(name[-1]) + '_split' + '.png')
    plt.savefig(strFile)

    #use a median filter to replace all suspicious pixels in each of the color channels of the image
    #sus for suspicious
    sus_removed_red, sus_removed_green_r, sus_removed_blue, sus_removed_green_b = replaceNoise(
        pass2SuspiciousPixels, redChan, green_rChan, blueChan, green_bChan)

    timestamp3 = time.time() - startTime
    print(f'Time after removing suspicious pixels from each of the color channels: {(timestamp3):.2f} seconds, {(timestamp3/60):.2f} minutes')

    #apply histogram equalization to the altered color channel images
    #the color channels are 16 bit so we have to convert them to 8 bit
    #todo: are all the images 16 bit? Make the conversion to 8 bit dependent on image bit size
    corrected_red_equalized = cv.equalizeHist((sus_removed_red/256).astype('uint8'))
    corrected_green_r_equalized = cv.equalizeHist((sus_removed_green_r/256).astype('uint8'))
    corrected_blue_equalized = cv.equalizeHist((sus_removed_blue/256).astype('uint8'))
    corrected_green_b_equalized = cv.equalizeHist((sus_removed_green_b/256).astype('uint8'))
    #save the individual color channel images
    cv.imwrite(f'{name[-1]}/corrected_red_equ.tif', corrected_red_equalized)
    cv.imwrite(f'{name[-1]}/corrected_green_r_equ.tif', corrected_green_r_equalized)
    cv.imwrite(f'{name[-1]}/corrected_blue_equ.tif', corrected_blue_equalized)
    cv.imwrite(f'{name[-1]}/corrected_green_b_equ.tif', corrected_green_b_equalized)


    #determine the shape of the rawMosaic to use to re mossaic the four color channels
    #number of rows, number of columns = 2844, 4284 for image iss041e008803
    number_of_rows, number_of_columns = rawMosaic.shape
    #pass the rawMosaic rows and columns and the altered four color channels to the re-mosaicing method
    CFA = remosaicNoNoiseColorChans(
        number_of_rows, number_of_columns, sus_removed_red, sus_removed_green_r, sus_removed_blue, sus_removed_green_b)
    CFA = CFA.astype(np.uint16) #make sure that all the values in the image are integers

    # beyond this point is attempts at demosaicing the altered color channels and printing the final image
    colorImage = cv.demosaicing(CFA, cv.COLOR_BayerBG2BGR)

    #colorImage = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(CFA)
    #cv.imwrite('image1.tif', colorImage)
    #todo: why do different images take different lengths of time
    timestamp4 = time.time() - startTime
    print(f'Time at end of program: {(timestamp4):.2f} seconds, {(timestamp4/60):.2f} minutes')

    #sys.exit()