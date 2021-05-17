## CRDIR_III_funcs.py
# import CRDIR_III_funcs as C3f
# Jeff B. Pelz  CIS RIT

# Helper functions for CRDIR-III project - Summer 2020
# Jeff Pelz

##

import os
import time
import numpy as np
import exifread
import imageio
import scipy.stats as stats
import rawpy
import cv2

import matplotlib.pyplot as plt


##


def list_nef_images(dir=None, verbose=False):
    #  Jeff B. Pelz May 2020
    #  Return a list of all NEF images in the current (or specified) directory.  If a different directory is
    #  specified, return to original directory before returning.

    #  input
    #         [default] dir=None  str directory to search for nef files
    #         [default] verbose=False boolean
    #  output
    #         list of str nefList - list of images in dir

    originalWorkingdirectory = os.getcwd()

    if dir is not None:  # if a directory is provided:
        # Save current working directory

        if verbose:
            print(f'originalWorkingDirectory = {originalWorkingdirectory}')

        # Set image library directory
        os.chdir(dir)
        if verbose:
            print(f'Now in {os.getcwd()}')
    else:  # dir == None
        dir = os.getcwd()  # Current directory - default unless dir is specified

    # Get listing of all nef images in current (or specified) directory:
    dirListing = os.listdir()  # get all files and directories

    nefList = []  # Initialize list to empty list
    for file in dirListing:
        if file.endswith(('nef', 'NEF')):  # create a list of all NEF (or nef) files
            nefList.append(file)

    if verbose:
        print(f'All nef files in directory {dir}:')
        for idx, nef in enumerate(nefList):
            print(f'{idx:5}: {nef}')

    # Return to original working directory
    os.chdir(originalWorkingdirectory)

    return nefList  # Return list of nef images

##


def extract_thumbnail(imFname, dir='', verbose=False):
    with rawpy.imread(dir+imFname) as raw:
        # raises rawpy.LibRawNoThumbnailError if thumbnail missing
        # raises rawpy.LibRawUnsupportedThumbnailError if unsupported format
        thumb = raw.extract_thumb()
        # if verbose:
        #     print(thumb)
    if thumb.format == rawpy.ThumbFormat.JPEG:
        print(f'Thumbnail opened; already in JPG format; saving to thumb.jpg')
        # thumb.data is already in JPEG format, save as-is
        with open('thumb.jpg', 'wb') as f:
            f.write(thumb.data)
    elif thumb.format == rawpy.ThumbFormat.BITMAP:
        print(f'Thumbnail opened; in BITMAP format; saving to thumb.jpg with imageio')
        # thumb.data is an RGB numpy array, convert with imageio
        imageio.imsave('thumb.jpg', thumb.data)
    else:
        print(f'Thumbnail not opened, or not in JPG or BITMAP formats ...')

##

def read_raw_image(imFname, dir='', verbose=False):

    try:
        rawImg = rawpy.imread(dir+imFname)  # Read the raw file into a RawPy object

        retDict = dict([('rawImg', rawImg),                    # Raw image - full
                      ('rawType', rawImg.raw_type),            # Raw images can be 'Flat' or 'Stack'
                      ('rawArray', rawImg.raw_image),          # The raw image as a numpy array
                      ('rawVisArr', rawImg.raw_image_visible), # The visible part of the raw array
                      ('rgbImg', rawImg.postprocess())])       # Demosaiced raw image into RGB image
        if verbose:
            print(f"\^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ")
            print(f"\nread_raw_image(): Returning {imFname} from {dir}.")
            print(f"rawImg.raw_image.shape = {rawImg.raw_image.shape}")
            print(f"rawImg.raw_image_visible.shape = {rawImg.raw_image_visible.shape}\n")
            print(f"rawImg.postprocess().shape = {rawImg.postprocess().shape}\n")

        return retDict

    except Exception as errMsg:
        print(f"Error reading {imFname} from {dir} ({errMsg}): returning None")
        print(f"Full path: {dir+imFname}")
        return None

##


# Define a function to extract the EXIF data from an image and return a subset
def EXIF_data_keyword(imgFname, tagKeyKeyword='', verbose=False):
    # Pass in image filename (with path) and keyword for key(s);
    # Return dict with desired key/value pairs
    # (Default ='' returns *all* key/value pairs)

    # Open file, read EXIF tags, close file
    f = open(imgFname, 'rb')
    AllTags = exifread.process_file(f)
    f.close()

    # Convert tagKeyKeyword to lower case
    tagKeyKeyword = tagKeyKeyword.lower()

    keyWordDict = {}  # initialize empty dict

    # Check key in the dict for an occurence of anything in the tagKeyList
    for tagKey in AllTags.keys():
        if tagKeyKeyword in tagKey.lower():
            keyWordDict[tagKey] = AllTags[tagKey]  # Add the tag to the dict to be returned
            if verbose:
                print(f'{tagKey}: {AllTags[tagKey]}')

    return keyWordDict

##


def EXIF_data_keyword_list(imgFname, tagKeyKeywordList=[''], verbose=False):

    combinedKeyWordDict = {}  # set up empty dictionary to hold combination of all

    for tagKeyword in tagKeyKeywordList:  # Call EXIF_data_keyword() for each keyword in list
        tagDict = EXIF_data_keyword(imgFname, tagKeyKeyword=tagKeyword, verbose=verbose)
        combinedKeyWordDict.update(tagDict)  # update (concatenate) tagDict into the combinedKeyWordDict

    return combinedKeyWordDict


##


def read_and_display_img(fName, dir='', figsize=(5,5), title='img',  cmap='gray', colorbar=False):
    if dir != '':  # if not empty string check for trailing /
        if not dir.endswith('/'):
            dir = dir+'/'

    img = imageio.imread(dir+fName)
    display_img(img, title=fName, figsize=figsize,  cmap='gray', colorbar=False)

##

def display_img(img, figsize=(5, 5), title='img', cmap='gray', colorbar=False, saveImages=False):
    # Jeff Pelz May 10, 2020
    # Generic (2D or 3D color) image display routine

    # input
    #        img: 2D or 3D nparray
    #        [optional] figsize:  tuple default=(5,5) - size of figure
    #        [optional] title:    str default='img' - title displayed over figure
    #        [optional] cmap:     str default='gray' - matplotlib colormap (see matplotlib documentation)
    #        [optional] colorbar: bool default=False - display colorbar next to image?

    # output
    #        bool True = successful display, False = failure

    if len(img.shape) == 3:  # 3D (color) image
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(title)
        if saveImages:
            plt.savefig(f'{title}.pdf')
        plt.show()
        return True

    elif len(img.shape) == 2:  # 2D (monochrome) image
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap=cmap)
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        if saveImages:
            plt.savefig(f'{title}.pdf')
        plt.show()
        return True

    else:  # invalid
        print('display_img() invalid input: img must be 2D or 3D image file')
        return False

##

def display_subimg_in_context(img, subImg, ulLoc, figsize=(5, 5),
                              title='SubImage in Context',
                              cmap='gray', colorbar=False,
                              saveImages=False, verbose=False):
    # Jeff Pelz May 10, 2020
    # Show subimage in image in context - ulLoc is location of upper-left pixel of subimg in img

    # input
    #        img: nparray (2D or 3D)
    #        subImg:  nparray (2D or 3D) - a 'subimage' from img
    #        ulLoc: tuple (R,C) ints defining location of upper-left pixel in img
    #        [optional] figsize:  tuple default=(10,10) - size of figure
    #        [optional] title:    str default='SubImage in Context' - title displayed over figure
    #        [optional] cmap:     str default='gray' - matplotlib colormap (see matplotlib documentation)
    #        [optional] colorbar: bool default=False - display colorbar next to image?

    # output
    #        bool True = successful display, False = failure

    if len(img.shape) == 3:  # color image

        # Superimpose subimg into img at location:
        tempImg = img.copy()
        subImgH, subImgW, sumImgD = subImg.shape

        # Copy the subImage into the correct location of the image
        tempImg[ulLoc[0]:ulLoc[0] + subImgH, ulLoc[1]:ulLoc[1] + subImgW, :] = subImg

        plt.figure(figsize=figsize)
        plt.imshow(tempImg)
        plt.title(title)
        plt.show()
        return True

    elif len(img.shape) == 2:  # monochrome image

        # Superimpose subimg into img at location:
        tempImg = img.copy()
        subImgH, subImgW = subImg.shape

        if verbose:
            print(f'\n= = = = = = = = = = = = = = = = = = = = = = = = =\nIn display_subimg_in_context:')
            print(f'img.shape = {img.shape}, subImg.shape = {subImg.shape}, ulLoc = {ulLoc}')

        # Copy the subImage into the correct location of the image
        tempImg[ulLoc[0]:ulLoc[0] + subImgH, ulLoc[1]:ulLoc[1] + subImgW] = subImg

        # Put a white border around the subImg
        lineColor = float(np.amax(tempImg))  # Without conversion, 16-bit value causes problems ...

        tempImg = cv2.rectangle(tempImg,
                                (ulLoc[1], ulLoc[0]),  # Note cv2, so col, row
                                (ulLoc[1] + subImgW, ulLoc[0] + subImgH),  # Note cv2, so col, row
                                lineColor,
                                thickness=4)

        plt.figure(figsize=figsize)
        plt.imshow(tempImg, cmap=cmap)
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        if saveImages:
            plt.savefig(f'{title}.pdf')
        plt.show()
        return True

    else:  # invalid
        print('display_subimg_in_context() invalid input: img must be 2D or 3D image file')
        return False

##

def display_subimg_in_context_with_img(img, subImg, zMin, zMax, ulLoc, figsize=(5, 10),
                                       label='', subImageIdx=None, nSubImg=None,
                                       relLoc=(None, None), absLoc=(None, None),
                                       cmap='gray', colorbar=False,
                                       saveImages=False, verbose=False):
    # Jeff Pelz May 10, 2020
    # Show subimage in image in context - ulLoc is location of upper-left pixel of subimg in img

    # input
    #        img: nparray (2D or 3D)
    #        subImg:  nparray (2D or 3D) - a 'subimage' from img
    #        ulLoc: tuple (R,C) ints defining location of upper-left pixel in img
    #        [optional] figsize:  tuple default=(10,10) - size of figure
    #        [optional] title:    str default='SubImage in Context' - title displayed over figure
    #        [optional] cmap:     str default='gray' - matplotlib colormap (see matplotlib documentation)
    #        [optional] colorbar: bool default=False - display colorbar next to image?

    # output
    #        bool True = successful display, False = failure

    if subImageIdx is None:
        subImageIdx = []
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)  # Set up the two vertical axes
    fig.suptitle(f"{label}")

    if len(img.shape) == 3:  # color image

        # Superimpose subimg into img at location:
        tempImg = img.copy()
        subImgH, subImgW, sumImgD = subImg.shape

        # Copy the subImage into the correct location of the image
        tempImg[ulLoc[0]:ulLoc[0] + subImgH, ulLoc[1]:ulLoc[1] + subImgW, :] = subImg


        # plt.figure(figsize=figsize)
        f1 = ax1.imshow(tempImg)
        titleTxt = f'zscoreSubImages[{subImageIdx} of {nSubImg}] (loc:[{ulLoc[0]}, {ulLoc[1]}])'
        ax1.title(titleTxt)
        ax1.show()
        return True

    elif len(img.shape) == 2:  # monochrome image

        # Superimpose subimg into img at location:
        tempImg = img.copy()
        subImgH, subImgW = subImg.shape

        if verbose:
            print(f'\n= = = = = = = = = = = = = = = = = = = = = = = = =\nIn display_subimg_in_context:')
            print(f'img.shape = {img.shape}, subImg.shape = {subImg.shape}, ulLoc = {ulLoc}')

        # Copy the subImage into the correct location of the image
        tempImg[ulLoc[0]:ulLoc[0] + subImgH, ulLoc[1]:ulLoc[1] + subImgW] = subImg

        # Put a white border around the subImg
        lineColor = float(np.amax(tempImg))  # Without conversion, 16-bit value causes problems ...

        tempImg = cv2.rectangle(tempImg,
                                (ulLoc[1], ulLoc[0]),  # Note cv2, so col, row
                                (ulLoc[1] + subImgW, ulLoc[0] + subImgH),  # Note cv2, so col, row
                                lineColor,
                                thickness=4)

        f1 = ax1.imshow(tempImg, cmap='gray')
        # if colorbar:
        #     ax1.colorbar(fraction=0.046, pad=0.04)
        ax1.title.set_text(f'[{ulLoc[0]}, {ulLoc[1]}]    zscoreSubImages[{subImageIdx} of {nSubImg}]')

        f2 = ax2.imshow(subImg, cmap=cmap)
        if colorbar:
            # cb = plt.colorbar(f2, ax=ax2)
            cb = plt.colorbar(f2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.title.set_text(f'Rel/Abs: {(relLoc[0],relLoc[1])} / {(absLoc[0],absLoc[1])}  zMin/Max: {zMin:0.1f}/{zMax:0.1f}')
        if saveImages:
            plt.savefig(f'{label}_{subImageIdx}.pdf')
        plt.show()
        return True



        # if saveImages:
        #     plt.savefig(f'{title}.pdf')
        # plt.show()
        # return True

    else:  # invalid
        print('display_subimg_in_context() invalid input: img must be 2D or 3D image file')
        return False

##

def display_subimg_in_context_with_img2(img, subImg, zMin, zMax, zExtreme, ulLoc, figsize=(5, 10),
                                       label='', subImageIdx=None, nSubImg=None,
                                       relLoc=(None, None), absLoc=(None, None),
                                       cmap='gray', colorbar=False,
                                       saveImages=False, verbose=False):
    # Jeff Pelz May 10, 2020
    # Show subimage in image in context - ulLoc is location of upper-left pixel of subimg in img

    # input
    #        img: nparray (2D or 3D)
    #        subImg:  nparray (2D or 3D) - a 'subimage' from img
    #        ulLoc: tuple (R,C) ints defining location of upper-left pixel in img
    #        [optional] figsize:  tuple default=(10,10) - size of figure
    #        [optional] title:    str default='SubImage in Context' - title displayed over figure
    #        [optional] cmap:     str default='gray' - matplotlib colormap (see matplotlib documentation)
    #        [optional] colorbar: bool default=False - display colorbar next to image?

    # output
    #        bool True = successful display, False = failure

    if subImageIdx is None:
        subImageIdx = []
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)  # Set up the two vertical axes
    fig.suptitle(f"{label}")

    if len(img.shape) == 3:  # color image

        # Superimpose subimg into img at location:
        tempImg = img.copy()
        subImgH, subImgW, sumImgD = subImg.shape

        # Copy the subImage into the correct location of the image
        tempImg[ulLoc[0]:ulLoc[0] + subImgH, ulLoc[1]:ulLoc[1] + subImgW, :] = subImg


        # plt.figure(figsize=figsize)
        f1 = ax1.imshow(tempImg)
        titleTxt = f'zscoreSubImages[{subImageIdx} of {nSubImg}] (loc:[{ulLoc[0]}, {ulLoc[1]}])'
        ax1.title(titleTxt)
        ax1.show()
        return True

    elif len(img.shape) == 2:  # monochrome image

        # Superimpose subimg into img at location:
        tempImg = img.copy()
        subImgH, subImgW = subImg.shape

        if verbose:
            print(f'\n= = = = = = = = = = = = = = = = = = = = = = = = =\nIn display_subimg_in_context:')
            print(f'img.shape = {img.shape}, subImg.shape = {subImg.shape}, ulLoc = {ulLoc}')

        # Copy the subImage into the correct location of the image
        tempImg[ulLoc[0]:ulLoc[0] + subImgH, ulLoc[1]:ulLoc[1] + subImgW] = subImg

        # Put a white border around the subImg
        lineColor = float(np.amax(tempImg))  # Without conversion, 16-bit value causes problems ...

        tempImg = cv2.rectangle(tempImg,
                                (ulLoc[1], ulLoc[0]),  # Note cv2, so col, row
                                (ulLoc[1] + subImgW, ulLoc[0] + subImgH),  # Note cv2, so col, row
                                lineColor,
                                thickness=4)

        f1 = ax1.imshow(tempImg, cmap='gray')
        # if colorbar:
        #     ax1.colorbar(fraction=0.046, pad=0.04)
        ax1.title.set_text(f'zscoreSubImages[{subImageIdx} of {nSubImg}]   UL=({ulLoc[0]}, {ulLoc[1]})')

        f2 = ax2.imshow(subImg, cmap=cmap)
        if colorbar:
            # cb = plt.colorbar(f2, ax=ax2)
            cb = plt.colorbar(f2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.title.set_text(f'Rel | Abs position: {(relLoc[0],relLoc[1])} | {(absLoc[0],absLoc[1])}\n \
                             \n zMin/Max: {zMin:0.1f}/{zMax:0.1f}    (zExtreme: {zExtreme:0.1f})')
        if saveImages:
            plt.savefig(f'{label}_{subImageIdx}.pdf')
        plt.show()
        return True



        # if saveImages:
        #     plt.savefig(f'{title}.pdf')
        # plt.show()
        # return True

    else:  # invalid
        print('display_subimg_in_context() invalid input: img must be 2D or 3D image file')
        return False

##


def recode_g_as_gr_and_gb(rawColors, RGrBGb=[0, 1, 2, 3], verbose=False):  # default number code for colors

    #  Jeff B. Pelz May 2020

    #  Takes a rawImg.raw_colors array as input. If it is a 3-color (RGB) image instead of
    #  a 4 color (RG1BG2) image, it renames the two G colors Gr and Gb to match
    #  the color of the other filter on the same row.
    #  For example, the filter array | R G R G | is renamed to  | R  Gr  R  Gr | (G on red row)
    #                                | G B G B |                | Gb B   Gb B  | (G on blue row)

    #  Return a list of all NEF images in the current (or specified) directory.  If a different directory is
    #  specified, return to original directory before returning.

    #  input
    #         rawColors  rawImg.raw_colors array
    #         [default]  RGrBGb list of ints default=[0,1,2,3] defining number code for R, Rg, B, & Gb
    #         verbose=False boolean

    #  output
    #         rawColors  rawimg.raw_colors array  rewritten array, with G in B rows renamed as Gb

    # local color names from default number codes:
    Red, Gred, Blue, Gblue = RGrBGb[0], RGrBGb[1], RGrBGb[2], RGrBGb[3]
    Green = Gred  # A single green (G1 == G2) is coded the same as Gred

    if verbose:
        print('\n - - - - - - - -\n In recode_g_as_gr_and_gb: \n')
        print(f'RGrBGb = {RGrBGb}\n')
        print(f'Red   = {Red}')
        print(f'Green = {Green}')
        print(f'Gred  = {Gred}')
        print(f'Gblue = {Gblue}')
        print(f'Blue  = {Blue}')
        print(f'\nBefore recoding: rawColors = \n{rawColors}')

    for idx, row in enumerate(rawColors):  # Step through each row in raw color
        if Blue in row:  # If this is a "Blue Row"
            row[row == Green] = Gblue  # Rename each G pixel in this row to a Gb pixel
            rawColors[idx, :] = row  # copy the renamed row back into the image

    if verbose:
        print(f'\nAfter recoding: rawColors = \n{rawColors}')

    return rawColors


##


def extractOneBayerChannel(rawImage, channelNumber, visible=True, verbose=False):
    # Function accepts as input rawImage representing a Bayer-CFA ** or the filename of a raw image **
    # and the channel number (0-3) and returns the appropriate (1/4) subimage

    if verbose:
        print(f'\n In extractOneBayerChannel: type(rawImage) = {type(rawImage)}')
        print(f'channelNumber = {channelNumber}')
        print(f'visible = {visible}\n')

    if isinstance(rawImage, str):  # Got a string; read in the image
        rawImg = rawpy.imread(rawImage)
        if visible:  # only convert 'visible' portion of image
            rawArr = rawImg.raw_image_visible
        else:
            rawArr = rawImg.raw_image

    elif isinstance(rawImage, rawpy._rawpy.RawPy):  # Got a rawImg; convert to rawArray
        rawImg = rawImage
        if visible:  # only convert 'visible' portion of image
            rawArr = rawImg.raw_image_visible
        else:
            rawArr = rawImg.raw_image

    else:  # ERROR
        rawArr = None
        print(f'\n *** Error in extractOneBayerChannel: ***\nReceived type {type(rawImage)}')
        print(f'\n I can only handle filename or RawPy raw image.')
        return None

    if verbose:
        print(f'\n In extractOneBayerChannel: visible = {visible}')
        print(f'rawArr.shape = {rawArr.shape}\n')

    numColors = rawImg.num_colors

    if numColors == 3:  # G1 == G2, so I have to split them up myself
        rawColors = recode_g_as_gr_and_gb(rawImg.raw_colors.copy(), verbose=verbose)

    elif numColors == 4:  # I can just return the four channels already defined
        pass
    else:  # print error message and return None
        print(f"\n *** Error in extractFourBayerChannels: got {numColors} colors - I can only handle 3 or 4\n")
        return None

    # Extract ONE of the subimages:
    # Create a boolean mask which is True for the target channel, False elsewhere
    chanMask = rawColors == channelNumber  # rawImg.raw_colors==colorChan

    # Determine nrows & ncols of subimage (1/4 size for Bayer array pattern)
    nrows = int(rawArr.shape[0] / 2)
    ncols = int(rawArr.shape[1] / 2)

    try:
        rawChanArr = rawArr[chanMask]  # extract the subpixels at the mask==True locations
    except Exception as err:
        print(f'Error trying to apply mask to rawArr: {err}')
        print(f'rawArr.shape = {rawArr.shape}')
        print(f'chanMask.shape = {chanMask.shape}')
        print(f'rawColors.shape = {rawColors.shape}')
        print(f'nrows, ncols = {nrows}, {ncols}')



    rawChanArr = rawChanArr.reshape(nrows, ncols)  # reshape 1D -> 2D

    if verbose:
        print(f'\n numColors = {numColors}')
        print(f'\n - - - - - - \n channelNumber = {channelNumber}:   chanMask = \n{chanMask}   chanMask.shape = {chanMask.shape}')
        print(f'\n - - - - - - \n rawArr/2: nrows, ncols = {nrows}, {ncols}')
        print(f'\n - - - - - - \n \nrawChanArr = \n{rawChanArr}   \nrawChanArr.shape = {rawChanArr.shape}')
        print(f'\n - - - - - - \n Reshaping rawChanArr from {rawChanArr.shape} to {nrows} x {ncols}')
        print(f'\n - - - - - - \n Completed calculating subArray # {channelNumber}: \n - - - - - - \n ')
        print(f'rawChanArr = \n{rawChanArr} ')

    return rawChanArr


##

def extractFourBayerChannels(rawImage, visible=True, verbose=False):
    # Function accepts as input rawImage representing a Bayer-CFA ** or the filename of a raw image **
    # and returns *ALL 4*  (1/4) (R, Gr, B, Gb) subimages

    if verbose:
        print(f'\n - - - - - - \n In extractFourBayerChannels: type(rawImage) = {type(rawImage)}')

    if isinstance(rawImage, str):  # Got a string; read in the image
        rawImg = rawpy.imread(rawImage)
    elif isinstance(rawImage, rawpy._rawpy.RawPy):  # Got a rawImg; convert to rawArray
        rawImg = rawImage
    else:  # ERROR
        print(f'\n *** Error in extractFourBayerChannels: ***\nReceived type {type(rawImage)}')
        print(f'\n I can only handle filename or RawPy raw image.')
        return None

    if verbose:
        print(f'\nIn extractFourBayerChannels: rawImg.raw_image_visible.shape = {rawImg.raw_image_visible.shape}\n')

    numColors = rawImg.num_colors

    if verbose:
        print(f'\n numColors = {numColors}')

    if numColors == 3:  # G1 == G2, so I have to split them up myself
        rawColors = rawImg.raw_colors.copy()  # make a local copy of the
        rawColors = recode_g_as_gr_and_gb(rawColors, verbose=verbose)

    elif numColors == 4:  # I can just return the four channels already defined
        pass
    else:  # print error message and return None
        print(f"\n *** Error in extractFourBayerChannels: got {numColors} colors - I can only handle 3 or 4\n")
        return None

    # Extract all four subimages:
    extractedSubArrays = []  # initialize to empty list
    for colorChan in range(4):
        rawChanArr = extractOneBayerChannel(rawImg, colorChan, visible=visible, verbose=verbose)
        extractedSubArrays.append(rawChanArr)

    return extractedSubArrays


##

def break_image_into_subimages(img, subImgHW, verbose=False):
    '''
    Divides img (2D ndarray) into subimages of size subImgHW-by-subImgHW (with trailing smaller if necessary)
    Returns list locations (upper-left corners) and list of subimages
    '''

    assert len(img.shape) == 2  # only works on 2D (monochrome) images

    # Get size of image
    imgH = img.shape[0]  # Height of (number of rows in) image
    imgW = img.shape[1]  # Width of (number of columns in) image

    # Calculate number of Rows and Columns to break the image up into:
    nRows = ((imgH - 1) // subImgHW) + 1  # subImages per column depends on number of rows  [nRows]
    nCols = ((imgW - 1) // subImgHW) + 1  # subImages per row depends on number of columns  [nCols]

    # Calculate H & W of padded image to hold all the rows and cols
    paddedImgH = subImgHW * nRows  # Height of padded image
    paddedImgW = subImgHW * nCols  # Width of padded image

    if verbose:
        print(
            f'nRows, nCols = {nRows}, {nCols}   imH, imW = {imgH}, {imgW}   paddedImg, paddedImgW = {paddedImgH}, {paddedImgW}')

    # pad array with NaNs so it can be divided by nRows and nCols
    A = np.nan * np.ones([paddedImgH, paddedImgW])  # Create an array of NaNs the size of the paddedImg
    A[:img.shape[0], :img.shape[1]] = img  # Copy the original img into the NaN array, overwriting the defined pixels

    subImage_list = []  # Initialize empty list of subimages
    ulLocList = []  # Initialize emply list of upper-left location

    for row_subImage in range(nRows):  # Step through each row of subimages
        rowNum = row_subImage * subImgHW  # global row number is subImg row x height of subImg

        colNum = 0  # Initialize column number within each row
        for col_subImage in range(nCols):  # Step through each column of subimages
            colNum = col_subImage * subImgHW  # global col number is subImg col x width of subImg

            if verbose:
                print(f'row_subImage, col_subImage = {row_subImage:3}, {col_subImage:3}', end='')
                print(f'    rowNum, colNum = {rowNum:4}, {colNum:4}')

            subImage = A[rowNum:rowNum + subImgHW, colNum:colNum + subImgHW]  # Create 'raw' subimage (with NaNs)

            # strip nan columns and nan rows
            nan_cols = np.all(np.isnan(subImage), axis=0)  # Find all cols that are all NaNs
            subImage = subImage[:, ~nan_cols]  # keep only those that are NOT all NaNs
            nan_rows = np.all(np.isnan(subImage), axis=1)  # find all rows that are all NaNs
            subImage = subImage[~nan_rows, :]  # keep only those that are NOT all NaNs

            # append remaining subimage to list of subimages, and the location of the upper-left corner to ulLocList
            if subImage.size:
                subImage_list.append(subImage)
                ulLocList.append((rowNum, colNum))

    return ulLocList, subImage_list

##

def find_high_z_score_regions(img, subImageHW=32, zScoreThresh=10.0,
                              label='', displayImages=True, figsize=(10,5), saveImages=False, verbose=False):
    # Take as input an image
    # Break the image up into subImageHW x subImageHW subimages;
    #    calculate the z-score for each pixel in that subimage
    # Report all pixel locations that exceed zScoreThresh in relative (subimage) and absolute (img) coords

    if len(img.shape) > 2:  # We want to deal with one channel at a time
        print(f'I want 1D, grayscale images - but you sent img.shape = {img.shape}.')
        return None
    else:  # Process this image

        timer = [time.time()]  # Initialize timer

        locations, subImages = break_image_into_subimages(img, subImageHW)
        nSubImg = len(subImages)

        if verbose:
            print('\n= = = = = = = = = = = = = = = = = = = =\nIn find_high_z_score_regions')
            print(f'len(subImages) = {len(subImages)}  len(locations) = {len(locations)}')

        timer.append(time.time())

        if verbose:
            print(f'subImg z-scores min -> max')

        zLimit = zScoreThresh  # Only print out if zScore exceeds this limit (+/-)

        zscoreSubImageList = []         # List of subimages
        subImageExceedsZlimitList = []  # List of subimages that exceed the threshold
        meanList = []
        stdList = []
        zMinList = []                   # List of zMin values found in each subimage that exceeded the threshold
        zMaxList = []                   # List of zMax values found in each subimage that exceeded the threshold
        relLocZminList = []                # List of RELATIVE LOCATIONS of zMin values found in each ...
        relLocZmaxList = []                # List of RELATIVE LOCATIONS of zMax values found in each ...
        absLocZminList = []             # List of ABSOLUTE LOCATIONS of zMin values found in each ...
        absLocZmaxList = []             # List of ABSOLUTE LOCATIONS of zMax values found in each ...

        maxAbsZ = -999.9

        for idx, subImage in enumerate(subImages):  # Process each subimage in turn
            zscoreSubImagesTemp = stats.zscore(subImage, axis=None)
            zMin = np.amin(zscoreSubImagesTemp)
            zMax = np.amax(zscoreSubImagesTemp)

            maxAbsZ = np.max((maxAbsZ, zMax, np.abs(zMin)))  # update absolute maximum zMax value

            if zMin < -zLimit or zMax > zLimit:  # Above set limit: Add subimage and values to list

                zscoreSubImageList.append(zscoreSubImagesTemp)  # Only save the subimage if it exceeds threshold
                subImageExceedsZlimitList.append(idx)  # The index of the subimage in the original image

                meanList.append(np.mean(subImage))  # Add the mean to the list of means
                stdList.append(np.std(subImage))    # Add the std to the list of stds

                zMinList.append(zMin)  # Add the min z-score to the list of min z-scores
                zMaxList.append(zMax)  # Add the max z-score to the list of max z-scores

                whereZmin = np.where(zscoreSubImagesTemp == zMin)  # location(s) in array where zMin occurs
                relLocZmin = np.array((whereZmin[0][0], whereZmin[1][0]))  # Take the first (or only) occurrence of the min value
                relLocZminList.append(relLocZmin)  # Add the pixel location of the minimum in this subimage to the list
                absLocZminList.append(relLocZmin + locations[idx])  # Absolute position is relative position + upper-left

                whereZmax = np.where(zscoreSubImagesTemp == zMax)  # location(s) in array where zMax occurs
                relLocZmax = np.array((whereZmax[0][0], whereZmax[1][0]))  # Take the first (or only) occurrence of the max value
                relLocZmaxList.append(relLocZmax)  # Add the pixel location of the maximum in this subimage to the list
                absLocZmaxList.append(relLocZmax + locations[idx])  # Absolute position is relative position + upper-left

                # if verbose:
                #     print('===================================================================')
                #     print(f'Found pixel #{len(meanList)}:  zMin/zMax = {zMin:0.2f}/{zMax:0.2f}')
                #     print(f'locations[{idx}] = {locations[idx]}     locZmin = {relLocZminList[-1]}', end='')
                #     print(f',   Abs location = {absLocZminList[-1]}')
                #     print(f'locations[{idx}] = {locations[idx]}     locZmax = {relLocZmaxList[-1]}', end='')
                #     print(f',   Abs location = {absLocZmaxList[-1]}')
                #     print('===================================================================')

                # Q) Should I have a different zLimitMin and zLimitMax?  zMin indicates regions with little variation

                if verbose:
                    print('===================================================================')
                    print(f'#{len(meanList)}:   (Loc:{idx:6}) {zMin:-5.1f} -> {zMax:4.1f}  [mean = {meanList[-1]:6.0f}  std = {stdList[-1]:6.0f}]')
                    print(f'RelLoc of min ({zMinList[-1]:-5.1f}) @ ({relLocZminList[-1][0]:4},  {relLocZminList[-1][1]:4});', end='')
                    print(f'  of max ({zMaxList[-1]:-5.1f}) @ ({relLocZmaxList[-1][0]:4},  {relLocZminList[-1][1]:4});')
                    print(f'AbsLoc of min ({zMinList[-1]:-5.1f}) @ ({absLocZminList[-1][0]:4},  {absLocZminList[-1][1]:4});', end='')
                    print(f'  of max ({zMaxList[-1]:-5.1f}) @ ({absLocZmaxList[-1][0]:4},  {absLocZminList[-1][1]:4});')

        timer.append(time.time())

        if verbose:
            print(f'It took {timer[-1]-timer[0]:0.2f} seconds for the whole process with {len(subImages):,} subImages')

            print(f'\nmaximum absolute value z-score = {maxAbsZ:0.1f}\n')

        if verbose or displayImages or saveImages:

            print(f'It took {timer[-1] - timer[0]:0.2f} seconds for all calculations. <<<<<<<<<<<<<<<<<<<<<<<<<')

            if len(subImageExceedsZlimitList) > 0:
                for idx, subImageIdx in enumerate(subImageExceedsZlimitList):

                    if verbose:
                        print(f'subImage {subImageIdx} > {zLimit}: ({zMinList[idx]:-5.1f} -> {zMaxList[idx]:4.1f})', end='')
                        print(f"   relLocZminList[idx] = {relLocZminList[idx]}   absLocZminList[idx] = {absLocZminList[idx]}   z={zMaxList[idx]:0.1f}")

                    if displayImages or saveImages:
                        display_subimg_in_context_with_img(img, zscoreSubImageList[idx], zMaxList[idx], locations[subImageIdx],
                                label=label, subImageIdx=subImageIdx, nSubImg=nSubImg,
                                relLoc=relLocZmaxList[idx], absLoc=absLocZmaxList[idx],
                                cmap='coolwarm', colorbar=True, figsize=figsize, saveImages=saveImages, verbose=False)

                        #     title=f'{label} zscoreSubImages[{subImageIdx} of {nSubImg}]\n Upper-left subImg: {locations[subImageIdx]}',

                        # display_img(zscoreSubImageList[idx],
                        #             title=f'{label} zscoreSubImages[{subImageIdx} of {nSubImg}]\n Rel: {relLocZmaxList[idx]} Abs: {absLocZmaxList[idx]}',
                        #             cmap='coolwarm', colorbar=True, saveImages=saveImages)

        resultDict = dict(
            [('means', meanList), ('stds', stdList),
             ('zMins', zMinList),  ('zMaxs', zMaxList),
             ('relLocZmins', relLocZminList),  ('relLocZmaxs', relLocZmaxList),
             ('absLocZmins', absLocZminList),  ('absLocZmaxs', absLocZmaxList),
             ('zscoreSubImages', zscoreSubImageList),
             ('sec', (timer[-1] - timer[0]))])

        return resultDict

##

# After "find_high_z_score_regions()
# Now find both extremes; high and low z scores, for 'hot' and 'cold' pixels ...

def find_extreme_z_score_regions(img, subImageHW=32, highZscoreThresh=10.0, lowZscoreThresh= -3.0,
                              label='', displayImages=True, figsize=(10,5), saveImages=False, verbose=False):
    # Take as input an image
    # Break the image up into subImageHW x subImageHW subimages;
    #    calculate the z-score for each pixel in that subimage
    # Report all pixel locations that exceed lowZscoreThresh or highZscoreThresh in
    # relative (subimage) and absolute (img) coords

    if len(img.shape) > 2:  # We want to deal with one channel at a time
        print(f'I want 1D, grayscale images - but you sent img.shape = {img.shape}.')
        return None
    else:  # Process this image

        timer = [time.time()]  # Initialize timer

        locations, subImages = break_image_into_subimages(img, subImageHW)
        nSubImg = len(subImages)

        if verbose:
            print('\n= = = = = = = = = = = = = = = = = = = =\nIn find_extreme_z_score_regions')
            print(f'len(subImages) = {len(subImages)}  len(locations) = {len(locations)}')

        timer.append(time.time())

        if verbose:
            print(f'subImg z-scores min -> max')

        lowZlimit = lowZscoreThresh  # Only print out if zScore exceeds this limit
        highZlimit = highZscoreThresh  # Only print out if zScore exceeds this limit

        zscoreSubImageList = []         # List of subimages
        subImageExceedsZlimitList = []  # List of subimages that exceed either threshold
        meanList = []
        stdList = []
        zMinList = []                   # List of zMin values found in each subimage that exceeded either threshold
        zMaxList = []                   # List of zMax values found in each subimage that exceeded either threshold
        zExtremeList = []               # List of zMin OR zMax value found responsible for exceeding the threshold
        relLocZminList = []             # List of RELATIVE LOCATIONS of zMin values found in each ...
        relLocZmaxList = []             # List of RELATIVE LOCATIONS of zMax values found in each ...
        absLocZminList = []             # List of ABSOLUTE LOCATIONS of zMin values found in each ...
        absLocZmaxList = []             # List of ABSOLUTE LOCATIONS of zMax values found in each ...

        maxAbsZ = -999.9
        minAbsZ = +999.9

        for idx, subImage in enumerate(subImages):  # Process each subimage in turn
            zscoreSubImagesTemp = stats.zscore(subImage, axis=None)

            zMin = np.amin(zscoreSubImagesTemp)
            zMax = np.amax(zscoreSubImagesTemp)

            maxAbsZ = np.max((maxAbsZ, zMax))  # update absolute maximum zMax value
            minAbsZ = np.min((minAbsZ, zMin))  # update absolute minimum zMin value

            if (zMin < lowZlimit) or (zMax > highZlimit):  # Above either set limit: Add subimage and values to list

                zscoreSubImageList.append(zscoreSubImagesTemp)  # Only save the subimage if it exceeds threshold
                subImageExceedsZlimitList.append(idx)  # The index of the subimage in the original image

                meanList.append(np.mean(subImage))  # Add the mean to the list of means
                stdList.append(np.std(subImage))    # Add the std to the list of stds

                zMinList.append(zMin)  # Add the min z-score to the list of min z-scores
                zMaxList.append(zMax)  # Add the max z-score to the list of max z-scores

                if (zMax > highZlimit):  # If it was the zMax that exceeded the limit:
                    extremeIsMax = True
                    zExtremeList.append(zMax)
                else:  # It was the zMin that exceeded the limit
                    extremeIsMax = False
                    zExtremeList.append(zMin)

                whereZmin = np.where(zscoreSubImagesTemp == zMin)  # location(s) in array where zMin occurs
                relLocZmin = np.array((whereZmin[0][0], whereZmin[1][0]))  # Take the first (or only) occurrence of the min value
                relLocZminList.append(relLocZmin)  # Add the pixel location of the minimum in this subimage to the list
                absLocZminList.append(relLocZmin + locations[idx])  # Absolute position is relative position + upper-left

                whereZmax = np.where(zscoreSubImagesTemp == zMax)  # location(s) in array where zMax occurs
                relLocZmax = np.array((whereZmax[0][0], whereZmax[1][0]))  # Take the first (or only) occurrence of the max value
                relLocZmaxList.append(relLocZmax)  # Add the pixel location of the maximum in this subimage to the list
                absLocZmaxList.append(relLocZmax + locations[idx])  # Absolute position is relative position + upper-left

                if verbose:
                    print('====================================================================================')
                    print(f'#{len(meanList)}: ', end='')
                    print(f'(SubImg:{idx:5}) zMin->zMax:{zMin:-5.1f} -> {zMax:4.1f}   ', end='')
                    print(f'zExtreme={zExtremeList[-1]:-5.1f}   ', end='')
                    print(f'[mean = {meanList[-1]:5.0f}  std = {stdList[-1]:4.0f}]')
                    print(f'Rel | Abs location of Min: ({relLocZminList[-1][0]:4},{relLocZminList[-1][1]:4}) | ', end='')
                    print(f'({absLocZminList[-1][0]:4},{absLocZminList[-1][1]:4})', end='')
                    if extremeIsMax:
                        print()
                    else:
                        print('  Extreme Min')
                    print(f'Rel | Abs location of Max: ({relLocZmaxList[-1][0]:4},{relLocZmaxList[-1][1]:4}) | ', end='')
                    print(f'({absLocZmaxList[-1][0]:4},{absLocZmaxList[-1][1]:4})', end='')
                    if extremeIsMax:
                        print('  Extreme Max')
                    else:
                        print()

        timer.append(time.time())

        if verbose:
            print(f'It took {timer[-1]-timer[0]:0.2f} seconds for the whole process with {len(subImages):,} subImages')

            print(f'\nmaximum absolute value z-score = {maxAbsZ:0.1f}\n')

        if verbose or displayImages or saveImages:

            print(f'It took {timer[-1] - timer[0]:0.2f} seconds for all calculations. <<<<<<<<<<<<<<<<<<<<<<<<<')

            if len(subImageExceedsZlimitList) > 0:
                for idx, subImageIdx in enumerate(subImageExceedsZlimitList):

                    if verbose:
                        print(f'subImage {subImageIdx:4} exceeds {lowZscoreThresh} or {highZscoreThresh}: ', end='')
                        print(f' ({zMinList[idx]:-5.1f} -> {zMaxList[idx]:4.1f})  vs ({minAbsZ:-5.1f} -> {maxAbsZ:4.1f})', end='')
                        print(f"   relLocZminList[idx] = {relLocZminList[idx]}   absLocZminList[idx] = {absLocZminList[idx]}   z={zMaxList[idx]:0.1f}")

                    if displayImages or saveImages:
                        display_subimg_in_context_with_img2(img, zscoreSubImageList[idx],
                                                           zMinList[idx], zMaxList[idx], zExtremeList[idx],
                                                           locations[subImageIdx],
                                                           label=label, subImageIdx=subImageIdx, nSubImg=nSubImg,
                                                           relLoc=relLocZmaxList[idx], absLoc=absLocZmaxList[idx],
                                                           cmap='coolwarm', colorbar=True,
                                                           figsize=figsize, saveImages=saveImages, verbose=False)

        resultDict = dict(
            [('means', meanList), ('stds', stdList),
             ('zMins', zMinList),  ('zMaxs', zMaxList), ('zExtremes', zExtremeList),
             ('relLocZmins', relLocZminList),  ('relLocZmaxs', relLocZmaxList),
             ('absLocZmins', absLocZminList),  ('absLocZmaxs', absLocZmaxList),
             ('zscoreSubImages', zscoreSubImageList),
             ('sec', (timer[-1] - timer[0]))])

        return resultDict


def convolve_img(img, kernel, displayImages=False, figsize=(10,5), label='',
                 saveImages=False, verbose=False):
    # Convolve 1-channel image img by 1-channel kernel,
    # convert img to float
    imgFloat = img.astype(np.float)
    convolvedImg = cv2.filter2D(imgFloat, -1, kernel)

    if displayImages:  # Display original image and kernel:
        display_img(imgFloat, title="input image as float", colorbar=True)  # show the image
        display_img(kernel, title="kernel", colorbar=True)  # show the kernel
        display_img(convolvedImg, title="result", colorbar=True)  # show the result

    return convolvedImg


