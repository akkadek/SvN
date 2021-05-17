## CRDIR_III_proj_1.py
# Jeff B. Pelz  CIS RIT

# import os
import time
import numpy as np
# import rawpy
import matplotlib.pyplot as plt
import matplotlib as mpl
import CRDIR_III_funcs as C3f   # import CRDIR-III helper functions

TITLE_SIZE=20
LABEL_SIZE=18
AXIS_SIZE=16

# Set matplotlib default parameters
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.figsize'] = [15, 15]

# Define colors and hatch patters for R, Gr, B, & Gb channel plots:
clrs=['r', [0.65,0.9,0], 'b', [0,0.9,0.65]]
htchs=['', '\\', '', '/']


# plt.figure(figsize=(15, 15))
# plt.tick_params(axis='x', labelsize=20)
# plt.title('Bayer Image - linear', fontsize=TITLE_SIZE)
# plt.ylabel('Frequency', fontsize=LABEL_SIZE)
# plt.xlabel('digital count', fontsize=LABEL_SIZE)


# from scipy import stats
# from skimage import transform

##
# [optionally specify a directory and] read all .nef images

# imDir = '/Users/pelz/PycharmProjects/nef_images/TERRESTRIAL_SAMPLES_WHITE_SAMPLE_PIX_FOR_RIT/'
imDir = ''

if imDir == '':
    allNefImNames = C3f.list_nef_images()
else:
    allNefImNames = C3f.list_nef_images(dir=imDir)

# Print the list of nef files found:
print(f'The {len(allNefImNames)} .nef files are:')
for idx, nefImName in enumerate(allNefImNames):
    print(f'{idx:4}:  {nefImName}')

##

# Read the first .nef file found:

verbose=False

imNum = 0

t1b = time.time()
rawImg = C3f.read_raw_image(allNefImNames[imNum], dir=imDir, verbose=True)
t2 = time.time()
print(f'It took {t2-t1b:0.3f} sec to read the raw image ...')

if verbose:
    print(f' > > > > > > > > > type(rawImg) = {type(rawImg)}')
    for key, value in rawImg.items():
        print(key)
        print(value)

    print(f" + + + + + + + + + + + + + rawImg['rgbImg'] = {rawImg['rgbImg']}")
    print(f" + + + + + + + + + + + + + rawImg['rgbImg'].shape = {rawImg['rgbImg'].shape}")
    print(f" + + + + + + + + + + + + + rawImg['rgbImg'].shape = {rawImg['rgbImg'].shape}")
    print(f" + + + + + + + + + + + + + rawImg['rgbImg'].shape = {rawImg['rgbImg'].shape}")

# print(rawImg['rawType'])
# print(f"rawRet['rawVisArray'] = \n{rawImg['rawVisArray']}")

# Extract image information from the exif data header:
keyWordList = ['Model', 'ISOSpeed', 'ExposureTime', 'FNumber', 'Pixel', 'EXIF DateTimeOriginal']
t3 = time.time()
EXIFtags = C3f.EXIF_data_keyword_list(imDir+allNefImNames[imNum], keyWordList)
# EXIFtags = C3f.EXIF_data_keyword_list(imDir+allNefImNames[imNum], 'bits')
t4 = time.time()
print(f'It took {t4-t3:0.3f} sec to extract the EXIF data ...')

print('\n===============================\n')
for key in EXIFtags.keys():
    print(f'{key}: {EXIFtags[key]}')
print('\n===============================\n')

# EXIFtags = C3f.EXIF_data_keyword_list(imDir+allNefImNames[imNum])  # Sending without a list returns everything ...
#
# print('\n===============================\n')
# for key in EXIFtags.keys():
#     print(f'{key}: {EXIFtags[key]}')
# print('\n===============================\n')

# allTags = C3f.EXIF_data_keyword(imDir+allNefImNames[imNum])
#
# print('\n===============================\n')
# for key in allTags.keys():
#     print(f'{key}: {allTags[key]}')
# print('\n===============================\n')

# print('Sending no list ...')
# ALLtags = C3f.EXIF_data_keyword_list(imDir+allNefImNames[imNum])
#
# print('\n===============================\n')
# for key in allTags.keys():
#     print(f'{key}: {allTags[key]}')
# print('\n===============================\n')


##

bayerImg = rawImg['rawVisArr']  # Raw Bayer filter image from sensor


C3f.display_img(bayerImg, title='Raw Bayer Display', figsize=(15,10), colorbar=True)
print(f"Min - Max pixel value: (np.amin(bayerImg) - (np.amax(bayerImg) = {np.amin(bayerImg)} - {np.amax(bayerImg)}")

bayerImgMin = np.amin(bayerImg)
bayerImgMax = np.amax(bayerImg)
print(f"Min - Max pixel value: (bayerImgMin, bayerImgMax)  = {bayerImgMin}, {bayerImgMax}")

bayerImgMaxLocs = np.where(bayerImg==bayerImgMax)

bayerImgBigLocs = np.where(bayerImg > bayerImgMax/2)

print(f'location(s) of {bayerImgMax}: {bayerImgMaxLocs}')

plt.plot(bayerImgBigLocs[1], 2844-bayerImgBigLocs[0], 'k+')
plt.plot(bayerImgMaxLocs[1], 2844-bayerImgMaxLocs[0], 'r*')
plt.show()

# exit(0)

# bayerSubImg = bayerImg[1885:2885, 0:100]
# C3f.display_img(bayerSubImg, title='Bayer subImg Display', figsize=(15,10), colorbar=True)
# print(f"Min - Max pixel value: (np.amin(subImg) - (np.amax(subImg) = {np.amin(bayerSubImg)} - {np.amax(bayerSubImg)}")

# Examine histograms of images:

plt.figure(figsize=(15, 15))

ax=plt.subplot(211)
# plt.tick_params(axis='x', labelsize=20)
plt.hist(bayerImg.ravel(), bins=128, log=False, color='k')  # range=(0,4095),
ax.text(0.5, 0.94, 'Bayer Image - linear', horizontalalignment='center', transform=ax.transAxes)
plt.ylabel('Frequency')
plt.xlabel('digital count')

ax=plt.subplot(212)
plt.tick_params(axis='x', labelsize=20)
plt.hist(bayerImg.ravel(), bins=128, log=True, color='k')  # range=(0,4095),
ax.text(0.5, 0.94, 'Bayer Image [log frequency]', horizontalalignment='center', transform=ax.transAxes)
plt.ylabel('log Frequency')
plt.xlabel('digital count')

plt.show()


kernel = np.array([( 0.0, -1.0,  0.0),
                   (-1.0,  4.0, -1.0),
                   ( 0.0, -1.0,  0.0)])

print(f'kernel = \n{kernel}')

edgeImg = C3f.convolve_img(bayerImg, kernel)
# C3f.display_img(edgeImg, title='edgeImg Display', figsize=(15,10), colorbar=True)
print(f"Min - Max pixel value: (np.amin(edgeImg) - (np.amax(edgeImg) = {np.amin(edgeImg)} - {np.amax(edgeImg)}")

# edgeSubImg = C3f.convolve_img(bayerSubImg, kernel)
# C3f.display_img(edgeSubImg, title='edgeSubImg Display', figsize=(15,10), colorbar=True)

# print(f"Min - Max pixel value: (np.amin(edgeSubImg) - (np.amax(edgeSubImg) = {np.amin(edgeSubImg)} - {np.amax(edgeSubImg)}")


# Examine histograms of EDGE images:
plt.figure(figsize=(15, 15))
ax=plt.subplot(211)
plt.hist(edgeImg.ravel(), bins=1024, range=(np.amin(edgeImg), np.amax(edgeImg)), log=False, color='k')
ax.text(0.5, 0.94, 'Bayer EDGE Image - linear', horizontalalignment='center', transform=ax.transAxes)
plt.ylabel('Frequency')
plt.xlabel('digital count')

ax=plt.subplot(212)
plt.hist(edgeImg.ravel(), bins=1024, range=(np.amin(edgeImg), np.amax(edgeImg)), log=True, color='k')
ax.text(0.5, 0.94, 'Bayer EDGE Image [log frequency]', horizontalalignment='center', transform=ax.transAxes)
plt.ylabel('log Frequency')
plt.xlabel('digital count')
plt.show()


# Next: 2 options to reduce the variation in edge values:
#     1) Repeat within each channel to reduce the inter-channel variation
#     2) Adjust the individual channels on a semi-local scale to reduce the inter-channel variation

#  Advantage of 1: simpler  Advantage of 2: maintain nearby pixel-pixel comparisons on chip.

# Extract four channels
col = ['R', 'Gr', 'B', 'Gb']
rawChans = []

for chanNum in range(4):
    print(f'Extracting channel {chanNum} ({col[chanNum]})...')
    # rawChans.append(C3f.extractOneBayerChannel(rawImg['rawImg'], chanNum, visible=True, verbose=False))
    rawChans.append(C3f.extractOneBayerChannel(rawImg['rawImg'], chanNum, visible=False, verbose=False))

print('Done.')

# Look at histograms of color channels separately:
for chanNum, channel in enumerate(rawChans):  # For each of the 4 channels:
    print(f'Channel #{chanNum} ({col[chanNum]}):')

    # C3f.display_img(bayerSubImg, title='Bayer subImg Display', figsize=(15, 10), colorbar=True)
    print(
        f"Min - Max pixel value: (np.amin(channel) - (np.amax(channel) = {np.amin(channel)} - {np.amax(channel)}")

    # Examine histograms of images:

    plt.figure(figsize=(15,15))
    ax=plt.subplot(211)
    plt.hist(channel.ravel(), bins=128, range=(np.amin(channel), np.amax(channel)),
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7, log=False)
    ax.text(0.5, 0.94, f'{col[chanNum]} Bayer Image - linear', horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel('Frequency')
    plt.xlabel('digital count')
    plt.legend()

    ax=plt.subplot(212)
    plt.hist(channel.ravel(), bins=256, range=(np.amin(channel), np.amax(channel)),
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7, log=True)
    ax.text(0.5, 0.94, f'{col[chanNum]} Bayer Image [log frequency]', horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel('log Frequency')
    plt.xlabel('digital count')
    plt.legend()
    plt.show()

# Look at histograms of color channels ON SAME PLOT:
plt.figure(figsize=(15,15))  # Set up subplots for overlapping histograms:

for chanNum, channel in enumerate(rawChans):  # For each of the 4 channels:
    print(f'Channel #{chanNum} ({col[chanNum]}):')

    # C3f.display_img(bayerSubImg, title='Bayer subImg Display', figsize=(15, 10), colorbar=True)
    print(
        f"Min - Max pixel value: (np.amin(channel) - (np.amax(channel) = {np.amin(channel)} - {np.amax(channel)}")

    # Examine histograms of images:
    # plt.figure(figsize=(15,15))
    ax=plt.subplot(211)
    plt.hist(channel.ravel(), bins=128, range=(np.amin(channel), np.amax(channel)), log=False,
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7)
    ax.text(0.5, 0.94, 'Bayer Image Channels - linear', horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel('Frequency')
    plt.xlabel('digital count')
    if chanNum == 3:
        plt.legend()

    ax=plt.subplot(212)
    plt.hist(channel.ravel(), bins=256, range=(np.amin(channel), np.amax(channel)), log=True,
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7)
    ax.text(.5, .94, 'Bayer Image Channels [log frequency]', horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel('log Frequency')
    plt.xlabel('digital count')

    if chanNum == 3:
        plt.legend()
        plt.show()

# Look at histograms of color-channel edge images separately:
for chanNum, channel in enumerate(rawChans):  # For each of the 4 channels:
    print(f'Channel #{chanNum} ({col[chanNum]}):')

    edgeChanImg = C3f.convolve_img(channel, kernel)

    print(f"{col[chanNum]}: Min - Max pixel value: (np.amin(edgeChanImg) - (np.amax(edgeChanImg) = {np.amin(edgeChanImg)} - {np.amax(edgeChanImg)}")

    # Examine histograms of EDGE images:
    plt.figure(figsize=(15, 15))
    ax=plt.subplot(211)
    plt.hist(edgeChanImg.ravel(), bins=128, range=(np.amin(edgeChanImg), np.amax(edgeChanImg)), log=False,
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7)
    ax.text(0.5, 0.94, f'{col[chanNum]} EDGE Image - linear', horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel('Frequency')
    plt.xlabel('digital count')

    ax=plt.subplot(212)
    plt.hist(edgeChanImg.ravel(), bins=128, range=(np.amin(edgeChanImg), np.amax(edgeChanImg)), log=True,
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7)
    ax.text(0.5, 0.94, f'{col[chanNum]} EDGE Image [log frequency]', horizontalalignment='center', transform=ax.transAxes)
    # plt.title(f'Channel {col[chanNum]} EDGE Image [log frequency]')
    plt.ylabel('log Frequency')
    plt.xlabel('digital count')
    plt.show()

# Look at histograms of color channel EDGE images ON SAME PLOT:
plt.figure(figsize=(15,15))  # Set up subplots for overlapping histograms:

for chanNum, channel in enumerate(rawChans):  # For each of the 4 channels:
    print(f'Channel #{chanNum} ({col[chanNum]}):')

    print(
        f"Min - Max pixel value: (np.amin(channel) - (np.amax(channel) = {np.amin(channel)} - {np.amax(channel)}")

    edgeChanImg = C3f.convolve_img(channel, kernel)

    ax=plt.subplot(211)
    plt.hist(edgeChanImg.ravel(), bins=128, range=(np.amin(edgeChanImg), np.amax(edgeChanImg)), log=False,
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7)
    ax.text(0.5, 0.94, 'EDGE Image Channels - linear', horizontalalignment='center', transform=ax.transAxes)
    # plt.title('EDGE Image Channels - linear')
    plt.ylabel('Frequency')
    plt.xlabel('digital count')
    if chanNum == 3:
        plt.legend()

    ax=plt.subplot(212)
    plt.hist(edgeChanImg.ravel(), bins=256, range=(np.amin(edgeChanImg), np.amax(edgeChanImg)), log=True,
             label=f'{col[chanNum]}', color=clrs[chanNum], hatch=htchs[chanNum], alpha=0.7)
    ax.text(0.5, 0.94, 'EDGE Image Channels [log frequency]', horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel('log Frequency')
    plt.xlabel('digital count')

    if chanNum == 3:
        plt.legend()
        plt.show()


quit(0)

























##
tic = time.time()
rawChans = []

for chanNum in range(4):
    rawChans.append(C3f.extractOneBayerChannel(rawImg['rawImg'], chanNum, visible=True, verbose=False))
    print(f'It took {time.time()-tic:0.3f} seconds to extract {chanNum+1} channels')

# ##
# tic = time.time()
# rawChans = []
#
# for chanNum in range(4):
#     rawChans.append(C3f.extractOneBayerChannel(rawImg['rawImg'], chanNum))
#     C3f.display_img(rawChans[chanNum], figsize=(4,4), title=f'rawChans[{chanNum}]')
#     print(f'It took {time.time()-tic:0.3f} seconds to extract AND DISPLAY {chanNum+1} channels')
#

# ##
#
# tic = time.time()
#
# rawChans2 = C3f.extractFourBayerChannels(rawImg['rawImg'])
# print(f'It took {time.time() - tic:0.3f} seconds to extract all 4 channels using extractFourBayerChannels')

##

# tic = time.time()
# ulLocList, subImage_list = C3f.break_image_into_subimages(rawChans[1], 64, verbose=False)
# print(f'It took {time.time() - tic:0.3f} seconds to break rawChans[1] into 64x64 subimages')

##

# results = C3f.find_high_z_score_regions(rawChans[0], subImageHW=32, zScoreThresh=8, displayImages=True, verbose=True)
# results = C3f.find_high_z_score_regions(rawChans[1], subImageHW=32, zScoreThresh=8, displayImages=True, verbose=True)
# results = C3f.find_high_z_score_regions(rawChans[2], subImageHW=32, zScoreThresh=8, displayImages=True, verbose=True)
# results = C3f.find_high_z_score_regions(rawChans[3], subImageHW=32, zScoreThresh=8, displayImages=True, verbose=True)


# resultDict = dict(
#[('means', means), ('stds', stds), ('zscoreSubImages', zscoreSubImages), ('sec', (timer[-1] - timer[0]))])

col = ['R', 'Gr', 'B', 'Gb']

# for idx, rawChan in enumerate(rawChans):
#     print(f'looping: idx = {idx}')
#     results = C3f.find_high_z_score_regions(rawChan, subImageHW=32, zScoreThresh=7, figsize=(5,9),
#                                             label=f'{allNefImNames[imNum]} [{col[idx]}]',
#                                             displayImages=True, saveImages=True, verbose=False)

for idx, rawChan in enumerate(rawChans):
    print(f'looping: idx = {idx}')
    results = C3f.find_extreme_z_score_regions(rawChan, subImageHW=64, lowZscoreThresh= -5, highZscoreThresh= 10,
                                               figsize=(5,9),
                                                label=f'{allNefImNames[imNum]} [{col[idx]}]',
                                                displayImages=True, saveImages=True, verbose=True)

# Q) what needs to change to look for extreme z scores instead of only high z scores???