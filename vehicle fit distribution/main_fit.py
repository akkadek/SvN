# %%

# import jupyter
# import re

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
#from scipy.stats import *
from sklearn.preprocessing import StandardScaler
import sys

# import warnings
# warnings.filterwarnings('ignore')
image_count = 0

# %%

# fname = 'vehicles_columns.csv'
#reading the datasheet of vehicles
fname = 'vehicles_columns_100K.csv'  # truncated version of csv file
print(f'Reading {fname} ...')
tic = time.time()
df = pd.read_csv(fname)
timeReadCSV = time.time()-tic

print(f'Done reading {fname} ({timeReadCSV:0.2f} sec)')


# %%

# print(f'df.head = {df.head}')
# df.head()

# %%

# print(f'df.shape = {df.shape}')


# %%

# df.dtypes

# %%

# df.columns

# %%

columns = ['id', 'price', 'year', 'manufacturer', 'condition', 'odometer']

# %%

df = df[columns]

# print('writing csv')
# df.to_csv('vehicles_columns.csv')
# print('done writing csv')

# %%

df['year'] = df['year'].apply(lambda x: str(int(x)) if x > 0 else x)

# %%

# print(f'df.head = {df.head(10)}')
# df.head(10)

# %%

# print(f'df.describe = {df.describe}')
# df.describe()

# %%

print(f'Number of NaNs = {df.isna().sum()}')
# df.isna().sum()

# %%

print("df['price'].hist()")
df['price'].hist()


# %%

def standarise(column, pct, pct_lower):
    sc = StandardScaler() #subtracts mean and scales to units of variance (zscore)
    #print(f' type = {type(column)}')
    y = df[column][df[column].notnull()].to_list() #throw away all null places
    y.sort()
    #print(f' y (sorted list) = {y}')
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)] #trimming y
    len_y = len(y)
    yy = ([[x] for x in y]) #making a list of all values in y
    #print(f'yy is {yy}')
    sc.fit(yy)
    y_standard = sc.transform(yy) #standardizes the data
    #print(f'y_standard[1000:100000:1000] before = {y_standard[1000:100000:1000]}')
    y_standard = y_standard.flatten() #flattens the calculated standardized data
    #print(f'y_standard[1000:100000:1000] after = {y_standard[1000:100000:1000]}')
    return y_standard, len_y, y #returns the standardized data, the length of that data, the data


# %%

def fit_distribution(column, pct, pct_lower):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    y_standard, size, y_organized = standarise(column, pct, pct_lower)
    dist_names = ['weibull_min', 'norm', 'weibull_max', 'beta',
                  'invgauss', 'uniform', 'gamma', 'expon', 'lognorm', 'pearson3', 'triang']

    chi_square_statistics = [] #to fill as the loop progresses
    # 11 bins
    percentile_bins = np.linspace(0, 100, 11) #returns the intervals for 11 evenly spaced bins
    percentile_cutoffs = np.percentile(y_standard, percentile_bins) #returns the percentiles of the y_standard
    observed_frequency, bins = (np.histogram(y_standard, bins=percentile_cutoffs))
    #returns the values of the histogram and the bin edges
    cum_observed_frequency = np.cumsum(observed_frequency) #returns the cumulative sum of data y_standard


    # Loop through candidate distributions

    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_standard)
        print("{}\n{}\n".format(dist, param))

        # Get expected counts in percentile bins
        # cdf of fitted distribution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = round(sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency), 0)
        chi_square_statistics.append(ss)

    # Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)

    print('\nDistributions listed by Betterment of fit:')
    print('............................................')
    print(results)


# %%

print('Calling fit_distribution ...')
fit_distribution('price', 0.99, 0.01)
print('Back from fit_distribution ...')

# %%

print('Calling standarise(price....) ...')
y_std, len_y, y = standarise('price', 0.99, 0.01)
print('Back from standarise(price....) ...')
print(f'length is {len_y} \n np.amax is {np.amax(y)}')


# %%

n, bins, patches = plt.hist(y, bins=64)
print(f'n = {n} \n bins = {bins} ')
plt.xlabel('Price')
plt.ylabel('Frequency')
image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
plt.show()

# %%

# f = plt.figure()
# plt.subplot(y,expon.pdf(y_std,-1.19, 1.19))
# plt.subplot(y,invgauss.pdf(y_std,0.45, -1.64, 3.61))
# f.show()


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(y)
axes[0].set_xlabel('Price\n\nHistogram plot of Oberseved Data')
axes[0].set_ylabel('Frequency')
axes[1].plot(y, scipy.stats.expon.pdf(y_std, -1.19, 1.19))
axes[1].set_xlabel('Price\n\nExponential Distribution')
axes[1].set_ylabel('pdf')
axes[2].plot(y, scipy.stats.invgauss.pdf(y_std, 0.45, -1.64, 3.61))
axes[2].set_xlabel('Price\n\nInverse-Gaussian Distribution')
axes[2].set_ylabel('pdf')

fig.tight_layout()
# %%

f = plt.figure()
plt.plot(y, scipy.stats.expon.pdf(y_std, -1.19, 1.19))
image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
f.show()


# %%

f = plt.figure()
plt.plot(y, scipy.stats.invgauss.pdf(y_std, 0.45, -1.64, 3.61))
image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
f.show()

# %%
#this is the graph for the dot plot, with inverse gaussian distribution and exponential distribution plotted on it
data_points = scipy.stats.expon.rvs(-1.19, 1.19, size=2000)
data_points2 = scipy.stats.invgauss.rvs(0.45, -1.64, 3.61, size=2000)

f, ax = plt.subplots(figsize=(8, 8))
ax.plot([-2, 8], [-2, 8], ls="--", c=".3")

percentile_bins = np.linspace(0, 100, 51)
percentile_cutoffs1 = np.percentile(y_std, percentile_bins)
percentile_cutoffs_expon = np.percentile(data_points, percentile_bins)

percentile_cutoffs_invgauss = np.percentile(data_points2, percentile_bins)

#

ax.scatter(percentile_cutoffs1, percentile_cutoffs_invgauss, c='r', label='Inverse-Gaussian Distribution', s=40)
ax.scatter(percentile_cutoffs1, percentile_cutoffs_expon, c='b', label='Exponential Distribution', s=40)

ax.set_xlabel('Theoretical cumulative distribution')
ax.set_ylabel('Observed cumulative distribution')
ax.legend()

image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
plt.show()

# %%

print('Calling fit_distribution(odometer,...) ...')
fit_distribution('odometer', 0.99, 0.01)
print('Back from fit_distribution(odometer,...) ...')

# %%

lst, len_lst, org_lst = standarise('odometer', 0.99, 0.01)

# %%

org_lst[:20]

# %%

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(org_lst)
axes[0].set_xlabel('Odometer (Distance)\n\nHistogram plot of Oberseved Data')
axes[0].set_ylabel('Frequency')
axes[1].plot(org_lst, scipy.stats.beta.pdf(lst, 1.51, 2.94, -1.71, 5.02))
axes[1].set_xlabel('Odometer (Distance)\n\nBeta Distribution')
axes[1].set_ylabel('pdf')
axes[2].plot(org_lst, scipy.stats.triang.pdf(lst, .12, -1.79, 4.90))
axes[2].set_xlabel('Odometer (Distance)\n\nTriangular Distribution')
axes[2].set_ylabel('pdf')
fig.tight_layout()

# %%

plt.hist(org_lst)

# %%

f = plt.figure()
plt.plot(org_lst, scipy.stats.beta.pdf(lst, 1.5111609633771699, 2.9428574390909983, -1.712121634564888, 5.022935095746597))
image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
f.show()

# %%

f = plt.figure()
plt.plot(org_lst, scipy.stats.triang.pdf(lst, .12388009897125515, -1.7967712548899337, 4.908020533304843))
image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
f.show()



data_points = scipy.stats.beta.rvs(1.51, 2.94, -1.71, 5.02, size=1000)

data_points2 = scipy.stats.triang.rvs(.12, -1.79, 4.90, size=1000)

f, ax = plt.subplots(figsize=(8, 8))
ax.plot([-2, 3], [-2, 3], ls="--", c=".3")

percentile_bins = np.linspace(0, 100, 101)
percentile_cutoffs1 = np.percentile(lst, percentile_bins)
percentile_cutoffs_beta = np.percentile(data_points, percentile_bins)

percentile_cutoffs_triang = np.percentile(data_points2, percentile_bins)
# print(percentile_cutoffs1,percentile_cutoffs2)

ax.scatter(percentile_cutoffs1, percentile_cutoffs_beta, c='b', label='Beta Distribution')
ax.scatter(percentile_cutoffs1, percentile_cutoffs_triang, c='r', label='Triangular Distribution', s=40)

ax.set_xlabel('Theoretical cumulative distribution')
ax.set_ylabel('Observed cumulative distribution')
ax.legend()

image_count = image_count + 1
plt.savefig('mainfit/' + f'{image_count}' + '.png')
print(image_count)
plt.show()
print('end of program')

sys.exit()