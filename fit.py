import os
import rawpy
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math
import scipy.stats as stats
from scipy.optimize import curve_fit
import seaborn as sns



def standarise(column, pct, pct_lower):
    sc = StandardScaler() #subtracts mean and scales to units of variance (zscore)
    y = df[column][df[column].notnull()].to_list() #throw away all null places
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)] #trimming y
    len_y = len(y)
    yy = ([[x] for x in y]) #making a list of all values in y
    sc.fit(yy) #computes the mean and std of yy
    y_standard = sc.transform(yy) #standardizes the data
    y_standard = y_standard.flatten() #flattens the calculated standardized data
    return y_standard, len_y, y #returns the standardized data, the length of that data, the data

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








'''
#fitting to a gamma dist from stack overflow
#generating gamma data
#np.random.seed(seed=1)
alpha = 2
loc = 0
beta = 20
data = stats.gamma.rvs(alpha, loc=loc, scale=beta, size=10000)
#data[-1], data[-3], data[0] = 0, 0, 25
#data = data - 100
print(data)
print(f' min is {min(data)}')

#fitting data to gamma distribution
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(data)
print(fit_alpha, fit_loc, fit_beta)

print(alpha, loc, beta)

#plt.hist(data, bins=17, density=True, label='Data')
x = np.linspace (0, 10, 1200)
y1 = stats.gamma.pdf(x, a=0.14, scale=(23.84))
plt.plot(x, y1, "y-", label=(f'alpha = {0.14}, \nbeta = {23.84}'))
plt.legend()
plt.xlim([0, 10])
plt.show()
'''


