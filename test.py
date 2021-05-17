
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

values = normal(127.5, 62.5, 10000).astype(int)

print(f'Type of data array is {type(values)}')
print(f'number of values is {values.shape}')
print(f'data type iside array is {type(values[0])}')
print(values)
plt.hist(values, 17, range=(50,200), density=True)
plt.show()
print(f'minimum before trimming is {min(values)}')
print(f'maximum before trimming is {max(values)}')
print((values>255) | (values<0))
values[values<0] = 0
values[values>255] = 255
print(values)
print(f'minimum after trimming: {min(values)}')
print(f'maaximum after trimming: {max(values)}')