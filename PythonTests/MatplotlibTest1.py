import matplotlib.pyplot as plt
import numpy as np


#multiple figures and axes
 
#def f(t):
#    return np.exp(-t) * np.cos(2*np.pi*t)

#t1 = np.arange(0.0, 5.0, 0.1)
#t2 = np.arange(0.0, 5.0, 0.02)

#plt.figure(1)
#plt.subplot(211)
#plt.plot(t1, f(t1), 'bo', t2, f(t2), 'm')

#plt.subplot(212)
#plt.plot(t2, np.cos(2*np.pi*t2), 'y-.')
#plt.show()

#hist

#gaussian_numbers = np.random.normal(size=1000)
#plt.hist(gaussian_numbers,30,color='c')
#plt.title("Gaussian Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()


gaussian_numbers = np.random.normal(size=1000)
uniform_numbers = np.random.uniform(low=-3, high=3, size=1000)
plt.hist(gaussian_numbers, bins=20, histtype='stepfilled', normed=True, color='g', label='Gaussian')
plt.hist(uniform_numbers, bins=20, histtype='stepfilled', normed=True, color='b', alpha=0.5, label='Uniform')
plt.title("Gaussian/Uniform Histogram")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.legend()
plt.show()



