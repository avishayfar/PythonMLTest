import matplotlib.pyplot as plt
import numpy as np

#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()

#plt.plot([1,2,3,4],[1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()

#plt.plot([1,2,3,4],[1,4,9,16])
#plt.ylabel('some numbers')
#plt.show()

#npList = np.arange(10)
#plt.plot(npList)
#plt.ylabel('npList')
#plt.show()

#only points without the lines between them

#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt.axis([0, 6, 0, 20])
#plt.show()

#plot(x, y)        # plot x and y using default line style and color
#plot(x, y, 'bo')  # plot x and y using blue circle markers
#plot(y)           # plot y using x as index array 0..N-1
#plot(y, 'r+')     # ditto, but with red plusses

#npList = np.arange(10)
#plt.plot(npList, npList   , 'o-', 
#         npList, npList**2, 'b*',
#         npList, npList**3, 'gh-')
#plt.ylabel('npList')
#plt.show()

#npList = np.arange(20)
#plt.plot(npList, npList**2   , color='cyan', linestyle='dotted', marker='o',
#     markerfacecolor='blue', markersize=12)
#plt.show()

#two grahps at  the same location
npList = np.arange(20)
lines = plt.plot(npList, npList, npList, npList**2)
plt.setp(lines,color='cyan', linestyle='dotted', marker='o',markerfacecolor='blue', markersize=12)
plt.show()





