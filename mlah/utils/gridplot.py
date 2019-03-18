import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

background_color = 0.5

X = np.ones((11,11))*background_color
X[0,0] = -1.0
X[10,10] = 1.0

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(X, cmap=cm.jet, interpolation='nearest')
plt.ion()
plt.show()

for i in range(10):
	if i == 5:
		X = np.zeros((11,11))
		background_color = 0.0
		X[10,10] = 1.0

	X[i,i] = background_color
	X[i+1,i+1] = -1.0

	plt.imshow(X, cmap=cm.jet, interpolation='nearest')
	plt.draw()
	plt.pause(0.5)
#ax.format_coord = format_coord
#plt.savefig('test')