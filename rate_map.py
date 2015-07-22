from recurrent_data import formatData
import matplotlib.pyplot as plt

trX, tvX, teX, trY, tvY, teY = formatData()

print(trY.shape)

plt.scatter(trY[0,:,0],trY[0,:,1])
plt.show()