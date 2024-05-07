import numpy as np

array = np.load('IMG_9939.npy')
print('Loaded array of size', array.shape)
print('The pens, from top to bottom, are red, green and blue')

# Extract Y channel
Y_channel = array[::2, ::2]  # rows and columns step by 2, starting at index 0
Y_channel_2 = array[1::2, 1::2]  # for the second Y in each block

# Combine both parts of Y
Y_channel = np.where(np.indices(Y_channel.shape)[0] % 2 == 0, Y_channel, Y_channel_2)

# Extract C1 channel
C1_channel = array[::2, 1::2]  # top right of each 2x2 block

# Extract C2 channel
C2_channel = array[1::2, ::2]  # bottom left of each 2x2 block

print("Y Channel:\n", Y_channel)
print("C1 Channel:\n", C1_channel)
print("C2 Channel:\n", C2_channel)
