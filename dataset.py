import numpy as np
import matplotlib.pyplot as plt

# Load the binary data from the t10k-images-idx3-ubyte file
with open('./data/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
    magic_number = int.from_bytes(f.read(4), byteorder='big')
    num_images = int.from_bytes(f.read(4), byteorder='big')
    num_rows = int.from_bytes(f.read(4), byteorder='big')
    num_cols = int.from_bytes(f.read(4  ), byteorder='big')
    images_raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

# Loop over all images and save them as PNG files
print(num_images)
for i in range(10):
    plt.imshow(images_raw[i], cmap='gray')
    plt.savefig(f'./train/image_{i:04}.png')