import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from hough import zero_crossing_detect

def gaussian_kernel(size=3, sigma=1):
    if size%2==0:
        return None
    kernel = np.zeros((size, size), dtype=np.float32)
    const = 1 / (2*np.pi*(sigma**2))

    center = size//2
    bt = 2 * sigma * sigma
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            tp = -1*(x*x + y*y)
            kernel[x, y] = const * np.exp(tp/bt)
    # Normalizing the kernel so sum is 1
    kernel = kernel/kernel.sum()
    print('here')
    return kernel

def convolution(image, filter):
    if filter.shape[0]!=filter.shape[1]:
        return None
    offset = filter.shape[0]//2

    for i in range(offset, image.shape[0]-offset):
        for j in range(offset, image.shape[1]-offset):
            window = image[i-offset:i+offset+1, j-offset:j+offset+1]
            print(i-offset,i+offset, i-offset+filter.shape[0])
            window = window.flatten() * filter.flatten()
            image[i][j] = window.sum()
    return image

def apply_gaussian_smoothing(image):
    filter = gaussian_kernel(size=3,sigma=2)
    img = convolution(image=image, filter=filter)
    return img

def apply_laplacian(image):
    filter = [[0,-1,0],
               [-1,4,-1],
              [0,-1,0]]
    filter = np.array(filter)
    img = convolution(image=image, filter=filter)
    return img

img = Image.open('./house.jpeg')
plt.imshow(img)
img = img.convert('L')
img = np.array(img)
# filter = gaussian_kernel()
# img = convolution(img, filter)
img = apply_gaussian_smoothing(img)
plt.imshow(img)
plt.show()
img = apply_laplacian(img)
img = zero_crossing_detect(img)
plt.imshow(img)
plt.show()
# print(gaussian_kernel())