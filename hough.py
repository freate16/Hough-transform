from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, sobel, laplace
import math
import numpy as np
from matplotlib import pyplot as plt

def preprocess(image_array):
    '''
    Here sobel filter has been applied to the image to get it's edges.
    To get the edges first the sobel along x is applied then sobel along y is applied.
    Then to get the final image the magnitude of the two components is calculated. This
    is now used as the final image.
    '''
    image_array = gaussian_filter(image_array, sigma=2)
    sobel_x = sobel(image_array, axis=0)
    sobel_y = sobel(image_array, axis=1)
    sq_sum = np.square(sobel_x) + np.square(sobel_y)
    edges = np.sqrt(sq_sum)
    print(np.max(edges))
    edges[edges<14]=0
    # edges = np.where(edges>np.mean(edges), 255, 0)
    return edges

def zero_crossing_detect(image_array):
    '''
    Implemented zero crossing edge detection to acquire the edges for the image.
    The Sobel filter was giving only the gradient, but this method is giving results as it
    is monitoring only those places where the gradient changes from positive to negative.
    '''
    image_array = image_array.astype(np.float64)
    edge_image = np.zeros(image_array.shape)
    edge_image = edge_image.astype(np.float64)
    image_array = gaussian_filter(image_array,sigma=6)
    laplacian_image = laplace(image_array)
    # print(laplacian_image)
    print(np.min(laplacian_image), np.max(laplacian_image))
    # Here the zero crossing is being detected on the image.
    for i in range(1,laplacian_image.shape[0]-1):
        for j in range(1,laplacian_image.shape[1]-1):
            patch = laplacian_image[i-1:i+2, j-1:j+2]
            min_val = np.min(patch)
            max_val = np.max(patch)
            if min_val<0<max_val:
                edge_image[i,j]=255
            else:
                edge_image[i,j]=0
    return edge_image

def hough_transform_polar(img):
    img = img.convert('L')
    # img = img.resize((400,400),1)
    img = np.array(img)
    plt.imshow(img)
    plt.show()
    rows,cols = img.shape
    img = zero_crossing_detect(img)
    plt.imshow(img)
    plt.show()
    max_dist = int(math.hypot(rows,cols))
    theta_range = np.deg2rad(np.arange(-90,90))
    accumulator = np.zeros(shape=(2*max_dist,len(theta_range)))
    y_idx, x_idx = np.nonzero(img)
    for i in range(len(x_idx)):
        print('Percentage Done :',i*100/len(x_idx))
        x=x_idx[i]
        y=y_idx[i]
        for j in range(len(theta_range)):
            theta = theta_range[j]
            rho = int((x*math.sin(theta)+y*math.cos(theta))) + max_dist
            # Any negative values have been converted to positive so they can be used as index.
            # print(rho,j)
            accumulator[rho, j] = accumulator[rho, j] + 1
    plt.figure(figsize=(10,10))
    plt.title('Accumulator Array after Hough Transform using Polar Co-ordinates')
    plt.imshow(accumulator)
    plt.show()

def hough_transform_euclidean(img):
    img = img.convert('L')
    # img = img.resize((400,400),1)
    img = np.array(img)
    plt.imshow(img)
    plt.show()
    rows,cols = img.shape
    diag_len = math.sqrt(rows**2+cols**2)
    img = zero_crossing_detect(img)
    plt.imshow(img)
    plt.show()

    slope_range = np.linspace(-5,5,360) # 300 slopes from -5 to 5
    intercept_range = np.linspace(-1*diag_len,diag_len,2*int(diag_len)) # values of intercept for the maximum distance image.
    accumulator = np.zeros((len(slope_range),len(intercept_range)))

    y_index, x_index = np.nonzero(img)

    for i in range(len(y_index)):
        x = x_index[i]
        y = y_index[i]
        print('Percentage Done :',i*100/len(y_index))
        for j in range(len(slope_range)):
            m = slope_range[j]
            c = y - (m*x)
            if -1*diag_len<=c<=diag_len:
                c = ((c+diag_len)/(2*diag_len))*len(intercept_range)
                # c+diag_len is done to get the value of c in the positive range as indexes are also positive.
                # value is divided by 2*diag_len to normalize then multiplied by length of intercept_range to find the bin
                # in which the value of c is to be placed.
                c = int(c)
                accumulator[j,c]=accumulator[j,c]+1
    plt.figure(figsize=(10,10))
    plt.title('Accumulator Array after Hough Transform using Euclidean Co-ordinates')
    plt.imshow(accumulator)
    plt.show()

img = Image.open('./house.jpeg')
hough_transform_polar(img)
hough_transform_euclidean(img)