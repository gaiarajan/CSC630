import math
import numpy as np
import cv2

def extract_blue(image):
    """returns the blue channel of the image.  I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  an image in BGR format (typical for openCV but many programs use RGB instead)

    Output:  numpy.array:  A 2d array containing the blue channel of the image
    """
    temp_image = np.copy(image)
    return temp_image[:,:,0]


def extract_red(image):
    """returns the red channel of the image.  I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  an image in BGR format (typical for openCV but many programs use RGB instead)

    Output:  numpy.array:  A 2d array containing the red channel of the image
    """
    temp_image = np.copy(image)
    return temp_image[:,:,2]

def swap_green_blue(image):
    """returns an image where the green and blue channels have been swapped.
    I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  an image in BGR format (typical for openCV but many programs use RGB instead)

    Output:  numpy.array:  A 3d array containing the modified image
    """
    temp_image = np.copy(image)
    temp_image[:,:,1] = image[:,:,0] #temp_image green=original blue
    temp_image[:,:,0] = image[:,:,1] #temp image blue= original green
    return temp_image

def copy_paste_middle(src, dst, size):
    """copies the middle of the src image into the dst image and returns the new image.  I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  two images and a tuple (size) containing the dimensions of the middle section to copy

    Output:  numpy.array:  a modified image
    """
    original_size = np.shape(src)
    start_x = int((original_size[0] - size[0]) / 2)
    end_x = int(original_size[0] - start_x)
    start_y = int((original_size[1] - size[1]) / 2)
    end_y = int(original_size[0] - start_x)
    temp_image = np.copy(src)
    dst = temp_image[start_x : end_x, start_y : end_y]
    return dst

def image_stats(image):
    """returns the tuple of (min, max, mean, std dev) statistics for the given input image.
    You should try to find predefined functions to do these operations in numpy.  I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  a monochrome image

    Output:  a 4 element tuple containing the min, max, mean and std dev values of the input image
    """
    temp_image =  np.copy(image)
    min = np.amin(temp_image)
    max = np.amax(temp_image)
    mean = np.mean(temp_image)
    std_dev = np.std(temp_image)
    ret = (min, max, mean, std_dev)
    return ret

def shift_image_left(image, shift):
    """returns the input image that has been shifted shift pixels to the left.
    the returned image should have the same shape as the original and use the BORDER_REPLICATE rule
    to determine the value of any needed pixels.
    I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  a monochrome image and an int representing a number of pixels to shift

    Output:  numpy.array:  a 2d image that has been shifted left
    """
    temp_image = np.copy(image)
    left_shift = temp_image[:, shift:]
    borderType = cv2.BORDER_REPLICATE
    ret = cv2.copyMakeBorder(left_shift, 0, 0, 0, shift, borderType);
    return ret

def difference_image(img1, img2):
    """returns the difference of the two images (img1 - img2).  The returned image should be normalized to
    have pixel values in the range [0, 255]
    I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  two monochrome images

    Output:  numpy.array:  a monochrome image that is the normalized subtraction of the two inputs
    """
    temp_image1 = np.copy(img1).astype(float)
    temp_image2 = np.copy(img2).astype(float)
    ret = temp_image1 - temp_image2
    cv2.normalize(ret, ret, 0, 255, cv2.NORM_MINMAX)
    return (ret/255)

def add_noise(image, channel, sigma):
    """returns a copy of the input image that has had gaussian noise added to the specified channel
    The mean of the noise is 0, and the std dev is sigma.
    The returned array should NOT be normalized (i.e. there may be pixel values outside [0, 255])

    Take a look at:  https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html
    for information about how to generate gaussian (aka normal) random numbers

    I would recommend that you first make a copy of
    the input image to avoid modifying the original array.  You can make a copy by calling:
    temp_image = np.copy(image)

    Input:  a color image, the channel to modify, and the sigma for the noise

    Output:  numpy.array:  the modified image
    """
    temp_image = np.copy(image)
    mean = 0.0
    std = sigma
    img = temp_image[:,:,channel]
    noisy_img = img + np.random.normal(mean, std, img.shape)
    return noisy_img
def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".
    Args: image(numpy.array): Input 2D image.
    scale (int or float): scale factor.
    Returns:
    numpy.array: Output 2D image.
    """
    ret = np.copy(image)
    ret = ret / np.std(image) * scale
    print(image_stats(ret))
    return ret;
