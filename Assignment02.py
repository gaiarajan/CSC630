"""
Submission for assignment 2
"""
import cv2
import math
import numpy as np

def blur_image_gaussian(image):
    """returns the image that has been blurred using a gaussian filter
    you should determing appropriate values for the filter to be used
    See here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html


    Input:  an image

    Output:  numpy.array:  the blurred image
    """
    blur = cv2.GaussianBlur(image,(5,5),0)
    return blur

def shifted_difference(image, left_shift):
    """returns the image that has been shifted left and then subtracted from itself.
    The image should be converted to grayscale first
    This was basically done in assignment 1

    Input: an image

    Output: numpy.array:  the result of shifting and subtracting the image
    """
    shift = left_shift
    temp_image = np.copy(image)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    left_shift_image = temp_image[:, shift:]
    left_shift_image = cv2.copyMakeBorder(left_shift_image, 0, 0, 0, shift, cv2.BORDER_REPLICATE);
    temp_image1 = np.copy(temp_image).astype(float)
    temp_image2 = np.copy(left_shift_image).astype(float)
    ret = temp_image2 - temp_image1
    cv2.normalize(ret, ret, 0, 255, cv2.NORM_MINMAX)
    ret/=255
    print(ret[0:10,0:10])
    return ret

def sobel_image(image):
    """returns the image that has had a Sobel filter applied to it.  Look here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?#sobel
    for more information about how to use a sobel filter in opencv
    You should make sure to convert the image to grayscale first, and you may want to blur it as well
    You should also mess around with the different arguments to see what the effects are

    Input: an image

    Output:  numpy.array:  an image
    """
    temp_image = np.copy(image)
    temp_image = cv2.GaussianBlur(temp_image, (3, 3), 0)
    gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def canny_image(image):
    """use the canny edge operator to highlight the edges of an image.  Look here:
    https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    for some information about how to use the canny edge detector
    You should make sure to convert the image to grayscale first, and you may want to blur it as well

    Input: an image

    Output:  numpy.array:  an edge image
    """
    temp_image = np.copy(image)
    temp_image = cv2.GaussianBlur(temp_image, (3, 3), 0)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(temp_image,100,200)
    return edges

def custom_line_detector(image):
    """create your own Hough Line accumulator that will tell you all of the lines on a given image
    to start you will want to setup the image by using Canny to create an edge image and maybe blur as well (notice a pattern??).
    Then you will  need to look at all the edges and have them "vote" for lines that they belong to.  Choose the most
    relevant lines and return them.  You can look here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    for more ideas

    Input:  an image

    Output:  Vector of lines in the form (rho, theta)
    """
    temp_image = np.copy(image)
    gray = cv2.cvtColor(temp_image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    edge_idx = np.where(edges!=0)
    print(edge_idx)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    return lines

    """
    diagonal = int(np.sqrt((image.shape[0])**2+(image.shape[1])**2))
    rows = 2*diagonal
    H = np.zeros((rows, 181))
    for i in range(edge_idx[0].shape[0]):
        x = edge_idx[1][i]
        y = edge_idx[0][i]
        for theta in range(181):
            d=int(np.cos(theta*np.pi/180)*x+np.sin(theta*np.pi/180)*y)
            H[d+diagonal][theta]+=1
    thresh = 15
    intermediate = np.where(H > thresh)
    final_arr = (intermediate[0]-i, intermediate[1]/180*np.pi)
    ret = np.expand_dims(np.array(final_arr).T, 1)
    return ret
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(grayscale,50,150)
    theta = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))
    diag = round(math.sqrt(edge.shape[0]**2 + edge.shape[1]**2))
    accumulator = np.zeros((2 * diag, len(theta)), dtype=np.int8)
    edge_pixels = np.where(edge!=0)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))
    for p in range(len(coordinates)):
        for t in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 1
    edge_pixels = np.where(accumulator > 110)
    coordinates = list(zip(edge_pixels[0]-diag, edge_pixels[1]))
    print(coordinates)
    return coordinates"""
def draw_lines_on_image(lines, image):
    """draws the given lines on the image.  Note that the input lines are the same values you
    return in custom_line_detector.  See how to draw lines here:
    https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#line

    Input:  an image and a list of lines

    Output:  numpy.array:  an image with lines drawn on it
    """
    for i in range(0, len(lines)):
        rho = lines[i][0]
        theta = lines[i][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return image
def hough_line_detector(image):
    """now you will use the Hough line detector that is available in open cv to redo
    what you did in the custom_line_detector method.  See here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?#houghlines

    Input:  an image

    Output:  Vector of lines in the form (rho, theta)
    """
    temp_image = np.copy(image)
    gray = cv2.cvtColor(temp_image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    return lines

def hough_circle_detector(image):
    """now use the Hough Circle detector to find circles in a given image.  See here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?#houghcircles

    input:  an image to find circles in

    output:  Vector of circles in the form (x, y, radius)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50)
    return circles

def draw_circles_on_image(circles, image):
    """draws the given circles on the image.  Note that the input circles are the same values you
    return in hough_circle_detector.  See here:
    https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#circle

    Input:  an image and a list of circles

    Output:  numpy.array:  an image with circles drawn on it
    """
    temp_image=np.copy(image)
    for i in circles[0,:]:
        cv2.circle(temp_image,(i[0],i[1]),i[2],(0,255,0),2)
    return temp_image
