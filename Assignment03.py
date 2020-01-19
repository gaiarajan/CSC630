"""
Submission for assignment 3
"""
import cv2
import numpy as np
from scipy import stats
def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (tuple): min and max values for rad8fsjdFSdfhsdlbhewfojfdjsohsdoghskldfjlsjfapsdjfklnsius

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    img_in2 = cv2.medianBlur(img_in,5)
    cimg = cv2.cvtColor(img_in2,cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayscale", cimg)
    cv2.waitKey(0)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20, param1=30,param2=15,minRadius=radii_range[0],maxRadius=radii_range[1])
    circles.sort(axis=0)
    ret = np.zeros((4,3))
    jet = np.zeros((4,3))
    finalCircles = np.array([[0,0,0]])
    if circles[0][0][0] != circles[0][1][0]:
        for a in circles[0,1:]:
            finalCircles = np.append(finalCircles, [a], axis = 0)
    else:
        for b in circles[0,0:3]:
            finalCircles = np.append(finalCircles, [b], axis = 0)
    finalCircles = np.delete(finalCircles,0,axis = 0)
    for i in finalCircles[:]:
        jet = cv2.circle(img_in2,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    x0=int(finalCircles[0][0])
    y0=int(finalCircles[0][1])
    x1=int(finalCircles[1][0])
    y1=int(finalCircles[1][1])
    x2=int(finalCircles[2][0])
    y2=int(finalCircles[2][1])
    green = img_in[x0,y0][1]
    yellow = int((int(img_in[x1,y1][1])+int(img_in[x1,y1][2]))/2)
    red = img_in[x2,y2][2]
    print((red, yellow, green))
    state = 'yellow'
    if green>yellow and green>red:
        state = 'green'
    if yellow>=green and yellow>=red:
        state = 'yellow'
    if red>yellow and red>green:
        state = 'red'
    return [(int(finalCircles[1][0]),int(finalCircles[1][1])), state]

def stop_sign_detection(img, noisy_img = False):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image that may contain a stop sign.
        noisy_img (boolean): tells whether the image has a lot of noise

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.  Or None
        if there is no stop sign in the scene
    """
    if(noisy_img):
        img = cv2.GaussianBlur(img,(5,5),0)
    ret,thresh1 = cv2.threshold(img[:,:,0], 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresh1', thresh1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        print(len(approx))
        if len(approx)==8:
            M = cv2.moments(cnt)
            x = int(M["m10"]/ M["m00"])
            y = int(M["m01"] / M["m00"])
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 6)
            cv2.imshow('sign', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return (x,y)
    return None

def yield_sign_detection(img, noisy_img = False):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image that may contain a yield sign
        noisy_img (boolean): tells whether the image has a lot of noise

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.  Or None
        if there is no yield sign in the scene
    """
    if(noisy_img):
        img = cv2.GaussianBlur(img,(5,5),0)
    ret,thresh1 = cv2.threshold(img[:,:,0], 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresh1', thresh1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        print(len(approx))
        if len(approx)==3:
            M = cv2.moments(cnt)
            x = int(M["m10"]/ M["m00"])
            y = int(M["m01"] / M["m00"])
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 6)
            cv2.imshow('sign', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return (x,y)
    return None


def do_not_enter_sign_detection(img, noisy_img = False):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image that may contain a do not enter sign.
        noisy_img (boolean): tells whether the image has a lot of noise

    Returns:
        (x,y) typle of the coordinates of the center of the sign.  Or None
        if there is no dne sign in the scene
    """
    if(noisy_img):
        img = cv2.GaussianBlur(img,(5,5),0)
    gray = img[:,:][2]
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            return center
    return None
