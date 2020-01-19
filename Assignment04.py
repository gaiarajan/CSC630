"""
    ?pu;jAssignment 4 imports
"""
import cv2
import numpy as np
import math
import operator


def find_markers(image, template = None, prevFrameMarkers = None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.
        prevFrameMarkers:  List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
            that were the marker locations in the previous frame of the video

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    """average convolution result and prev frame"""
    print("running function for markers")
    copy = np.copy(image)
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    copy = cv2.GaussianBlur(image,(9,9),0)
    #find corners and get set of points
    harris = cv2.cornerHarris(gray, blockSize = 6, ksize = 7, k = 0.04)
    (r, c) = np.where(harris > np.max(harris) / 8.0)
    points = np.float32(np.vstack((c, r)).T)  #convert the output of np.where to a 2d array of points (Nx2), this is needed for using kmeans
    corner_copy = np.copy(image)
    for p in points:
        cv2.circle(corner_copy, tuple(p), 1, (234, 26, 232), thickness = -1)  #show points on the image, note: we need to change p from a list to a tuple
        compactness, classified_points, means = cv2.kmeans(data = points, K=4,
            bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
            attempts=1, flags=cv2.KMEANS_PP_CENTERS)
    ret = []
    for i in range(4):
        ret.append((int(means[i][0]), int(means[i][1])))
    print(ret)
    ret.sort(key = operator.itemgetter(0))
    ret[0:2] = sorted(ret[0:2], key=lambda tup: tup[1])
    print(ret)
    ret[2:4] = sorted(ret[2:4], key=lambda tup: tup[1])
    print(ret)
    return ret

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    cv2.line(image, markers[0], markers[1], thickness)
    cv2.line(image, markers[0], markers[2], thickness)
    cv2.line(image, markers[1], markers[3], thickness)
    cv2.line(image, markers[2], markers[3], thickness)
    return image



def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Take a look at: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography#findhomography
    It will do almost everything for you, you just need to set the parameters and deal with the return values
    Alternatively you can code it by hand:  http://www.csc.kth.se/~perrose/files/pose-init-model/node17.html

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    h, mask = cv2.findHomography(np.float32(src_points), np.float32(dst_points), cv2.RANSAC,5.0)
    return h


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the homography given we should be able to easily project imageA into the correct spot in imageB
    Take a look at:  https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective
    Hint:  use cv2.BORDER_TRANSPARENT for the border mode

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    height, width, channels = imageB.shape
    final = cv2.warpPerspective(imageA, homography, (width, height), imageB, borderMode=cv2.BORDER_TRANSPARENT)
    return final

def insert_video(baseVideo, addVideo):
    """Inserts the addVideo into the marked portion of the baseVideo.  You may want
    to take a look at some of the methods in tests.py

    hint:  take a look at helper_for_part_4_and_5 in tests.py to see how to open and save videos

    inputs:  two video files

    output:  nothing, but you should save the file you generated
    """
    image_gen = baseVideo
    image_add = addVideo
    print(addVideo)
    image = next(image_gen)
    image2 = next(image_add)
    h, w, d = image.shape

    out_path = "output/" + "part_6"
    video_out = mp4_video_writer(out_path, (w, h), 40)

    # Optional template image
    template = cv2.imread("inputs/template.jpg")

    src_points = get_corners_list(image2)

    frame_num = 1

    markers = None

    while image is not None:

        print ("Processing frame " + str(frame_num))

        markers = find_markers(image, template, markers)

        homography = find_four_point_transform(src_points, markers)
        image = project_imageA_onto_imageB(image2, image, homography)

        video_out.write(image)

        image = next(image_gen)
        image2 = next(image_add)

        frame_num += 1

    video_out.release()

#the methods below you don't need to modify (but you may want to use in insert_video)
def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)


    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None;


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi = filename.split('.')[0] + '.avi'
    print(avi)
    return cv2.VideoWriter(avi, fourcc, fps, frame_size)

def get_corners_list(image):
    """Returns a list of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    return [(0, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, 0), (image.shape[1] - 1, image.shape[0] - 1)];
