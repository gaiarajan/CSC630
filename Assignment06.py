import numpy as np
import cv2
import os

#most of this class is already written,
#but you will need to write the evaluate function
class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size


    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0] // 2, self.position[1]:self.position[1] + self.size[1]] = 255
        img[self.position[0]+ self.size[0] // 2:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = 126
        return img;

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1] // 2] = 255
        img[self.position[0]:self.position[0] + self.size[0], self.position[1] + self.size[1] // 2:self.position[1] + self.size[1]] = 126
        return img;

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype = np.uint32)
        img[self.position[0]:self.position[0] + self.size[0] // 3, self.position[1]:self.position[1] + self.size[1]] = 255
        img[self.position[0]+ self.size[0] // 3:self.position[0] + 2 * self.size[0] // 3, self.position[1]:self.position[1] + self.size[1]] = 126
        img[self.position[0]+ 2 * self.size[0] // 3:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = 255
        return img;


    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1] // 3] = 255
        img[self.position[0]:self.position[0] + self.size[0], self.position[1] + self.size[1] // 3:self.position[1] + 2 * self.size[1] // 3] = 126
        img[self.position[0]:self.position[0] + self.size[0], self.position[1] + 2 * self.size[1] // 3:self.position[1] + self.size[1]] = 255
        return img;

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype = np.uint32)
        img[self.position[0]:self.position[0] + self.size[0] // 2, self.position[1]:self.position[1] + self.size[1] // 2] = 126
        img[self.position[0]+ self.size[0] // 2:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1] // 2] = 255
        img[self.position[0]:self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2:self.position[1] + self.size[1]] = 255
        img[self.position[0]+ self.size[0] // 2:self.position[0] + self.size[0], self.position[1] + self.size[1] // 2:self.position[1] + self.size[1]] = 126

        return img;

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):
            X = self._create_four_square_feature(shape)

        return X

    def get_sum(self, ii, top_left, bottom_right):
        top_left=(int(top_left[0]), int(top_left[1]))
        bottom_right=(int(bottom_right[0]), int(bottom_right[1]))
        s4 = ii[bottom_right[1]][bottom_right[0]]

        s2_pos = (bottom_right[1], top_left[0])
        s2 = ii[s2_pos[0]][s2_pos[1]]

        s3_pos = (top_left[1], bottom_right[0])
        s3 = ii[s3_pos[0]][s3_pos[1]]

        s1 = ii[top_left[1]][top_left[0]]

        overallsum = s4 - s2 - s3 + s1
        return np.int32(overallsum)

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        To see what the different feature types look like, check the Haar Feature Types image

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        overallsum = 0

        if self.feat_type == (2, 1):
            print("two_vertical")
            row_height = self.size[0] / 2
            sum_of_white = self.get_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1))
            sum_of_grey = self.get_sum(ii, (self.position[1] - 1, self.position[0] + row_height - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))

            overallsum = sum_of_white - sum_of_grey

        elif self.feat_type == (1, 2):
            print("two_horizontal")
            column_width = self.size[1] / 2
            sum_of_white = self.get_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1))
            sum_of_grey = self.get_sum(ii, (self.position[1] + column_width - 1, self.position[0] - 1), (self.position[1] +  self.size[1] - 1, self.position[0] + self.size[0] - 1))

            overallsum = sum_of_white - sum_of_grey

        elif self.feat_type == (3, 1):
            print("three_horizontal")
            row_height = int(self.size[0] / 3)
            sum_of_white1 = self.get_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1))
            sum_of_grey = self.get_sum(ii, (self.position[1] - 1, self.position[0] + row_height - 1), (self.position[1] + self.size[1] - 1, self.position[0] + (row_height * 2) - 1))
            sum_of_white2 = self.get_sum(ii, (self.position[1] - 1, self.position[0] + (row_height * 2) - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))

            overallsum = sum_of_white1 + sum_of_white2 - sum_of_grey

        elif self.feat_type == (1, 3):
            print("three_vertical")
            column_width = self.size[1] / 3
            sum_of_white1 = self.get_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1))
            sum_of_grey = self.get_sum(ii, (self.position[1] + column_width - 1, self.position[0] - 1), (self.position[1] + (column_width * 2) - 1, self.position[0] + self.size[0] - 1))
            sum_of_white2 = self.get_sum(ii, (self.position[1] + (column_width * 2) - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))

            overallsum = sum_of_white1 + sum_of_white2 - sum_of_grey

        elif self.feat_type == (2, 2):
            print("square")
            row_height = int(self.size[0] / 2)
            column_width = int(self.size[1] / 2)

            sum_of_grey1 = self.get_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + column_width - 1, self.position[0] + row_height - 1))
            sum_of_white1 = self.get_sum(ii, (self.position[1] + column_width - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1))
            sum_of_white2 = self.get_sum(ii, (self.position[1] - 1, self.position[0] + row_height - 1), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1))
            sum_of_grey2 = self.get_sum(ii, (self.position[1] + column_width - 1, self.position[0] + row_height - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))
            overallsum = np.int32(0)
            overallsum = np.int32(sum_of_white1)
            overallsum -= np.int32(sum_of_grey2)
            overallsum += np.int32(sum_of_white2)
            overallsum -= np.int32(sum_of_grey1)
        print("sum: "+str(overallsum))
        return overallsum


def convert_image_to_integral_image(img):
    """Convert a list of grayscale images to integral images.

    Args:
        image :  Grayscale image (uint8 or float).

    Returns:
        2d Array : integral image.
    """
    return np.cumsum(np.cumsum(img, 0), 1)

#for this class, you will write the train, predict and faceDetect functions
class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        #now create all possible haar features that would work in a 24x24 window
        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(2*feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(2*feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):
         #This scores array is used to train a weak classifier using VJ_Classifier
        #class in steps 2, 3 of the boosting algorithm (which has been written for you)
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        #TODO: use the provided algorithm to select the best classifiers
        #Reminder that we use -1 for negative images (instead of 0)
        #self.posImages contains all the positive images, while self.negImages contains the negatives
        #self.labels contains the identity of each images corresponding to self.integralImages
        #Step 1 is to initialize the weights

        #TODO: initialize weights
        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        for t in range(num_classifiers):
            #TODO: normalize weights
            weights /= weights.sum()
            #DONE: choose the best classifier
            h = VJ_Classifier(scores, self.labels, weights)
            h.train(self.haarFeatures)
            self.classifiers.append(h)
            #TODO: update the weights.  To check if an image is classified correctly

            #you should use the predict function of h
            beta = h.error/(1-h.error)
            alpha = np.log(1/beta)
            for i in range(0, len(weights)):
                ei = 1
                if(h.predict(self.integralImages[i]) == self.labels[i]):
                    ei = 0
                weights[i] = weights[i] * (beta**(1-ei))
            #TODO: save the alpha value for our classifier
            self.alphas.append(alpha)

    def predict(self, image):
        """Return prediction for a given image.
        Use the strong classifier you've created to determine if a given image
        is a face or not.  Looking at the algorithm, you will need alphas, which you
        should have saved in train, and h_t(x) which is the predicted value from each
        weak classifier

        Args:
            image (numpy.array): a 24x24 image which may contain a face.

        Returns:
            int: 1 or -1, 1 if there is a face -1 if not a face
        """

        ii = convert_image_to_integral_image(image)
        for clf in self.classifiers:
            wk_clf = VJ_Classifier(clf.Xtrain, clf.ytrain, clf.weights)
            feat_id = wk_clf.feature
            hf = self.haarFeatures[feat_id]
        result = []
        threshold = np.array(self.alphas).sum()
        score_sum = 0
        i = 0
        ret = 0
        for clf in self.classifiers:
            score_sum += clf.predict(ii) * self.alphas[i]
            i += 1

        if(score_sum  >= threshold * .5):
            ret = 1
        else:
            ret = -1
        return ret

    def faceDetection(self, img):
        """Scans for faces in a given image.
        You will want to take every 24x24 window in the input image, and check if
        it contains a face.  You will probably get several hits, so you may want to
        combine the hits if they are nearby.  You can also consider increasing the 1/2 value
        in the inequality to reduce the number of hits (e.g. try 1/1.5)
        You should then draw a box around each face you find

        Args:
            image (numpy.array): Input image.
        Returns:
            an image with a box drawn around each face
        """
        slices = []
        cv2.imshow('hi',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        x_points = []
        y_points = []

        for x in range(0, img.shape[0] - 24):
            for y in range(0, img.shape[1] - 24):
                small_slice = img[x:x+24, y:y+24]
                slices.append(small_slice)
                prediction = self.predict(small_slice)
                slices = []
                if(prediction == 1):
                    x_points.append(x)
                    y_points.append(y)
        average_point = (int(np.average(x_points)), int(np.average(y_points)))
        resized_x = average_point[0]
        resized_y = average_point[1]
        print((resized_x, resized_y))
        resized_x = int(resized_x * img.shape[0] / img.shape[0])
        resized_y = int(resized_y * img.shape[1] / img.shape[1])
        print((resized_x, resized_y))
        resized_width = int(24 * img.shape[1] / img.shape[1])
        resized_height = int(24 * img.shape[0] / img.shape[0])
        cv2.rectangle(img, (resized_y, resized_x), (resized_y + 24, resized_x + 24), (0, 255, 0))
        return img


#you don't need to write anything down here,
#but you will need to use the predict function
class VJ_Classifier:
    """Weak classifier for Viola Jones procedure

    Args:
        X (numpy.array): Feature scores for each image. Rows: number of images
                         Columns: number of features.
        y (numpy.array): Labels array of shape (num images, )
        weights (numpy.array): observations weights array of shape (num observations, )

    Attributes:
        Xtrain (numpy.array): Feature scores, one for each image.
        ytrain (numpy.array): Labels, one per image.
        weights (float): Observations weights
        threshold (float): Integral image score minimum value.
        feat (int): index of the feature that leads to minimum classification error.
        polarity (float): Feature's sign value. Defaults to 1.
        error (float): minimized error (epsilon)
    """
    def __init__(self, X, y, weights, thresh=0, feat=0, polarity=1):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.weights = weights
        self.threshold = thresh
        self.feature = feat
        self.polarity = polarity
        self.error = 0

    def train(self, haar_features):
        """Trains a weak classifier that uses Haar-like feature scores.

        This process finds the feature that minimizes the error as shown in
        the Viola-Jones paper.

        Once found, the following attributes are updated:
        - feature: The column id in X.
        - threshold: Threshold (theta) used.
        - polarity: Sign used (another way to find the parity shown in the
                    paper).
        - error: lowest error (epsilon).
        """
        signs = [1] * self.Xtrain.shape[1]
        thresholds = [0] * self.Xtrain.shape[1]
        errors = [100] * self.Xtrain.shape[1]

        for f in range(self.Xtrain.shape[1]):
            tmp_thresholds = self.Xtrain[:,f].copy()
            tmp_thresholds = np.unique(tmp_thresholds)
            tmp_thresholds.sort()
            tmp_thresholds = [(tmp_thresholds[i]+tmp_thresholds[i+1])/2 for i in
                              range(len(tmp_thresholds)-1)]

            min_e = 10000000000000
            for theta in tmp_thresholds:
                for s in [1,-1]:
                    tmp_r = self.weights * ( s*((self.Xtrain[:,f]<theta)*2-1) != self.ytrain )
                    tmp_e = sum(tmp_r)
                    if tmp_e < min_e:
                        thresholds[f] = theta
                        signs[f] = s
                        errors[f] = tmp_e
                        min_e = tmp_e

        feat = errors.index(min(errors))
        self.feature = haar_features[feat]
        self.threshold = thresholds[feat]
        self.polarity = signs[feat]
        self.error = errors[feat]

    def predict(self, ii):
        """Returns a predicted label.

        Inequality shown in the Viola Jones paper for h_j(x).

        Args:
            ii (numpy.array):  integral image of the image we want to predict

        Returns:
            float: predicted label (1 or -1)
        """
        return self.polarity * ((self.feature.evaluate(ii) < self.threshold) * 2 - 1)
