import numpy as np
import cv2

def get_input_values_for_filter(part_number):
    """A function to get the values for various constant values for the
    different parts of this assignment

    Args:
        part_number (int):  which part of the project we're getting values for, in [1, 4]
    Returns:
        tuple of numbers: (num_particles, sigma_exponential, sigma_dynamic, alpha)
    """
    if part_number == 1:
        #CHANGE THESE VALUES
        num_particles = 400
        sigma_exponential = 2
        sigma_dynamic = 5
        alpha = 0
    elif part_number == 2:
        #CHANGE THESE VALUES
        num_particles = 1000
        sigma_exponential = 5
        sigma_dynamic = 5
        alpha = 0
    elif part_number == 3:
        #CHANGE THESE VALUES
        num_particles = 1000
        sigma_exponential = 3
        sigma_dynamic = 30
        alpha = 0.1
    elif part_number == 4:
        #CHANGE THESE VALUES
        num_particles = 1000
        sigma_exponential = 3
        sigma_dynamic = 30
        alpha = 0
    return (num_particles, sigma_exponential, sigma_dynamic, alpha)



class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in tests.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template_rect, num_particles, sigma_exp, sigma_dyn, alpha = 0):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one).
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template_rect (dict):  Template coordinates with x, y,
                                            width, and height values.
            num_particles (int): number of particles.
            sigma_exp (float): sigma value used in the similarity measure.
            sigma_dyn (float): sigma value that can be used when adding gaussian noise to particle positions.
            alpha (float):  value used to determine how much we adjust our template
        """
        self.num_particles = num_particles
        self.sigma_exp = sigma_exp
        self.sigma_dyn = sigma_dyn
        self.alpha = alpha
        self.template_rect = template_rect
        x = template_rect['x']
        y = template_rect['y']
        w = template_rect['w']
        h = template_rect['h']
        self.template = (frame[int(y):int(y+h), int(x):int(x+w)])[:,:,0]
        """cv2.imshow("template", self.template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        self.frame = frame
        self.particles = np.zeros((self.num_particles,2))
        self.weights = np.ones((self.num_particles, 1))*(1.0/self.num_particles)
        self.frameCount = 0
        self.particles[:,0] = self.template_rect['x'] + (np.random.rand(self.num_particles)*self.template.shape[0]).astype('int')
        self.particles[:,1] = self.template_rect['y'] + (np.random.rand(self.num_particles)*self.template.shape[1]).astype('int')
    def get_likelihood(self, template, frame_cutout):
        """Returns the likelihood (probability) measure of observing the template at the given
            frame_cutout location.  We will do this by calculating the mean squared difference between
            the template we're tracking and the frame_cutout, then we use an exponential function
            to convert that value to a probability
            You should try to calculate exp(-1*meanSqDiff / 2 * sigma^2) and return that value
        Returns:
            float: likelihood value
        """
        rows,cols = template.shape[0],template.shape[1]
        squareError = np.mean((template.astype("float") - frame_cutout.astype("float"))**2)
        #meanSquareError = squareError/float(rows*cols)
        newWeight = np.exp(-1*squareError/float(2*self.sigma_exp**2))
        return newWeight

    def resample_particles(self):
        """Returns a new set of particles

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        newParticles = np.zeros((self.num_particles,2))
        particleDistribution = np.random.multinomial(self.num_particles, self.weights.reshape(self.num_particles).tolist(), size=1)
        particleCount = 0
        for i in range(self.num_particles):
            newParticles[particleCount:particleDistribution[0][i]+particleCount] = self.particles[i]
            particleCount += particleDistribution[0][i]
        return newParticles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Your general algorithm should look something like this:
        resample particles
        for each particle:
            disperse it using sigma_dyn
            get the cutout corresponding to the particle
            calculate the likelyhood of the cutout based on the template
            change the weight of the particle
        normalize the weights
        update the template based on alpha (not needed for part 1, 2)

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = np.average(frame,axis=2).astype('uint8')
        newParticles = self.resample_particles()

        newParticles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles = newParticles

        newWeight = np.zeros((self.num_particles, 1))

        for i in range(self.num_particles):
            v,u = self.particles[i][0],self.particles[i][1]
            frame_cutout = self.get_frame_cutout(u,v,self.template,frame)
            newWeight[i] = self.get_likelihood(self.template, frame_cutout)

        self.weights = newWeight/np.sum(newWeight)
        if self.alpha>0:
            highestWeight = self.weights[0]
            highestWeightIdx = 0
            for i in range(self.num_particles):
                if self.weights[i] > highestWeight:
                    highestWeight = self.weights[i]
                    highestWeightIdx = i
            u,v = self.particles[highestWeightIdx,0], self.particles[highestWeightIdx,1]
            bestWindow = self.get_frame_cutout(u,v,self.template,frame)
            self.template = (self.alpha*bestWindow + (1-self.alpha)*self.template).astype('uint8')
            self.template = cv2.normalize(self.template,None,0,255,cv2.NORM_MINMAX)
    def get_frame_cutout(self, u, v, template,frame):
        rows = template.shape[0]
        cols = template.shape[1]
        upper = u-rows/2
        lower = u+rows/2
        left = v-cols/2
        right = v+cols/2
        if upper < 0:
            upper = 0
        elif upper >= frame.shape[0]:
            upper = frame.shape[0]-1
        if lower > frame.shape[0]:
            lower = frame.shape[0]-1
        elif lower < 0:
            lower = 0
        if left < 0:
            left = 0
        elif left > frame.shape[1]:
            left = frame.shape[1]-1
        if right > frame.shape[1]:
            right = frame.shape[1]-1
        elif right < 0:
            right = 0
        upper = int(upper)
        lower = int(lower)
        left = int(left)
        right = int(right)
        cutoutRows = lower - upper
        cutoutCols = right - left
        frame_cutout = np.zeros(template.shape)
        frame_cutout[0:cutoutRows, 0:cutoutCols] = frame[upper:lower, left:right]
        return frame_cutout
    def render(self, frame_in):
        """Visualizes current particle filter state.

        Don't do any model updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius (Make it a color that will standout!).
        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        Returns:
            Nothing, but you should do all of your drawing on the frame_in
        """
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        mean = [x_weighted_mean,y_weighted_mean]

        diff = np.zeros((self.num_particles, 1))
        for i in range(self.num_particles):
            diff[i,0] = np.sqrt((self.particles[i,0]-mean[0])**2 + (self.particles[i,1]-mean[1])**2)

        spread = np.sum(diff*self.weights)/float(np.sum(self.weights))

        for i in range(self.num_particles):
            pt1 = (int(self.particles[i,0]), int(self.particles[i,1]))
            cv2.circle(frame_in, pt1, 1, (0,255,0), thickness=1)
        self.width, self.height = self.template.shape[1],self.template.shape[0]
        pt1 = (int(mean[0]-self.width/2), int(mean[1]-self.height/2))
        pt2 = (int(mean[0]+self.width/2), int(mean[1]+self.height/2))
        cv2.rectangle(frame_in, pt1, pt2, (0,255,0), thickness=1)
        cv2.circle(frame_in, (mean[0], mean[1]), spread.astype('int'), (0,0,255), thickness=2)
