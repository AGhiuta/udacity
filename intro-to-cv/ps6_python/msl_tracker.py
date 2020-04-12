import cv2
import numpy as np


class MSLTracker(object):
    def __init__(self, model, search_space,
                 num_particles=100, state_dims=2, std_control=10,
                 std_CHI=20, num_bins=8, color_space="BGR", alpha=0.0):
        self.model = model
        self.num_particles = num_particles
        self.std_control = std_control
        self.std_CHI = std_CHI
        self.num_bins = num_bins
        self.color_space = color_space
        self.alpha = alpha
        self.particles = np.array([np.random.uniform(
            0, search_space[i], num_particles) for i in range(state_dims)]).T
        self.weights = np.ones(num_particles, dtype=np.float) / num_particles
        self.idxs = np.arange(num_particles)
        self.model_hist = self.__compute_hist(model)
        self.__estimate_state()

    def update(self, frame):
        self.particles += np.random.normal(0,
                                           self.std_control,
                                           size=self.particles.shape)
        self.__update_weights(frame)
        self.__resample()
        self.__estimate_state()

        if self.alpha:
            self.__update_model(frame)

    def visualize(self, frame):
        self.__draw_particles(frame)
        self.__draw_bbox(frame)
        self.__draw_std(frame)

    def __update_model(self, frame):
        mh, mw = self.model.shape[:2]
        miny = (self.state[1] - mh/2).astype(np.int)
        minx = (self.state[0] - mw/2).astype(np.int)
        best_model = frame[miny:miny+mh, minx:minx+mw]

        if best_model.shape == self.model.shape:
            self.model = (self.alpha * best_model +
                          (1 - self.alpha) * self.model).astype(np.uint8)
            self.model_hist = self.__compute_hist(self.model)

    def __update_weights(self, frame):
        mh, mw = self.model.shape[:2]
        miny = (self.particles[:, 1] - mh/2).astype(np.int)
        minx = (self.particles[:, 0] - mw/2).astype(np.int)
        patches = [frame[y:y+mh, x:x+mw] for (y, x) in zip(miny, minx)]
        self.weights = np.array([self.__compute_CHI(patch)
                                 for patch in patches])
        self.weights /= np.sum(self.weights)

    def __resample(self):
        idxs = np.random.choice(self.idxs, self.num_particles,
                                replace=True, p=self.weights)
        self.particles = self.particles[idxs]

    def __estimate_state(self):
        self.state = np.average(self.particles, axis=0, weights=self.weights)
        # self.state = self.particles[np.argmax(self.weights)]

    def __compute_CHI(self, patch):
        if patch.shape != self.model.shape:
            return 0

        if self.color_space == "HSV":
            tmp = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        elif self.color_space == "GRAY":
            tmp = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            tmp = patch

        hist = self.__compute_hist(tmp)
        chi = cv2.compareHist(self.model_hist, hist, method=cv2.HISTCMP_CHISQR)

        return np.exp(-0.5 * chi / self.std_CHI**2)

    def __compute_hist(self, patch):
        num_channels = 1 if len(patch.shape) == 2 else patch.shape[-1]
        channels = list(range(num_channels))

        if self.color_space == "HSV":
            hist_size = [6, 3, 3]
            hist_ranges = [0, 180, 0, 256, 0, 256]
        else:
            hist_size = [self.num_bins] * num_channels
            hist_ranges = [0, 256] * num_channels

        hist = cv2.calcHist([patch], channels, None,
                            hist_size, hist_ranges, accumulate=False)

        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return hist

    def __draw_particles(self, frame):
        for p in self.particles.astype(np.int):
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)

    def __draw_bbox(self, frame):
        pt1 = (self.state -
               np.array(self.model.shape[:2][::-1]) / 2).astype(np.int)
        pt2 = pt1 + np.array(self.model.shape[:2][::-1])

        cv2.rectangle(frame, tuple(pt1), tuple(pt2),
                      (0, 255, 0), 2, lineType=cv2.LINE_AA)

    def __draw_std(self, frame):
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.reshape((-1, 1)))

        cv2.circle(frame, tuple(self.state.astype(np.int)),
                   int(weighted_sum), (255, 255, 255), 1)
