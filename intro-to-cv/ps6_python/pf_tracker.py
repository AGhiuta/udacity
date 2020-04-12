import cv2
import numpy as np


class PFTracker(object):
    def __init__(self, model, search_space,
                 num_particles=100, state_dims=2, std_control=10,
                 std_MSE=20, alpha=0.0):
        self.model = model
        self.num_particles = num_particles
        self.std_control = std_control
        self.std_MSE = std_MSE
        self.alpha = alpha
        self.particles = np.array([np.random.uniform(
            0, search_space[i], num_particles) for i in range(state_dims)]).T
        self.weights = np.ones(num_particles, dtype=np.float) / num_particles
        self.idxs = np.arange(num_particles)
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

    def __update_weights(self, frame):
        mh, mw = self.model.shape[:2]
        miny = (self.particles[:, 1] - mh/2).astype(np.int)
        minx = (self.particles[:, 0] - mw/2).astype(np.int)
        patches = [frame[y:y+mh, x:x+mw] for (y, x) in zip(miny, minx)]
        self.weights = np.array([self.__compute_MSE(patch)
                                 for patch in patches])
        self.weights /= np.sum(self.weights)

    def __resample(self):
        idxs = np.random.choice(self.idxs, self.num_particles,
                                replace=True, p=self.weights)
        self.particles = self.particles[idxs]

    def __estimate_state(self):
        self.state = np.average(self.particles, axis=0, weights=self.weights)

    def __compute_MSE(self, patch):
        if patch.shape != self.model.shape:
            return 0

        mse = np.sum(np.subtract(patch, self.model, dtype=np.float)
                     ** 2) / float(np.prod(patch.shape))

        return np.exp(-0.5 * mse / self.std_MSE**2)

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
