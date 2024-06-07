import numpy as np
import scipy.linalg


class KalmanFilter(object):
    def __init__(self):
        # Set dim
        self.dim_x = 8
        self.dim_z = 4

        # Set motion matrix (F)
        self.motion_mat = np.eye(self.dim_x)
        for i in range(self.dim_x // 2):
            self.motion_mat[i, self.dim_z + i] = 1

        # Set update matrix (H)
        self.update_mat = np.eye(self.dim_z, self.dim_x)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self.std_pos = 1. / 20.
        self.std_vel = 1. / 160.

        # Set motion covariance
        self.motion_cov = np.eye(self.dim_x)
        self.motion_cov[:4, :4] *= self.std_pos
        self.motion_cov[4:, 4:] *= self.std_vel

        # Set innovation cov
        self.innovation_cov = np.eye(self.dim_z)
        self.innovation_cov *= self.std_pos

    def initiate(self, measurement):
        # Initialize mean
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initialize covariance
        covariance = np.eye(self.dim_x)
        covariance[:4, :4] *= 2.
        covariance[4:, 4:] *= 10.
        covariance[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        covariance[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        covariance = np.square(covariance)

        return mean, covariance

    def predict(self, mean, covariance):
        # Predict mean
        mean = np.dot(self.motion_mat, mean)

        # Get motion covariance
        motion_cov = self.motion_cov.copy()
        motion_cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        motion_cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        motion_cov = np.square(motion_cov)

        # Predict covariance
        covariance = np.linalg.multi_dot((self.motion_mat, covariance, self.motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence):
        # Project mean
        mean = np.dot(self.update_mat, mean)

        # Get innovation covariance
        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= mean[2]
        innovation_cov[[1, 3], [1, 3]] *= mean[3]
        innovation_cov = np.square(innovation_cov)

        # NSA Kalman
        innovation_cov *= (1 - confidence)

        # Project covariance
        covariance = np.linalg.multi_dot((self.update_mat, covariance, self.update_mat.T)) + innovation_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, confidence):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)

        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self.update_mat.T).T,
                                             check_finite=False).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance
