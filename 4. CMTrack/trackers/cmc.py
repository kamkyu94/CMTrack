import pickle
import numpy as np


class CMC:
    def __init__(self, vid_name):
        super(CMC, self).__init__()

        if 'MOT17' in vid_name:
            vid_name = vid_name.split('-FRCNN')[0]
        elif 'dance' in vid_name:
            vid_name = 'dancetrack-' + vid_name.split('dancetrack')[1]

        self.gmcFile = open('./trackers/cmc/' + 'GMC-' + vid_name + ".txt", 'r')

    def get_warp_matrix(self):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        warp_matrix = np.eye(2, 3, dtype=np.float_)
        warp_matrix[0, 0] = float(tokens[1])
        warp_matrix[0, 1] = float(tokens[2])
        warp_matrix[0, 2] = float(tokens[3])
        warp_matrix[1, 0] = float(tokens[4])
        warp_matrix[1, 1] = float(tokens[5])
        warp_matrix[1, 2] = float(tokens[6])

        return warp_matrix


def apply_cmc(tracks, warp_matrix=np.eye(2, 3)):
    # Check
    if len(tracks) == 0:
        return 0

    # Get mean, covariance
    multi_mean = np.asarray([t.mean.copy() for t in tracks])
    multi_covariance = np.asarray([t.covariance for t in tracks])

    # Get warp matrix
    rot = warp_matrix[:, :2]
    rot_8x8 = np.kron(np.eye(4, dtype=float), rot)
    trans = warp_matrix[:, 2]

    # Warp
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        mean = rot_8x8 @ mean
        mean[:2] += trans
        cov = rot_8x8 @ cov @ rot_8x8.T

        tracks[i].mean = mean
        tracks[i].covariance = cov
