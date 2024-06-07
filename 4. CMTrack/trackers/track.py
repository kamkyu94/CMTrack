import numpy as np
from trackers.kalman_filter import KalmanFilter


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackCounter(object):
    track_count = 0

    def get_track_id(self):
        self.track_count += 1
        return self.track_count


class BaseTrack(object):
    track_id = 0
    start_frame_id = 0
    current_frame_id = 0
    state = TrackState.New

    @property
    def end_frame_id(self):
        return self.current_frame_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class Track(BaseTrack):
    def __init__(self, args, detection):
        # Initialize 1
        self.args = args
        self.box = detection[:4]  # x1y1x2y2
        self.score = detection[4]

        # Initialize 2
        self.history = None
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        # Initialize 3
        self.alpha = 0.90
        self.feat = detection[6:][np.newaxis, :].copy()

    def update_features(self, feat, score):
        # For Confidence-aware feature update (Delete for BaseTracker)
        self.alpha = 0.95 + 0.05 * (1 - score)

        # Update and normalize
        self.feat = self.alpha * self.feat + (1 - self.alpha) * feat
        self.feat /= np.linalg.norm(self.feat)

    def predict(self):
        # Zero out the velocity of w and h when track is lost. maintain the w and h.
        if self.state != TrackState.Tracked:
            self.mean[6] = 0
            self.mean[7] = 0

        # Predict
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def initiate(self, frame_id, counter):
        # Get new track id
        self.track_id = counter.get_track_id()

        # Initiate Kalman filter
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())

        # Initiate history
        self.history = [[frame_id, self.box.copy(), self.score.copy(), self.mean.copy(), self.covariance.copy(),
                         self.feat.copy()]]

        # Initiate parameters
        self.start_frame_id = frame_id
        self.current_frame_id = frame_id
        self.state = TrackState.New

    def update(self, frame_id, detection):
        # Observation centric re-update
        if self.state == TrackState.Lost:
            # Update state
            self.state = TrackState.Tracked

            # Get old observation
            obs_x1, obs_y1, obs_x2, obs_y2 = self.history[-1][1].copy()
            self.mean, self.covariance = self.history[-1][3].copy(), self.history[-1][4].copy()

            # Get detected box
            det_x1, det_y1, det_x2, det_y2 = detection.box

            # Calculate velocity with constant velocity hypothesis
            dt = frame_id - self.history[-1][0]
            vel = [(det_x1 - obs_x1) / dt, (det_y1 - obs_y1) / dt, (det_x2 - obs_x2) / dt, (det_y2 - obs_y2) / dt]

            # Generate virtual trajectory (Linear)
            for i in range(dt):
                # Generate new box
                x1 = obs_x1 + (i + 1) * vel[0]
                y1 = obs_y1 + (i + 1) * vel[1]
                x2 = obs_x2 + (i + 1) * vel[2]
                y2 = obs_y2 + (i + 1) * vel[3]
                new_detection = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)])

                # Predict, update
                self.predict()
                self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                                       new_detection.copy(), detection.score)

        # Update Kalman filter
        else:
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                                   detection.cxcywh.copy(), detection.score)

        # Update history
        self.history.append([frame_id, detection.box.copy(), detection.score,
                             self.mean.copy(), self.covariance.copy(), detection.feat.copy()])

        # Update parameters
        self.box = detection.box.copy()
        self.score = detection.score
        self.current_frame_id = frame_id
        self.state = TrackState.Tracked if len(self.history) >= self.args.min_len else TrackState.New
        self.update_features(detection.feat.copy(), detection.score)

    @property
    def cxcywh(self):
        # Get current position in bounding box format `(center x, center y, aspect ratio, height)`.
        if self.mean is None:
            cx = (self.box[0] + self.box[2]) / 2
            cy = (self.box[1] + self.box[3]) / 2
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            cx = self.mean[0]
            cy = self.mean[1]
            w = self.mean[2]
            h = self.mean[3]

        return np.array([cx, cy, w, h])

    @property
    def x1y1wh(self):
        # Get current position in bounding box format `(left top x, left top y, right bottom x, right bottom y)`.
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            w = self.mean[2]
            h = self.mean[3]

        return np.array([x1, y1, w, h])

    @property
    def x1y1x2y2(self):
        # Get current position in bounding box format `(left top x, left top y, right bottom x, right bottom y)`.
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            x2 = self.box[2]
            y2 = self.box[3]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            x2 = self.mean[0] + self.mean[2] / 2
            y2 = self.mean[1] + self.mean[3] / 2

        return np.array([x1, y1, x2, y2])
