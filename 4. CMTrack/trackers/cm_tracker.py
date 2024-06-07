from trackers.cmc import *
from trackers.matching import *


class CMTrack(object):
    def __init__(self, args, vid_name):
        # Initialize
        self.args = args
        self.max_time_lost = args.max_time_lost

        # Initialize
        self.tracks = []
        self.finished = []
        self.frame_id = 0
        self.counter = TrackCounter()

        # Set global motion compensation model
        self.cmc = CMC(vid_name)

    def update(self, detections):
        # ==============================================================================================================
        # Update frame id
        self.frame_id += 1

        # Encode detections with Track
        detections = [Track(self.args, d) for d in detections]

        # Split already tracked and new tracks
        tracks_tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        tracks_new = [t for t in self.tracks if t.state == TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(tracks_tracked_lost, warp_matrix)
        apply_cmc(tracks_new, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in tracks_tracked_lost]
        [t.predict() for t in tracks_new]

        # ==============================================================================================================
        # Association between (tracked and lost tracks) & (detections)
        u_tracks, u_detections = cascade_match(tracks_tracked_lost, detections, self.frame_id, self.args.interval)

        # Mark "lost" to unmatched tracks
        for u_t in u_tracks:
            u_t.mark_lost()

        # ==============================================================================================================
        # Association between (new tracks) & (left detections)
        u_tracks, u_detections = cascade_match(tracks_new, u_detections, self.frame_id, self.args.interval)

        # Mark "remove" to unmatched tracks
        for u_t in u_tracks:
            u_t.mark_removed()

        # Init new tracks
        for u_d in u_detections:
            if self.args.new_track_thresh <= u_d.score:
                u_d.initiate(self.frame_id, self.counter)
                self.tracks.append(u_d)

        # ==============================================================================================================
        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()
                self.finished.append(track)

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Gather tracked tracks
        tracked = [t for t in self.tracks if t.state == TrackState.Tracked]

        return tracked

    def update_no_detections(self):
        # Update frame id
        self.frame_id += 1

        # Only maintain already tracked and new tracks, Drop all the new tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(self.tracks, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in self.tracks]

        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Gather tracked tracks
        tracked = [t for t in self.tracks if t.state == TrackState.Tracked]

        return tracked
