import lap
from trackers.track import *
from cython_bbox import bbox_overlaps as bbox_ious


def calc_iou(a_x1y1x2y2, b_x1y1x2y2):
    iou = bbox_ious(a_x1y1x2y2, b_x1y1x2y2)
    return iou


def calc_d_iou(a_x1y1x2y2, b_x1y1x2y2):
    # Expand dimension
    a_x1y1x2y2 = np.expand_dims(a_x1y1x2y2, 1)
    b_x1y1x2y2 = np.expand_dims(b_x1y1x2y2, 0)

    # Get coordinate
    x1 = np.maximum(a_x1y1x2y2[..., 0], b_x1y1x2y2[..., 0])
    y1 = np.maximum(a_x1y1x2y2[..., 1], b_x1y1x2y2[..., 1])
    x2 = np.minimum(a_x1y1x2y2[..., 2], b_x1y1x2y2[..., 2])
    y2 = np.minimum(a_x1y1x2y2[..., 3], b_x1y1x2y2[..., 3])

    # Calculate IoU
    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)
    inter = w * h
    union = ((a_x1y1x2y2[..., 2] - a_x1y1x2y2[..., 0]) * (a_x1y1x2y2[..., 3] - a_x1y1x2y2[..., 1])
             + (b_x1y1x2y2[..., 2] - b_x1y1x2y2[..., 0]) * (b_x1y1x2y2[..., 3] - b_x1y1x2y2[..., 1]) - inter)
    iou = inter / union

    # Get coordinate
    cx1 = (a_x1y1x2y2[..., 0] + a_x1y1x2y2[..., 2]) / 2
    cy1 = (a_x1y1x2y2[..., 1] + a_x1y1x2y2[..., 3]) / 2
    cx2 = (b_x1y1x2y2[..., 0] + b_x1y1x2y2[..., 2]) / 2
    cy2 = (b_x1y1x2y2[..., 1] + b_x1y1x2y2[..., 3]) / 2

    # Calculate euclidean distance between center points of two boxes
    center_dist = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    # Get coordinate
    x1 = np.minimum(a_x1y1x2y2[..., 0], b_x1y1x2y2[..., 0])
    y1 = np.minimum(a_x1y1x2y2[..., 1], b_x1y1x2y2[..., 1])
    x2 = np.maximum(a_x1y1x2y2[..., 2], b_x1y1x2y2[..., 2])
    y2 = np.maximum(a_x1y1x2y2[..., 3], b_x1y1x2y2[..., 3])

    # Calculate outer diagonal length
    outer_diag = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Calculate DIoU (-1 ~ 1)
    d_iou = iou - center_dist / outer_diag

    # Clip
    d_iou = np.clip(d_iou, a_min=0., a_max=1.)

    return d_iou


def calc_confluence(a_x1y1x2y2, b_x1y1x2y2):
    # Calculate minimum and maximum values
    min_x = np.minimum(a_x1y1x2y2[:, 0:1], b_x1y1x2y2[:, 0:1].T)
    min_y = np.minimum(a_x1y1x2y2[:, 1:2], b_x1y1x2y2[:, 1:2].T)
    max_x = np.maximum(a_x1y1x2y2[:, 2:3], b_x1y1x2y2[:, 2:3].T)
    max_y = np.maximum(a_x1y1x2y2[:, 3:4], b_x1y1x2y2[:, 3:4].T)

    # Calculate normalized coordinates
    a_x1, a_y1 = (a_x1y1x2y2[:, 0:1] - min_x) / (max_x - min_x), (a_x1y1x2y2[:, 1:2] - min_y) / (max_y - min_y)
    a_x2, a_y2 = (a_x1y1x2y2[:, 2:3] - min_x) / (max_x - min_x), (a_x1y1x2y2[:, 3:4] - min_y) / (max_y - min_y)
    b_x1, b_y1 = (b_x1y1x2y2[:, 0:1].T - min_x) / (max_x - min_x), (b_x1y1x2y2[:, 1:2].T - min_y) / (max_y - min_y)
    b_x2, b_y2 = (b_x1y1x2y2[:, 2:3].T - min_x) / (max_x - min_x), (b_x1y1x2y2[:, 3:4].T - min_y) / (max_y - min_y)

    # Calculate manhattan distances (confluence)
    md_x1, md_x2, md_y1, md_y2 = abs(a_x1 - b_x1), abs(a_x2 - b_x2), abs(a_y1 - b_y1), abs(a_y2 - b_y2)
    confluence = 1 - (md_x1 + md_x2 + md_y1 + md_y2)

    # Clip
    confluence = np.clip(confluence, a_min=0., a_max=1.)

    return confluence


def iou_similarity(a_tracks, b_tracks, iou_func=calc_d_iou):
    # Initialization
    a_x1y1x2y2 = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_x1y1x2y2 = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU
    if len(a_x1y1x2y2) == 0 or len(b_x1y1x2y2) == 0:
        sim_matrix = np.zeros((len(a_x1y1x2y2), len(b_x1y1x2y2)), dtype=np.float64)
    else:
        sim_matrix = iou_func(a_x1y1x2y2, b_x1y1x2y2)

    return sim_matrix


def embedding_similarity(tracks, detections):
    # Initialization
    sim_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if len(tracks) == 0 or len(detections) == 0:
        return sim_matrix

    # Calculate cosine similarity
    t_feat = np.concatenate([t.feat for t in tracks], axis=0)
    d_feat = np.concatenate([d.feat for d in detections], axis=0)
    sim_matrix = np.dot(t_feat, d_feat.T)

    # Clip
    sim_matrix = np.clip(sim_matrix, a_min=0., a_max=1.)

    return sim_matrix


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def get_mask(objects, interval):
    # Initialize
    scores = np.array([obj.score for obj in objects])
    intervals = np.arange(0.1, 1.0, interval)

    # Get mask
    mask = []
    for idx in range(len(intervals)):
        if idx < len(intervals) - 1:
            mask.insert(0, (intervals[idx] <= scores) & (scores < intervals[idx + 1]))
        else:
            mask.insert(0, (intervals[idx] <= scores))

    return mask


def cascade_match(tracks, detections, frame_id, interval):
    # Get masks
    det_mask = get_mask(detections, interval) if len(detections) > 0 else []

    # Initialize
    idx = 0
    u_tracks, u_detections = tracks, []
    match_threshes = np.arange(0.9, 0.0, -interval)

    if len(tracks) != 0:
        for ddx, dm in enumerate(det_mask):
            # Search and merge detections
            det_idx = np.argwhere(dm)
            det_ = [detections[idd[0]] for idd in det_idx] + u_detections

            # Get IoU similarities
            iou_sim = iou_similarity(u_tracks, det_)

            # Get Cosine similarities
            cos_sim = embedding_similarity(u_tracks, det_)

            # # Geometric mean (If we do not use CMF)
            # iou_sim = 1 - np.sqrt((1 - iou_sim) * (1 - cos_sim * (iou_sim > 0)))

            # Confidence-aware metric fusion
            w_c = np.array([d.score for d in det_]).reshape(1, len(det_)) * (iou_sim > 0)
            iou_sim = (iou_sim + w_c * cos_sim) / (1 + w_c)

            # Linear assignment
            matches, u_track_, u_det_ = linear_assignment(1 - iou_sim, match_threshes[idx])
            idx += 1

            # Update
            for t, d in matches:
                u_tracks[t].update(frame_id, det_[d])

            # Select
            u_tracks = [u_tracks[t] for t in u_track_]
            u_detections = [det_[d] for d in u_det_]

    else:
        u_detections = detections

    return u_tracks, u_detections
