import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer


def make_parser():
    parser = argparse.ArgumentParser("Track")

    # Data args
    parser.add_argument("--pickle_path", type=str, default="../outputs/1. det/")
    parser.add_argument("--output_path", type=str, default="../outputs/2. det_feat/")
    parser.add_argument("--data_path", type=str, default="../../dataset/")
    parser.add_argument("--dataset", type=str, default="mot17")

    # Else
    parser.add_argument("--seed", type=float, default=10000)

    return parser


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    dataset = args.dataset

    if dataset == 'mot17':
        pickle_path = args.pickle_path + 'mot17_test.pickle'
        output_path = args.output_path + 'mot17_test_fast.pickle'
        data_path = args.data_path + 'MOT17/test/'
    elif dataset == 'mot20':
        pickle_path = args.pickle_path + 'mot20_test.pickle'
        output_path = args.output_path + 'mot20_test_fast.pickle'
        data_path = args.data_path + 'MOT20/test/'
    else:
        pickle_path = args.pickle_path + 'dance_test.pickle'
        output_path = args.output_path + 'dance_test_fast.pickle'
        data_path = args.data_path + 'DanceTrack/test/'

    # Get encoder
    embedder = EmbeddingComputer(dataset=dataset, test_dataset=True, grid_off=True)

    # Read
    with open(pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Feature extraction
    for vid_name in detections.keys():
        img_names = os.listdir(data_path + vid_name + '/img1/')
        img_names.sort()

        for img_name in img_names:
            # Read image
            frame_id = int(img_name.split('.')[0])
            img = cv2.imread(data_path + vid_name + '/img1/' + img_name)

            # Initialize
            tag = f"{vid_name}:{frame_id}"
            det = detections[vid_name][frame_id]

            # Get features
            if det is not None:
                emb = embedder.compute_embedding(img, det[:, :4], tag)
                detections[vid_name][frame_id] = np.concatenate([det, emb], axis=1)

            # Logging
            print(vid_name, frame_id)

    # Save
    with open(output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
