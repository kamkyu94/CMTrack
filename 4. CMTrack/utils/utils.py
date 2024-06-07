import trackeval
import numpy as np


def set_parameters(args, dataset, mode):
    # Set properly for each dataset
    if dataset == 'MOT17':
        # Path
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'mot17_val_half_ghost.pickle'
            args.data_path = args.data_dir + 'MOT17/train/'
        else:
            args.pickle_path = args.pickle_dir + 'mot17_test_fast.pickle'
            args.data_path = args.data_dir + 'MOT17/test/'

        # Ours setting
        args.interval = 0.20
        args.new_track_thresh = 0.70

    elif dataset == 'MOT20':
        # Path
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'mot20_val_half_ghost.pickle'
            args.data_path = args.data_dir + 'MOT20/train/'
        else:
            args.pickle_path = args.pickle_dir + 'mot20_test_fast.pickle'
            args.data_path = args.data_dir + 'MOT20/test/'

        # Ours setting
        args.interval = 0.20
        args.new_track_thresh = 0.50

    else:
        # Path
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'dance_val_ghost.pickle'
            args.data_path = args.data_dir + 'DanceTrack/val/'
        else:
            args.pickle_path = args.pickle_dir + 'dance_test_fast.pickle'
            args.data_path = args.data_dir + 'DanceTrack/test/'

        # Ours setting
        args.interval = 0.20
        args.new_track_thresh = 0.70


def track_history_to_dict(tracks):
    track_dict = {}
    for track in tracks:
        # Initialization (7 + 2048, 7 for detection info, 2048 for feature)
        track_result = np.zeros((len(track.history), 7 + track.history[-1][5].shape[-1]))

        # Gather
        for hdx, hist in enumerate(track.history):

            track_result[hdx, 0] = hist[0]  # Frame ID
            track_result[hdx, 1] = track.track_id  # Track ID
            track_result[hdx, 2:6] = hist[1]  # x1y1x2y2
            track_result[hdx, 6] = hist[2]  # Conf
            track_result[hdx, 7:] = hist[5]  # Feature

        # Save
        track_dict[track.track_id] = track_result

    return track_dict


def write_results(filename, results):
    # Set save format
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'

    # Open file
    f = open(filename, 'w')

    # Write
    for frame_id, track_ids, x1y1whs, scores in results:
        for track_id, x1y1wh, score in zip(track_ids, x1y1whs, scores):
            # Get box
            x1, y1, w, h = x1y1wh

            # Generate line to write
            line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                      w=round(w, 1), h=round(h, 1), s=round(score, 2))

            # Write
            f.write(line)

    # Close
    f.close()


def evaluate(args, trackers_to_eval, dataset):
    # Set evaluation configurations
    eval_config = {'USE_PARALLEL': True,
                   'NUM_PARALLEL_CORES': 8,
                   'BREAK_ON_ERROR': True,
                   'RETURN_ON_ERROR': False,
                   'LOG_ON_ERROR': '../outputs/error_log.txt',

                   'PRINT_RESULTS': False,
                   'PRINT_ONLY_COMBINED': False,
                   'PRINT_CONFIG': False,
                   'TIME_PROGRESS': False,
                   'DISPLAY_LESS_PROGRESS': True,

                   'OUTPUT_SUMMARY': False,
                   'OUTPUT_EMPTY_CLASSES': False,
                   'OUTPUT_DETAILED': False,
                   'PLOT_CURVES': False}

    dataset_config = {'GT_FOLDER': args.data_path,
                      'TRACKERS_FOLDER': args.output_dir,
                      'OUTPUT_FOLDER': None,
                      'TRACKERS_TO_EVAL': [trackers_to_eval],
                      'CLASSES_TO_EVAL': ['pedestrian'],
                      'BENCHMARK': dataset if 'MOT' in dataset else 'MOT17',
                      'SPLIT_TO_EVAL': 'val',
                      'INPUT_AS_ZIP': False,
                      'PRINT_CONFIG': False,
                      'DO_PREPROC': True,
                      'TRACKER_SUB_FOLDER': '',
                      'OUTPUT_SUB_FOLDER': '',
                      'TRACKER_DISPLAY_NAMES': None,
                      'SEQMAP_FOLDER': None,
                      'SEQMAP_FILE': './trackeval/seqmap/%s/val.txt' % dataset.lower(),
                      'SEQ_INFO': None,
                      'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt_val_half.txt'
                                       if 'MOT' in dataset else '{gt_folder}/{seq}/gt/gt.txt',
                      'SKIP_SPLIT_FOL': True}

    # Set configuration
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Get
    hota = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
    idf1 = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']
    mota = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
    assa = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
    deta = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()

    # Print
    print('%.3f %.3f %.3f %.3f %.3f' % (hota * 100, idf1 * 100, mota * 100, assa * 100, deta * 100), flush=True)
