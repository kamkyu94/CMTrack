import net
import copy
import torch
import pickle
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor


def get_pad_amount(img, detection):
    # Get image size
    img_h, img_w = img.shape[1], img.shape[2]

    # Get detection result
    det_x1 = int(round(detection[0]))
    det_y1 = int(round(detection[1]))
    det_x2 = int(round(detection[2]))
    det_y2 = int(round(detection[3]))

    # Get padding amount for each side
    l_pad = abs(det_x1) if det_x1 < 0 else 0
    t_pad = abs(det_y1) if det_y1 < 0 else 0
    r_pad = abs(det_x2 - img_w) if det_x2 > img_w else 0
    b_pad = abs(det_y2 - img_h) if det_y2 > img_h else 0

    return l_pad, r_pad, t_pad, b_pad


# Resize a rectangular image to a padded rectangular
def letterbox(img, patch_size, color=0):
    # shape = [height, width]
    shape = img.shape[1:]
    ratio = min(float(patch_size[0]) / shape[0], float(patch_size[1]) / shape[1])

    # new_shape = [channel, height, width]
    new_shape = (round(shape[0] * ratio), round(shape[1] * ratio))

    # Padding
    dh = (patch_size[0] - new_shape[0]) / 2
    dw = (patch_size[1] - new_shape[1]) / 2

    # Top, bottom, left, right
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border, padded rectangular
    img = F.interpolate(img.unsqueeze(0), size=new_shape, mode='bicubic').squeeze()
    img = F.pad(img, (left, right, top, bottom), mode='constant', value=color)

    return img


def feat_ext(dataset_name, dataset_type, domain='off'):
    # Set path
    data_path = '../../dataset/%s' % dataset_name
    det_pickle_path = '../outputs/1. det/%s.pickle' % dataset_type
    det_feat_pickle_path = '../outputs/2. det_feat/%s_%s_ghost.pickle' % (dataset_type, domain)

    # Initialization 1
    to_tensor = ToTensor()

    # Initialization 2
    new_size = [384, 128]
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])

    # Initialization 3
    encoder, _, _ = net.load_net(2820, net_type='resnet50', neck=0, pretrained_path='./weights/resnet50_Market.pth',
                                 red=4, add_distractors=0, pool='max')
    encoder = encoder.cuda()

    # Turn off on the fly domain adaptation
    if domain == 'off':
        encoder.eval()

    # Open detection result
    with open(det_pickle_path, 'rb') as f:
        det_results = pickle.load(f)

    # Copy
    det_feat_results = copy.deepcopy(det_results)

    # Start
    for vid_name in det_results.keys():
        for frame_id in det_results[vid_name].keys():
            # If there is no detection
            if det_results[vid_name][frame_id] is None:
                continue

            # Set image path
            image_path = data_path + vid_name + '/img1/'
            image_path += '%06d.jpg' % frame_id if 'MOT' in data_path else '%08d.jpg' % frame_id

            # Read image
            image = to_tensor(Image.open(image_path).convert("RGB"))

            # Generate patches
            patches = []
            for det_result in det_results[vid_name][frame_id]:
                # Clamp detection result
                x1 = int(round(max(det_result[0], 0)))
                y1 = int(round(max(det_result[1], 0)))
                x2 = int(round(min(det_result[2], image.shape[2])))
                y2 = int(round(min(det_result[3], image.shape[1])))

                # Get patch
                patch = image[:, y1:y2, x1:x2]

                # Get pad amount
                left_pad, right_pad, top_pad, bot_pad = get_pad_amount(image, det_result)

                # Pad if there are something to pad
                if left_pad + right_pad + top_pad + bot_pad > 0:
                    zero_pad = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bot_pad))
                    patch = zero_pad(patch)

                # Resize
                patch = F.interpolate(patch.unsqueeze(0), size=new_size, mode='bilinear').squeeze()

                # Apply transforms
                patches.append(transform(patch))

            # Stack
            if len(patches) > 1:
                patches = torch.stack(patches, 0).cuda()
            else:
                patches = patches[0][None].cuda()

            # Forward pass
            with torch.no_grad():
                _, feats = encoder(patches, output_option='plain')
            feats = feats.squeeze() if len(feats) > 1 else feats.squeeze()[None]
            feats = feats.detach().cpu().numpy()

            # Save
            det_feat_results[vid_name][frame_id] = np.concatenate([det_feat_results[vid_name][frame_id], feats], axis=1)

            # Logging
            print('%s %s %d finished' % (dataset_type, vid_name, frame_id))

    # Save
    with open(det_feat_pickle_path, 'wb') as handle:
        pickle.dump(det_feat_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    feat_ext('MOT17/train/', 'mot17_val_half', 'on')
    feat_ext('MOT20/train/', 'mot20_val_half', 'on')
    feat_ext('DanceTrack/val/', 'dance_val_dc', 'on')
