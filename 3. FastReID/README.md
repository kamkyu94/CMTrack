## Model Zoo
Save weights files under "./weights/"
  - [mot17_sbs_S50.pth](https://drive.google.com/file/d/1XpC27lWBL-wSf-9ceh2fsnAQeOGlirig/view?usp=drive_link)
  - [mot20_sbs_S50.pth](https://drive.google.com/file/d/1UiVMWtGf-ktGRUFRfp2L5UaAiUk8jZCR/view?usp=drive_link)
  - [dance_sbs_S50.pth](https://drive.google.com/file/d/1JZj__3I94X60s6JLTWbHAB-nR3pU0CCp/view?usp=drive_link)


## Run
Detection + feature extraction results will be created under "../outputs/2. det_feat/" as pickle files
```
# For MOT17 test
python ext_feats.py --dataset "mot20"

# For MOT20 test
python ext_feats.py --dataset "mot20"

# For DanceTrack test
python ext_feats.py --dataset "dancetrack"
```


## Reference
  - https://github.com/JDAI-CV/fast-reid
