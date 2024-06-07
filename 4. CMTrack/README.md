## Run
Tracking results will be created under "../outputs/3. track/"

```
# For MOT17 validation
python track.py --dataset "MOT17" --mode "val"

# For MOT17 test
python run.py --dataset "MOT17" --mode "test"

# For MOT20 validation
python run.py --dataset "MOT20" --mode "val"

# For MOT20 test
python run.py --dataset "MOT20" --mode "test"

# For DanceTrack validation
python run.py --dataset "DanceTrack" --mode "val"

# For DanceTrack test
python run.py --dataset "DanceTrack" --mode "test"
```


## Reference
  - https://github.com/NirAharon/BoT-SORT
  - https://github.com/dyhBUPT/StrongSORT
  - https://github.com/gerardmaggiolino/deep-oc-sort
