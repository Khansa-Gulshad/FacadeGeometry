# Building Area (Width*Height) estimation



## Code Structure
```
Gdańsk, Poland/                 # segmentation output
  save_rgb/           
    imgs                        # rgb images
  seg                           # .npz files having label for each class
  seg_3class_vis                # 3 classes segmented images
  seg_full_overlay              # segmented overlay with rgb
  seg_full_vis                  # all classes (Cityscapes) segmented images
NeurVPS scripts/                # code keep for reproducibility of NeurVPS
  prep_wflike.py                # JPG -> 512×512 PNG + split lists
  run_eval.py                   # wrapper to run NeurVPS eval.py and dump predictions
  vpt_postprocess.py            # 3D→2D VP transform (FOV, ordering, scaling)

config/
    estimation_config.ini
data/                           # default folder for placing the data
  images/                       # original street-view JPG images (inputs)
  lines/                        # .npz and .svg files from LCNN model
  vpts/                         # outputs from NeurVPS + post-processing
    000000.npz ...              # raw model outputs (contain vpts_pd)
    .....png                    # optional visualization overlays (PNG)
    su3_error.npz               # AA curve file (unused for you; safe to ignore)
    json/                       # per-image 2D VP results (ordered + scaled)
      <image_stem>.json
  wflike/                       # "wireframe-like" dataset view for NeurVPS
    valid.txt                   # split list (relative paths like A/xxx.png)
    test.txt
    val.txt
  

external/                       # third-party code
  neurvps/                      # NeurVPS repo at commit 72d9502

modules/
  process_data.py              # to fetch and segment street view images
  road_network.py              # to fetch road network
  segmentation.py              # segmentation classes
  
sihe/                           # SIHE model + configs used
  vps_models/
    neurvps_sihe_checkpoint.pth.tar   # SIHE retrained model
    config.yaml                       # SIHE’s config file that we used
```


##  Apptainer environment (used for GPU eval)

- Module: `trytonp/apptainer/1.3.0`
- SIF built/pulled from: `nvcr.io/nvidia/pytorch:21.11-py3`
- In-container CUDA toolkit: 11.5 (`nvcc --version`)
- PyTorch: 1.11 (CUDA build 11.5)
- Run with `apptainer exec --nv ... pytorch_21.11-py3.sif`

We bind:  
- `$REPO/external/neurvps` -> /w/[neurvps](https://github.com/zhou13/neurvps)
- `$REPO/sihe`         -> /w/[SIHE](https://github.com/yzre/SIHE?tab=readme-ov-file)
- `$REPO/LCNN`         -> /w/[LCNN](https://github.com/zhou13/lcnn)

We install in-container (user-space):  
`docopt "tensorboardX<3" "protobuf<4" yacs pyyaml tqdm opencv-python-headless scikit-image scipy ninja pillow imageio numpy`
