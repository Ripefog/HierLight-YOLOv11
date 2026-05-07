# HierLight-YOLO: Hierarchical Lightweight YOLO for UAV Object Detection

Re-implementation of **HierLight-YOLO** (arXiv:2509.22365) built on top of YOLOv11-pt, optimized for UAV imagery with small object detection.

## What's New

HierLight-YOLO introduces 4 key modules compared to YOLOv11:

| Module | Description | Benefit |
|---|---|---|
| **IRDCB** | Inverted Residual Depthwise Convolution Block | Replaces CSP, reduces params by 22.1% |
| **LDown** | Lightweight Downsample (Depthwise Conv) | Replaces Conv stride=2, reduces params by 11.4% |
| **HFCC** | Hierarchical Feature Channel Compression | 1x1 conv compression before feature fusion |
| **HEPAN** | Hierarchical Extended PAN | 4-scale feature fusion with dense skip connections |

### Architecture Comparison

| | YOLOv11 | HierLight-YOLO |
|---|---|---|
| Backbone | DarkNet (CSP + Conv) | IRDCB + LDown |
| Neck | DarkFPN (3-scale) | HEPAN (4-scale) |
| Head | 3-scale (P3/P4/P5) | **4-scale (P2/P3/P4/P5)** |
| PSA Attention | Yes | Removed |
| Strides | [8, 16, 32] | **[4, 8, 16, 32]** |
| Max resolution | 80x80 | **160x160 (P2 head)** |

### Model Variants

| Model | Params (M) | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|
| HierLight-YOLO-N | 1.36 | - | - |
| HierLight-YOLO-S | 4.43 | - | - |
| HierLight-YOLO-M | 10.36 | - | - |

## Installation

```bash
conda create -n YOLO python=3.10.10
conda activate YOLO
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python PyYAML tqdm thop
```

## Train

```bash
# Train from scratch
python main.py --model hierlight_yolo_s --train --epochs 600

# Train with partial pretrained weights (transfer learning from YOLOv11-S)
python main.py --model hierlight_yolo_s --train --epochs 600 --weights weights/yolo_v11_s.pt
```

Configure dataset path in `main.py` (`data_dir` variable).

### Available Models

`yolo_v11_n`, `yolo_v11_t`, `yolo_v11_s`, `yolo_v11_m`, `yolo_v11_l`, `yolo_v11_x`, `hierlight_yolo_n`, `hierlight_yolo_s`, `hierlight_yolo_m`

## Test

```bash
python main.py --model hierlight_yolo_s --test
```

## Dataset Structure

Supports VisDrone2019 (10 classes) and COCO format:

```
dataset/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/
│   ├── train2017/
│   └── val2017/
├── train2017.txt
└── val2017.txt
```

### VisDrone2019 Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

## Project Structure

```
├── main.py              # Entry point: train/test/profile
├── nets/
│   └── nn.py            # Network architecture (YOLOv11 + HierLight-YOLO)
├── utils/
│   ├── args.yaml        # Hyperparameters and class names
│   ├── dataset.py       # Data loading and augmentation
│   └── util.py          # Loss, metrics, NMS, EMA, schedulers
└── weights/             # Saved models
```

## YOLOv11 Results (COCO)

| Version | Epochs | Box mAP | Download |
|:---:|:---:|---:|---:|
| v11_n | 600 | 38.6 | [Model](./weights/best.pt) |
| v11_n* | - | 39.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_n.pt) |
| v11_s* | - | 46.5 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_s.pt) |
| v11_m* | - | 51.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_m.pt) |
| v11_l* | - | 53.0 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_l.pt) |
| v11_x* | - | 54.3 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_x.pt) |

`*` from original repository. In official YOLOv11, mask annotation is used for higher performance.

## References

- HierLight-YOLO paper: arXiv:2509.22365
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [YOLOv11-pt (original)](https://github.com/jahongir7174/YOLOv11-pt)
- [YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)
