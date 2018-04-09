## Main Configuration

GPU: 2 x GTX1080

BaseNet: ResNet50 + FPN

BASE_LR: 0.0025

GAMMA: 0.1

MAX_ITER: 96000

STEPS: [0, 72000]

#### Scale fixed according to GPU memory

Train SCALES: 800~960

Test SCALES: 960

MAX_SIZE: 1920


## Instance Seg Results

| ROI-Align    | training data               | testing data              | AP      | AP@0.5  | training time |
|--------------|-----------------------------|---------------------------|---------|---------|---------------|
| push version | fine_instanceonly_seg_train | fine_instanceonly_seg_val | 31.0    | 58.3    | 0.73s / iter  |
| pull version | fine_instanceonly_seg_train | fine_instanceonly_seg_val | 30.0    | 57.1    | 2.19s / iter  |

#### push version:

| what           |             AP     |   AP_50%  |
|----------------|--------------------|-----------|
| person         |          0.297     |    0.627  |
| rider          |          0.242     |    0.625  |
| car            |          0.499     |    0.751  |
| truck          |          0.322     |    0.497  |
| bus            |          0.508     |    0.718  |
| train          |          0.276     |    0.529  |
| motorcycle     |          0.161     |    0.414  |
| bicycle        |          0.178     |    0.501  |
|----------------|--------------------|-----------|
| average        |          0.310     |    0.583  |

#### pull version:

| what           |             AP     |   AP_50%  |
|----------------|--------------------|-----------|
| person         |          0.297     |    0.629  |
| rider          |          0.234     |    0.626  |
| car            |          0.496     |    0.749  |
| truck          |          0.296     |    0.452  |
| bus            |          0.498     |    0.694  |
| train          |          0.228     |    0.488  |
| motorcycle     |          0.174     |    0.429  |
| bicycle        |          0.174     |    0.503  |
|----------------|--------------------|-----------|
| average        |          0.300     |    0.571  |

## Det Results

| ROI-Align    | AP     | AP50   | AP75   | APs    | APm    | APl    |
|--------------|-----------------|--------|--------|--------|--------|
| push version | 0.3593 | 0.6183 | 0.3607 | 0.1464 | 0.3541 | 0.5446 |
| pull version | 0.3468 | 0.5996 | 0.3522 | 0.1334 | 0.3553 | 0.5173 |

