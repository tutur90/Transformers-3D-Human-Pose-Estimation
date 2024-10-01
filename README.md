# Transformmers for 3D Human Pose Estimation from 2D Human Pose

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> 

## [Report](./Report.pdf)

<!-- (<div style="width: 60px; height: 20px; background-color: #550040; border-radius: 5px; solid #000;   display: flex; justify-content: center;"><p style="color: white;">Report</p></div> -->

## Disclamer

This project, including the code and the report, was developed for educational purposes. Some information may be inaccurate or incomplete. If you identify any errors or areas for improvement, please feel free to contact us at arthur.garon@gmail.com. We would greatly appreciate your feedback! ; )


## Environment
The project is developed under the following environment:
- Python 3.8.10
- PyTorch 2.0.0
- CUDA 12.2

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 
## Dataset
### Human3.6M
#### Preprocessing
1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:

**For MotionAGFormer-Base and MotionAGFormer-Large**:
```text
python h36m.py  --n-frames 243
```

**For MotionAGFormer-Small**:
```text
python h36m.py --n-frames 81
```

**For MotionAGFormer-XSmall**:
```text
python h36m.py --n-frames 27
```

#### Visualization
Run the following command in the `data/preprocess` directory (it expects 243 frames):
```text
python visualize.py --dataset h36m --sequence-number <AN ARBITRARY NUMBER>
```
This should create a gif file named `h36m_pose<SEQ_NUMBER>.gif` within `data` directory.

### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.
#### Visualization
Run it same as the visualization for Human3.6M, but `--dataset` should be set to `mpi`.
## Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python train.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/h36m`. You can also use [weight and biases](wandb.ai) for logging the training and validation error by adding `--use-wandb` at the end. In case of using it, you can set the name using `--wandb-name`. e.g.:

```
python train.py --config configs/h36m/MotionAGFormer-base.yaml --use-wandb --wandb-name MotionAGFormer-base
```
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/mpi`. Like Human3.6M, weight and biases can be used.

## Demo
Our demo is a modified version of the one provided by [MHFormer](https://github.com/Vegetebird/MHFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.

Run the command below:
```
python demo/vis.py --video sample_video.mp4
```
Sample demo output:

<p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>

## Acknowledgement
Our code refer to the following repositorie:

- [MotionAGFormer](https://github.com/taatiteam/motionagformer)

We thank the authors for releasing their codes.

