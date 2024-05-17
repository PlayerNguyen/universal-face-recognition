# Introduction

The state-of-the-art machine learning face recognition pipeline implemented on Python with **FaceNet** for my a research on Nvidia Jetson Nano and Nvidia Jetson Orin Nano.

## Architecture

The main architecture of this face recognition application based on [FaceNet]()

<!-- Todo: add citation here -->

# Requirements

- `Python >= 3.6`
- `pytorch and torchvision` (with or without CUDA)

# Usage

## Input a face into dataset (using webcam)

To input a face into a dataset, execute this python script

```
python3 ./input.py
```

Then, wait for the device evaluate and extract features from your face and store into a face datasets.

<!-- For more detail, read  -->

## Inference a face (using webcam)

To run inference mode, using

```
python3 ./inference.py
```
