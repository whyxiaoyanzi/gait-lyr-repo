# GaitAnalysisVideo

## Gait Analysis Video Processing Programs
Video processing pipeline programs for
  - Human body detection and annotation
  - Gait event detection and parameter computation
  - Human face detection and blurring
  - Parameter results upload to database
  - Push notification to inform video submitter on completion of processing

## Installation
This code is tested on Ubuntu 22.04.1 LTS (GNU/Linux 5.15.0-53-generic x86_64) with:
  - Singularity 3.10
  - CUDA 12.1
  - Nvidia driver 530.30 on RTX A5000 (Ampere) GPU

### Singularity container
Create a Singularity image (inamed opp20.sif) using the files in singularity folder for compilation and runtime use

### Openpose Python Library
Create a build folder and compile Openpose from source uing the Singularity image environment

## Directory Structure
The programs assume the following directory structure:
- /data/openpose/gait : base for this git files
- /data/openpose/opp20.sif : Singularity image file
- /data/openpose/openpose : base for Openpose git files
- /data/openpose/openpose/build : build folder for Openpose compiled codes
- /data/pdlogger : base for web application that initiate the video processing


## Usage
Once cloned to a local folder and with all required software installed,
  1. wsubmit.sh to be launched by submit queue worker
  2. adhoc scripts can be copied locally for edit before run

## Support
For enquiries, contact jslow@nus.edu.sg

## Acknowledgment
This code is mostly obtained from https://gitlab.com/laukinon/gait-postanalysis.git prepared by Lau Kin On from Electrical and Computer Engineering Department, College of Design and Engineering, National University of Singapore.

## License
This software uses Openpose, which is for academic or non-profit organization noncommercial research use only. Ref: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE

## Project status
Active

