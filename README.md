# MPAFNet-Multi-Path-Parametric-Attention-Fusion-Network-for-Drone-RF-Fingerprint-Identification
MPAFNet: Multi-Path Parametric Attention Fusion Network for drone RF fingerprint ID. Achieves 100%, 99.7%, 99.5% accuracy on 2/4/10-class tasks via TabNet-SA + EffNet-Slim + learnable fusion. Achieved high-precision RF signal identification for UAVs.

# Project Overview
This project focuses on end-to-end classification of drone RF signals.
The workflow consists of:

Pre-processing the raw RF data.

Generating time-frequency spectrograms using the ECSG method.

Training and evaluating a Multi-Path-Parametric-Attention-Fusion-Network (RF-TCNet) that combines the strengths of TabNet-SA, EffNet-Slim and parametric attention fusion architectures.

# Datasets
Dataset description Public datasets DroneRF.

# Installation
Clone this repository and install dependencies:
git clone https://github.com/FAITHSHUNAA/MPAFNet-Multi-Path-Parametric-Attention-Fusion-Network-for-Drone-RF-Fingerprint-Identification.git

# Usage
Prepare spectrograms
Download the DroneRF datasets.
Generate spectrograms.
Split datasets
Use the provided dataset partitioning script to create train/validation/test splits.
Train the model
python RF_TCNet_Train_val.py (10 fold cross validation)


# Model Highlights
1. MPAFNet integrates four distinct feature paths for drone RF fingerprint identification, enhancing the comprehensiveness and discriminative capability of feature representation. 
2. TabNet-SA and EffNet-Slim modules to process statistical features and spectrograms, respectively, reducing model complexity while maintaining performance. 
3. PAF strategy adaptively weights the outputs of the four paths, dynamically adjusts their importance, and fully utilizes the complementary information of different paths. 
4. On the DroneRF dataset, MPAFNet achieved average accuracies of 100.00%, 99.70%, and 99.54% for 2-class, 4-class, and 10-class tasks, respectively, demonstrating significant advantages over existing methods. 

# License
This project is released under the MIT License.

# Acknowledgements
We thank the authors of the DroneRF datasets for making their data publicly available.
