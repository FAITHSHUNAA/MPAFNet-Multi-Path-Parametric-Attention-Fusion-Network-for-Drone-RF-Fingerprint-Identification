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
git clone https://github.com/FAITHSHUNAA/RF-TCNet-A-Lightweight-Topology-Compression-Network-for-Drone-RF-Fingerprint-Identification.git

# Usage
Prepare spectrograms
Download the DroneRF and DroneRFa datasets.
Apply the ECSG preprocessing to generate spectrograms.
Split datasets
Use the provided dataset partitioning script to create train/validation/test splits.
Train the model
python RF_TCNet_Train.py
python RF_TCNet_Test.py

# Model Highlights
Topology Compression: Reduces redundant connections while retaining key spectral features.
Dynamic Frequency Attention: Adapts to variations in UAV transmission characteristics.
Edge-Efficient Design: Achieves low latency and small model size without sacrificing accuracy.

# License
This project is released under the MIT License.
See the LICENSE.

# Acknowledgements
We thank the authors of the DroneRF and DroneRFa datasets for making their data publicly available.
