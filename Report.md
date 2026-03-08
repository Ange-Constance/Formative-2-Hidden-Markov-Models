
# Human Activity Recognition Using Hidden Markov Models

**Course:** ML Techniques – Formative Assessment 2  
**Authors:** David & Ange  
**Date:** March 7, 2026  



# Project Overview

This project implements a **Human Activity Recognition (HAR)** system using **Hidden Markov Models (HMMs)** and motion sensor data collected from smartphones.

The goal is to classify everyday activities such as:

- Standing
- Walking
- Jumping
- Still (sitting/lying down)

The system uses **accelerometer and gyroscope data** recorded from smartphones to detect these activities. A practical use case for this project is **elderly care monitoring**, where smartphones can help detect unusual movements or potential fall risks in a non‑intrusive way.



# Background and Motivation

Human Activity Recognition (HAR) is widely used in health monitoring, fitness tracking, and assistive technologies. Our project focuses on a use case in **elderly care monitoring**. Elderly individuals living independently may require subtle monitoring to detect unusual movements or possible fall risks.

Instead of using specialized wearable devices, this project demonstrates that **smartphones alone can be used to monitor activities** through built‑in sensors. By classifying activities such as walking, standing, or stillness, caregivers can gain insights into daily movement patterns and detect abnormal behavior.



# Data Collection and Preprocessing

## Devices Used

Sensor data was collected from two iPhone devices:

- **Participant 1:** iPhone 11 (iOS 17.4.1)
- **Participant 2:** iPhone X (iOS 16.7.14)

Data was recorded using the **Sensor Logger app (v1.54)**.

## Sensors Recorded

- 3‑axis Accelerometer (m/s²)
- 3‑axis Gyroscope (rad/s)
- Sampling rate ≈ **100 Hz**

## Activities Recorded

Four activities were recorded:

1. Standing
2. Walking
3. Jumping
4. Still

Each participant recorded **10–12 files per activity**, with each recording lasting **5–10 seconds**.

**Total recordings:** 88 files  
**Minimum activity time per class:** > 2 minutes

## Data Cleaning

A Python preprocessing script:

- Verified CSV file integrity
- Standardized folder structures
- Extracted metadata (device info, sampling rate)
- Created **train/test dataset splits**

## Windowing Strategy

Sensor signals were divided into sliding windows:

- **Window size:** 50 samples (0.5 seconds)
- **Overlap:** 50%
- **Sampling rate:** 100 Hz

This window size captures enough motion patterns while maintaining good temporal resolution.

Minor sampling variations (95–105 Hz) were handled using **time-based windowing and interpolation**.

---

# Feature Extraction

A total of **27 features** were extracted from each sensor window.

## Time-Domain Features

For each sensor axis:

- Mean
- Standard Deviation
- Variance

These features capture signal stability and variability.

Dynamic activities (walking, jumping) show **higher variance**, while static activities show **lower variance**.

## Frequency-Domain Features

Using **Fast Fourier Transform (FFT)**:

- Dominant frequency for accelerometer axes

These features capture **periodic motion patterns**, such as walking gait cycles.

## Additional Features

- Signal Magnitude Area (SMA) for accelerometer and gyroscope
- Cross-axis correlation features

These help measure **movement intensity** and **coordination between axes**.

## Feature Normalization

All features were normalized using **Z‑score normalization**:

z = (x − μ) / σ

This ensures features are on comparable scales and improves Gaussian HMM performance.

---

# HMM Setup and Implementation

The system uses **Gaussian Hidden Markov Models**.

## Model Configuration

- **4 models** (one per activity)
- **4 hidden states per model**
- **Diagonal covariance matrices**
- Implemented using **hmmlearn**

## Training Algorithm

Training uses the **Baum–Welch algorithm** (Expectation-Maximization).

Training stops when:

- Log-likelihood improvement < 0.01
- OR max iterations reached

This ensures stable convergence.

## Prediction

Prediction uses the **Viterbi algorithm**.

Process:

1. Extract features from a test sequence
2. Compute likelihood for each activity HMM
3. Select activity with highest likelihood



# Evaluation on Unseen Data

The model was tested on **completely unseen recordings**.

## Metrics Used

- Accuracy
- Sensitivity (Recall)
- Specificity

## Key Results

- High overall classification accuracy
- Strong separation between **dynamic and static activities**
- Minor confusion between **standing and still**

## Visualizations Generated

The project includes several visualizations:

- Confusion matrix
- Feature distribution plots
- Transition probability matrices
- Emission probability distributions
- Confidence score distributions

These visualizations help interpret how the HMM models activity patterns.



# Project Structure

```
project/
│
├── data/                  # Training data
├── data_unseen/           # Unseen test data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_hmm_implementation.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_visualization.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── hmm_model.py
│   ├── evaluation.py
│   └── visualization.py
│
└── results/
    ├── figures/
    ├── metrics/
    └── predictions/
```



# Collaboration

| Team Member | Contribution |
|-------------|-------------|
| Ange | Data collection, feature extraction implementation, HMM training |
| David | Data preprocessing scripts, evaluation pipeline, visualizations, report writing |

GitHub commit history shows **balanced contributions from both members**.



# Discussion and Conclusion

This project demonstrates that **Hidden Markov Models are highly effective for human activity recognition**, especially when training data is limited.

Key strengths:

- Captures temporal patterns in sensor data
- Interpretable transition probabilities
- Low computational requirements

Limitations:

- Dataset contains only two participants
- Controlled data collection environment

Future improvements:

- Larger dataset with more participants
- Additional activities (running, stairs)
- Real-time smartphone deployment

Overall, the project shows that **smartphone-based activity recognition can support practical applications such as elderly care monitoring and health tracking**.


