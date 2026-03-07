# Human Activity Recognition using Hidden Markov Models

A machine learning project that uses Hidden Markov Models (HMMs) to recognize human activities from smartphone sensor data (accelerometer and gyroscope).

## Project Overview

This project implements a complete pipeline for activity recognition including data collection, preprocessing, feature extraction, HMM model training, and evaluation. The system classifies four activities:

- **Standing** - Person standing still
- **Walking** - Person walking at normal pace
- **Jumping** - Person jumping in place
- **Still** - Person sitting or lying down

## Dataset Information

### Data Collection

- **Devices**: iPhone 11 (iOS 17.4.1) and iPhone X (iOS 16.7.14)
- **App**: Sensor Logger v1.54
- **Sensors**: Accelerometer and Gyroscope
- **Sampling Rate**: 100 Hz (10ms intervals)
- **Participants**: 2 participants (Ange and David)
- **Recordings**: ~10-12 recordings per activity per participant

### Data Structure

```
data/
├── ange/
│   ├── jumping/
│   ├── standing/
│   ├── still/
│   └── walking/
└── david/
    ├── jumping/
    ├── standing/
    ├── still/
    └── walking/

data_unseen/  # Test data with same structure
```

Each recording folder contains:

- `Accelerometer.csv` - 3-axis acceleration (x, y, z in m/s²)
- `Gyroscope.csv` - 3-axis angular velocity (x, y, z in rad/s)
- `Metadata.csv` - Device and recording information

## Model Architecture

### Hidden Markov Models

- **Model Type**: Gaussian HMM with diagonal covariance
- **Hidden States**: 4 states per activity
- **Training**: One HMM trained per activity class
- **Classification**: Maximum likelihood prediction

### Feature Engineering (27 features)

**Time-domain features (18):**

- Mean, Standard Deviation, Variance for each axis (6 axes × 3 = 18)

**Frequency-domain features (3):**

- Dominant frequency (FFT) for accelerometer axes (x, y, z)

**Signal features (6):**

- Signal Magnitude Area (SMA) for accelerometer and gyroscope
- Cross-axis correlations (accel_xy, accel_xz, gyro_xy, gyro_xz)

### Window Processing

- **Window Size**: 50 samples (~0.5 seconds at 100Hz)
- **Overlap**: 50% (step size = 25 samples)

## Project Structure

```
Formative-2-Hidden-Markov-Models/
├── data/                          # Training data
├── data_unseen/                   # Test data
├── models/                        # Saved models
│   ├── hmm_models.pkl            # Trained HMM models
│   └── scaler.pkl                # Feature scaler
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_hmm_implementation.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_visualization.ipynb
├── processed_data/               # Processed features
│   ├── training_features.csv
│   └── merged_sensor_data.csv
├── results/                      # Evaluation results
│   ├── figures/                  # Visualizations
│   ├── metrics/                  # Performance metrics
│   └── predictions/              # Model predictions
├── scripts/                      # Utility scripts
│   └── restructure_data.py      # Data reorganization
├── src/                          # Source code
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── hmm_model.py
│   ├── evaluation.py
│   └── visualization.py
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Results

### Model Performance (Unseen Test Data)

- **Overall Accuracy**: See `results/metrics/evaluation_summary.txt`
- **Confusion Matrix**: `results/metrics/confusion_matrix_unseen.csv`
- **Predictions**: `results/predictions/unseen_test_predictions.csv`

### Key Findings

- HMM models successfully distinguish between dynamic (jumping, walking) and static (standing, still) activities
- Transition matrices reveal distinct patterns for each activity:
  - High self-transition probabilities for static activities
  - More state transitions for dynamic activities
- Feature analysis shows clear separation between activity classes

### Visualizations

All visualizations available in `results/figures/`:

1. `hmm_transition_matrices.png` - State transition probabilities
2. `initial_state_probabilities.png` - Starting state distributions
3. `feature_distributions.png` - Feature values across activities
4. `feature_correlation.png` - Feature correlation heatmap
5. `accuracy_by_activity.png` - Per-activity performance
6. `confidence_distribution.png` - Prediction confidence analysis
7. `sample_sensor_data.png` - Raw sensor readings
8. `confusion_matrix_detailed.png` - Detailed confusion analysis

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- pandas
- numpy
- hmmlearn
- scikit-learn
- scipy
- matplotlib
- seaborn
- joblib

## Usage

### 1. Data Organization

If you have new sensor data, use the restructuring script:

```bash
python scripts/restructure_data.py
```

### 2. Run the Analysis Pipeline

Execute notebooks in order:

**a. Data Exploration**

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**b. Feature Extraction**

```bash
jupyter notebook notebooks/02_feature_extraction.ipynb
```

- Extracts features from raw sensor data
- Creates sliding windows
- Saves to `processed_data/training_features.csv`

**c. HMM Training**

```bash
jupyter notebook notebooks/03_hmm_implementation.ipynb
```

- Trains one HMM per activity
- Saves models to `models/hmm_models.pkl`

**d. Model Evaluation**

```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

- Tests on unseen data
- Generates confusion matrix and metrics

**e. Visualization**

```bash
jupyter notebook notebooks/05_visualization.ipynb
```

- Creates comprehensive visualizations
- Saves all figures to `results/figures/`

### 3. Using Python Modules Directly

```python
from src.data_loader import load_sensor_data
from src.feature_extraction import extract_features_from_recording
from src.hmm_model import train_hmm_models, predict_activity

# Load data
accel, gyro = load_sensor_data('path/to/recording')

# Extract features
features = extract_features_from_recording(accel, gyro)

# Load trained models
import joblib
models = joblib.load('models/hmm_models.pkl')

# Predict activity
prediction = predict_activity(features, models)
```

## Technical Details

### Feature Extraction

The feature extraction module (`src/feature_extraction.py`) processes sensor data:

- Handles both raw (x, y, z) and processed (accel_x, accel_y, accel_z) column formats
- Applies sliding window segmentation
- Computes time-domain, frequency-domain, and correlation features

### HMM Training

Model training (`src/hmm_model.py`):

- StandardScaler normalization
- 80/20 train-test split for validation
- GaussianHMM with 4 hidden states
- Diagonal covariance matrix
- 100 training iterations

### Evaluation Metrics

Comprehensive evaluation (`src/evaluation.py`):

- Per-activity accuracy
- Confusion matrix (counts and percentages)
- Prediction confidence scores
- Misclassification analysis

## Future Improvements

1. **Model Enhancements**
   - Try different HMM configurations (more states, full covariance)
   - Implement ensemble methods
   - Experiment with other temporal models (RNNs, LSTMs)

2. **Feature Engineering**
   - Add more frequency-domain features (spectral entropy, power)
   - Implement wavelet transforms
   - Try automatic feature selection

3. **Data Collection**
   - Collect more data for improved generalization
   - Add more participants for diversity
   - Include transition activities (standing to walking)

4. **Real-time Inference**
   - Implement streaming prediction
   - Optimize for mobile deployment
   - Create mobile app integration

## Contributors

- **David CYUBAHIRO** - Data collection, project setup, evaluation, visualization
- **Ange Constance** - Data collection, feature extraction, HMM implementation

## License

This project is for educational purposes as part of ML Techniques coursework.

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- Shoaib, M., et al. (2014). Fusion of smartphone motion sensors for physical activity recognition.
- hmmlearn documentation: https://hmmlearn.readthedocs.io/

## Contact

For questions or collaboration, please contact the project contributors.

---

**Last Updated**: March 7, 2026
