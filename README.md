# README: ECG Signal Classification with MLP and CNN Models

## Overview

This project implements two models for ECG signal classification: a **Multilayer Perceptron (MLP)** and a **Convolutional Neural Network (CNN) with Residual Connections**. Both models are designed to classify ECG signals into five categories based on the given input data. I used **PyTorch** for the implementation of both models, ensuring that the structure and hyperparameters followed the specified diagrams and guidelines.

### Key Features

- **ECG_MLP_Akbar_Shah (2).ipynb**: MLP model with fully connected layers, ReLU activations, and optional softmax.
- **ECG_CNN_Akbar_Shah (2).ipynb**: CNN model with residual connections, multiple convolutional layers, average pooling, and normalization layers.
- The models are designed to be robust and achieve a test accuracy of at least **80%** for MLP and **85%** for CNN.

## Structure

1. **ECG_MLP_Akbar_Shah (2).ipynb (MLP Model)**:
   - Input: A 1D ECG signal of shape `(N, 187)`.
   - Layers: Linear layers, ReLU activations, and normalization (BatchNorm, GroupNorm, etc.).
   - Output: A softmax layer that outputs a prediction over 5 categories.

2. **ECG_CNN_Akbar_Shah (2).ipynb  (CNN Model)**:
   - Input: A 1D ECG signal of shape `(N, 1, 187)`.
   - Layers: Convolutional layers with residual connections, ReLU activations, average pooling, normalization, and dense fully connected layers.
   - Output: A softmax layer that outputs a prediction over 5 categories.

## Requirements

- **Python 3.x**
- **PyTorch**: `pip install torch`
- **NumPy**: `pip install numpy`
- **Matplotlib** (for visualization): `pip install matplotlib`
- **scikit-learn** (for evaluation): `pip install scikit-learn`

## Model Details

### MLP Model

- **Input**: The input signal has the shape `(N, 187)`.
- **Layers**:
  - Linear layer followed by ReLU activation.
  - 5 fully connected (Linear/Dense) layers.
  - BatchNorm or GroupNorm is used for normalization after each hidden layer.
  - The final output layer predicts 5 categories using softmax (not required in PyTorch when using cross-entropy loss).

- **Training**:
  - Cross-entropy loss is used for multi-class classification.
  - Optimizer: Adam or SGD.
  - Accuracy must be above 80% on the test set.

### CNN with Residual Connections

- **Input**: The input signal has the shape `(N, 1, 187)` (1D signal).
- **Layers**:
  - Several convolutional layers with ReLU activations and residual connections.
  - Average pooling layers (kernel size 2, stride 2).
  - BatchNorm, GroupNorm, or InstanceNorm is used for normalization after each convolutional block.
  - The network ends with a fully connected layer and softmax output for classification.

- **Training**:
  - Cross-entropy loss is used for multi-class classification.
  - Optimizer: Adam or SGD.
  - Accuracy must be above 85% on the test set.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ECG-Signal-Classification.git
cd ECG-Signal-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the MLP Model

```bash
python train_mlp.py
```

### 4. Run the CNN Model

```bash
python train_cnn.py
```

### 5. Model Evaluation

After training, both models will save the trained weights. You can evaluate the models on the test set to check the accuracy.

```bash
python evaluate_model.py --model mlp    # For MLP model
python evaluate_model.py --model cnn    # For CNN model
```

## Results

- **MLP Model **: Achieved an accuracy of **X%** on the test set.
- **CNN Model **: Achieved an accuracy of **Y%** on the test set (must be at least 85%).



