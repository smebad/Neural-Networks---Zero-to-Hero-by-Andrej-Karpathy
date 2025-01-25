# Lecture 4: MakeMore Part 3 - Neural Networks: Zero to Hero

## Overview
In this lecture, we extended the **MakeMore** project to enhance our understanding of key concepts in machine learning and deep learning. This project revolves around building and training a neural network to generate names character by character. Key improvements and innovations in this lecture include implementing **Kaiming Initialization** and **Batch Normalization**, techniques that are cornerstones of modern deep learning systems.

## What Was Done
### 1. **Data Preparation**
- Loaded and cleaned the dataset of names from `cleaned_names.txt`.
- Built a dataset to model the task of predicting the next character based on a context of previous characters using a sliding window approach.
- Split the dataset into training, validation, and test sets (80%/10%/10%).

### 2. **Model Architecture**
- Implemented a simple Multilayer Perceptron (MLP) with the following layers:
  - **Embedding Layer**: Transformed input characters into dense vector representations.
  - **Linear Layer**: Learned transformations between feature spaces.
  - **Batch Normalization**: Normalized hidden layer outputs to stabilize training.
  - **Non-Linearity**: Applied `tanh` activation to hidden layer outputs.
  - **Output Layer**: Produced logits for the next character prediction.

### 3. **Optimization and Training**
- Used cross-entropy loss as the objective function.
- Employed mini-batch stochastic gradient descent with step learning rate decay.
- Implemented forward and backward passes manually for better conceptual understanding.
- Calibrated batch normalization statistics after training using the entire training set.

## Fundamentals Learned
### 1. **Kaiming Initialization**
- Introduced by Kaiming He in the research paper [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852).
- Helps maintain signal variance across layers during training by scaling weights based on the input size.
- This concept ensures gradients neither vanish nor explode, enabling stable and faster training.

### 2. **Batch Normalization**
- Based on the research paper [Batch Normalization by Sergey Ioffe and Christian Szegedy](https://arxiv.org/abs/1502.03167).
- Addresses internal covariate shift by normalizing layer activations during training.
- Helps reduce training time and improves generalization by making the network more robust to changes in intermediate representations.

### 3. **Softmax and Cross-Entropy**
- Softmax was used to convert logits into probabilities for character prediction.
- Cross-entropy was utilized to compute the loss between predicted probabilities and true labels.

### 4. **Practical Debugging**
- Visualized data distributions and activations using histograms to diagnose issues with initialization and normalization.
- Observed the impact of different initialization strategies on layer outputs and gradients.

## Key Insights
- **Kaiming Initialization** greatly improved the stability of gradient flow in deep networks.
- **Batch Normalization** allowed the model to train more effectively by stabilizing hidden layer activations and reducing dependency on precise initialization.
- These techniques combined enabled deeper networks with faster convergence and better generalization.

## Research Paper Highlights
### [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- Advocates for using Rectified Linear Units (ReLU) and their variants.
- Introduces a method for weight initialization that ensures stable forward and backward signal propagation.

### [Batch Normalization](https://arxiv.org/abs/1502.03167)
- Demonstrates how normalization over mini-batches improves optimization.
- Reduces sensitivity to hyperparameter choices like learning rates and initialization.

## New Concepts Introduced
- **Initialization Techniques**: Explained the need for proper initialization strategies in deep learning.
- **Batch Normalization Mechanics**: Detailed how batch mean and variance are computed and used for scaling and shifting activations.
- **Training Calibration**: Showed the importance of recalibrating batch normalization statistics using the entire training set for better evaluation on unseen data.

## What Helped Solidify Understanding
- Implementing every component manually clarified how modern deep learning techniques work under the hood.
- Debugging training dynamics using histograms and visualizations provided valuable intuition about data flow through the network.

---

This lecture marked a significant step in understanding and implementing practical deep learning concepts. It provided hands-on experience with essential techniques that power state-of-the-art models today.
