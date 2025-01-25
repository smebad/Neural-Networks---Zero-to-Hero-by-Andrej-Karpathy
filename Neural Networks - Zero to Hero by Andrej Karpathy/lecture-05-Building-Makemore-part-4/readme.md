# Lecture 5: Makemore Part 4 - Building a Neural Network with BatchNorm and Manual Backpropagation

This project is part of the "Neural Networks: Zero to Hero" series by Andrej Karpathy. In this lecture, we explored fundamental concepts of neural networks while constructing a more sophisticated model for character-level language modeling. The key highlights include implementing Batch Normalization, a detailed manual backpropagation, and improving our understanding of neural network forward and backward passes.

---

## Key Concepts and Tasks

### 1. **Dataset Preparation**
- **Words Data**: The dataset used consists of character-based names loaded from `cleaned_names.txt`.
- **Vocabulary and Mapping**: Created a vocabulary of unique characters, mapping characters to integers (`stoi`) and integers to characters (`itos`).
- **Dataset Splits**: Divided the dataset into training (80%), validation (10%), and test (10%) splits.
- **Context Windows**: Built input-output pairs using a sliding window approach with a fixed block size (context length of 3).

### 2. **Network Architecture**
- **Embedding Layer**: Embedded the input characters into a dense vector representation.
- **Multilayer Perceptron (MLP)**:
  - Two linear layers with a hidden layer containing 64 neurons.
  - Batch Normalization to stabilize training and improve gradient flow.
  - `tanh` activation function applied to hidden layer outputs.
- **Output Layer**: Produced logits and applied cross-entropy loss for classification.

### 3. **Batch Normalization**
- Implemented Batch Normalization manually by normalizing the hidden layer outputs during the forward pass. This involved:
  - Calculating the batch mean and variance.
  - Normalizing the pre-activation values.
  - Scaling and shifting the normalized values using trainable parameters (`bngain` and `bnbias`).

### 4. **Manual Backpropagation**
- **Backward Pass**: Implemented manual gradient computation for every layer, including Batch Normalization.
- **Key Steps**:
  - Derived gradients for outputs, intermediate variables, and all network parameters.
  - Compared manually computed gradients with PyTorch's autograd for verification.
- **Utility Function**: Used a custom comparison function (`cmp`) to ensure exactness and approximation accuracy of manual gradients.

### 5. **Loss Function**
- **Cross-Entropy Loss**: Calculated manually by:
  - Stabilizing logits with numerical adjustments (subtracting the maximum logit value).
  - Computing softmax probabilities and their logarithms.
  - Calculating the mean negative log-probability of the correct classes.

---

## Fundamentals Learned

### **1. Neural Network Layers**
- Developed a clear understanding of embedding layers, linear transformations, activation functions, and Batch Normalization.
- Learned how layer-specific parameters (weights, biases) are initialized and updated.

### **2. Batch Normalization**
- Understood how Batch Normalization normalizes inputs during training to reduce internal covariate shift.
- Learned the mathematical details of mean, variance, and their impact on gradient flow.

### **3. Manual Backpropagation**
- Gained insights into the chain rule and its application for gradient computation.
- Realized the importance of maintaining intermediate variables for efficient backpropagation.

### **4. Debugging and Verification**
- Compared manual gradients with PyTorch's autograd to ensure correctness, reinforcing trust in the implemented computations.

---

## Challenges and Insights
- **Challenges**:
  - Implementing Batch Normalization from scratch required precise mathematical understanding.
  - Debugging manual backpropagation to ensure accurate gradient flow was time-intensive.
- **Insights**:
  - Observed the importance of numerical stability in softmax and loss computation.
  - Manual implementation deepened the understanding of how PyTorch abstracts complexities behind the scenes.

---

## New Concepts Introduced
- **Batch Normalization**: How it normalizes activations and its impact on training dynamics.
- **Manual Backpropagation**: Deriving gradients step-by-step for each layer and operation.
- **Numerical Stability**: Techniques to avoid overflow/underflow issues, especially in softmax and logarithmic computations.

---

## Acknowledgments
This lecture significantly enhanced my understanding of:
- The mathematical foundations of neural networks.
- Practical challenges and nuances in implementing advanced neural network features.
- The power of PyTorch's autograd system as a tool for rapid prototyping and experimentation.

---

Feel free to explore the code for a hands-on experience with Batch Normalization and manual backpropagation. These concepts form the building blocks for designing and understanding more advanced neural network architectures!
