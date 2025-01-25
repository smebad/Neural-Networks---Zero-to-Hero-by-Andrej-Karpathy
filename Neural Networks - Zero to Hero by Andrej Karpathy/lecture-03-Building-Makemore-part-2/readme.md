# Lecture 3: Building Makemore - Part 2

## Overview

In this lecture, we continued building on the foundational concepts introduced in the Makemore series, focusing on implementing a **Multi-Layer Perceptron (MLP)** for character-level language modeling. We used the Bengio et al. (2003) paper as inspiration, diving deeper into neural network fundamentals and practical implementation details.

## What This Code Does

### 1. Data Preparation
- Processed a dataset of names from the `cleaned_names.txt` file to build a vocabulary of characters.
- Mapped characters to integers (`stoi`) and back to characters (`itos`).
- Created context-based input-output pairs using a sliding window of size 3 (context length).

### 2. Dataset Splitting
- Split the dataset into training (80%), validation (10%), and test (10%) sets to evaluate the model's generalization.
- Implemented a reusable function `build_dataset()` to automate dataset creation.

### 3. Embeddings
- Represented each character as a learned embedding vector.
- Verified embeddings by one-hot encoding characters and projecting them into embedding space.

### 4. Model Architecture
- Built an **MLP** with the following components:
  - **Embedding Layer:** Converts character indices to dense vectors.
  - **Hidden Layer:** A linear layer with weights `W1` and biases `b1`, followed by the `tanh` activation function.
  - **Output Layer:** A linear layer with weights `W2` and biases `b2`, producing logits (unnormalized scores) for all possible next characters.

### 5. Loss Function
- Used **Cross-Entropy Loss** for training, ensuring numerical stability and efficiency.

### 6. Training
- Implemented a training loop with mini-batch gradient descent:
  - Forward pass: Compute the embeddings, hidden layer activations, and logits.
  - Loss computation: Calculate the loss using the cross-entropy function.
  - Backward pass: Compute gradients for all parameters using backpropagation.
  - Parameter updates: Update weights and biases using gradient descent with a learning rate of `0.01`.

### 7. Evaluation
- Computed the loss on the validation and test sets to assess model performance and avoid overfitting.

---

## Fundamentals Learned

### Neural Network Basics
- **Embeddings:** How to represent discrete tokens (characters) as continuous vectors for downstream learning.
- **Activation Functions:** Learned the role of non-linear activations (`tanh`) in introducing non-linearity to the model.
- **Forward and Backward Passes:** Built a deeper understanding of how data flows through a network and how gradients are propagated.

### Model Training
- **Mini-Batch Gradient Descent:** Efficiently updated weights using smaller chunks of data.
- **Learning Rate:** Experimented with learning rates to control the step size of updates.
- **Loss Function:** Understood how cross-entropy loss works for classification tasks.

### PyTorch Concepts
- Efficient tensor operations (e.g., `torch.tanh`, `torch.matmul`).
- Automatic differentiation using `requires_grad` and `loss.backward()`.

---

## Key Concepts Introduced
- **Embedding Layers:** Learned embeddings directly from data to improve model performance.
- **Parameter Initialization:** Initialized weights and biases randomly for effective learning.
- **Numerical Stability:** Used PyTorch's built-in cross-entropy function for stable and efficient computations.
- **Training-Validation-Test Split:** Ensured proper evaluation of the model by splitting the dataset into three subsets.

---

## What Helped Me Understand More
1. **Visualizing Embeddings:** Visualizing the learned embeddings using scatter plots clarified how characters are represented in a continuous space.
2. **Hands-On Gradient Descent:** Manually implementing the gradient computation and parameter updates deepened my understanding of how optimization works.
3. **Iterative Debugging:** Breaking down the forward and backward passes step-by-step helped identify and fix issues quickly.

---

## Challenges and Future Directions
- Understanding the mathematical underpinnings of `tanh` activation and its gradient computation.
- Extending the model to include more complex architectures, such as deeper networks or recurrent models.
- Experimenting with different context sizes and embedding dimensions to optimize performance.

---

## References
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model.
- PyTorch Documentation: [torch.nn.functional](https://pytorch.org/docs/stable/nn.functional.html)

--- 

This lecture was a significant step in solidifying my understanding of neural network fundamentals while building a practical language model. Looking forward to enhancing this model further!
