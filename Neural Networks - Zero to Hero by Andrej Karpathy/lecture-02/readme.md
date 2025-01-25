# Lecture 2: Building Makemore - Fundamentals of Language Modeling with PyTorch

## Overview
In this lecture, we took a significant step toward building a bigram-based character-level language model for generating names. Using a dataset of names, we explored fundamental concepts in statistical modeling and implemented these concepts with PyTorch. The goal was to develop a deeper understanding of data representation, probability, and neural network basics.

## Key Concepts and What We Learned
### 1. **Data Cleaning and Preparation**
   - Processed a raw dataset of names (`names.txt`) to remove duplicates, standardize formatting (lowercase and stripped spaces), and sort names alphabetically.
   - Saved the cleaned dataset as `cleaned_names.txt`.

### 2. **Bigram Character Statistics**
   - Calculated bigram (character-pair) frequencies to understand character relationships in names.
   - Stored counts in a 2D matrix `N` representing transitions between characters.
   - Visualized the bigram matrix using `matplotlib` for better understanding.

### 3. **Probability Distribution**
   - Converted the bigram frequency matrix into probabilities using **maximum likelihood estimation**.
   - Explored probability normalization and the use of PyTorch for efficient calculations.
   - Learned how to sample from a probability distribution using `torch.multinomial`.

### 4. **One-Hot Encoding**
   - Represented characters as one-hot encoded vectors to use them as inputs to a neural network.
   - Practiced matrix multiplication and its application in neural networks.

### 5. **Softmax and Log-Likelihood**
   - Computed logits (log-counts) and applied the **softmax function** to convert them into probabilities.
   - Calculated negative log-likelihoods to measure the model's performance.
   - Understood the importance of minimizing average negative log-likelihood to train the model effectively.

### 6. **Neural Network Basics**
   - Initialized random weights for a neural network with PyTorch.
   - Performed a forward pass to calculate the output probabilities of the network.
   - Observed how the network assigns probabilities to the next character based on input characters.

## Hands-On Insights
This lecture reinforced critical programming and mathematical concepts:
- **Matrix Manipulation:** The use of matrices to represent bigram frequencies and probabilities.
- **Broadcasting Semantics in PyTorch:** Efficiently performing operations across tensors.
- **Statistical Modeling:** Building a foundation in likelihood estimation and probability.
- **Sampling and Randomness:** Using PyTorch's `multinomial` function to sample from learned probabilities.

## Key Takeaways
- Language modeling involves understanding the relationships between characters and their probabilities.
- Negative log-likelihood is a crucial metric for evaluating model performance.
- A softmax function is essential for converting logits into interpretable probabilities.

## Additional Resources
To dive deeper into the concepts, the following resources were helpful:
- [WolframAlpha](https://www.wolframalpha.com) for exploring logarithmic properties.
- PyTorch documentation for tensor operations and softmax functionality.
- Tutorials on **Maximum Likelihood Estimation (MLE)** for statistical modeling fundamentals.

## What's Next?
In the upcoming lectures, we will:
- Explore how to train the neural network to optimize weights for better performance.
- Learn more about gradient-based optimization techniques like backpropagation.
- Gradually build a more complex model for name generation.

---

This project is part of the [Zero to Hero: Neural Networks](https://karpathy.ai/zero-to-hero.html) series by Andrej Karpathy.
