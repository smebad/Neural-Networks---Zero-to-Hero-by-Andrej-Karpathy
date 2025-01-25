# Lecture 6: Building Makemore Part 5 - Wavenet

This lecture builds upon the foundational concepts from the "Zero to Hero" series and introduces the architecture of Wavenet. The focus is on implementing a hierarchical model that uses principles of autoregression for generating sequences. Here's an overview of what I accomplished, the fundamentals I reinforced, and the new concepts I learned during this session.

---

## Key Accomplishments

1. **Dataset Preparation**:
   - Processed a dataset of names from the `cleaned_names.txt` file.
   - Constructed training, validation, and test datasets using an autoregressive context of 8 characters to predict the next character.

2. **Vocabulary Building**:
   - Created mappings (`stoi` and `itos`) to translate between characters and their corresponding integer indices, enabling seamless embedding operations.

3. **Hierarchical Neural Network Design**:
   - Designed a hierarchical Wavenet-inspired model with three layers of convolutions and linear transformations to capture long-range dependencies efficiently.
   - Initialized parameters with specific constraints (e.g., scaling the last layer weights) to stabilize training.

4. **Model Training**:
   - Implemented a training loop using:
     - Stochastic Gradient Descent (SGD) with step learning rate decay.
     - Cross-entropy loss for optimizing character predictions.
   - Tracked loss progression over 200,000 steps to ensure convergence.

5. **Sequence Sampling**:
   - Deployed the trained model to generate new sequences (names) by autoregressively predicting characters.

6. **Evaluation**:
   - Measured the performance of the model on train, validation, and test splits using cross-entropy loss.

---

## Fundamentals Reinforced

### 1. Autoregressive Models:
   - Explored how autoregression leverages past context (previous characters) to predict the next character.
   - Solidified understanding of how sliding contexts can be used to model sequences.

### 2. Embedding Layers:
   - Gained deeper insights into how embedding layers map discrete tokens into continuous vector spaces.

### 3. Training Loops:
   - Revisited concepts of backpropagation, gradient computation, and SGD optimization.
   - Implemented learning rate scheduling to balance convergence speed and stability.

---

## New Concepts Introduced

### 1. **Wavenet Architecture**:
   - Introduced hierarchical convolutions to capture dependencies across multiple resolutions of input.
   - Highlighted the importance of dilated convolutions (though not directly implemented here) for efficiently handling large contexts.

### 2. **Batch Normalization**:
   - Improved model training by normalizing intermediate layer outputs to stabilize gradients.

### 3. **Sampling Strategies**:
   - Implemented the `torch.multinomial` function to sample from probability distributions output by the model.

### 4. **Research Insights from the Wavenet Paper**:
   - Understood how Wavenet uses dilated convolutions for audio generation.
   - Gained appreciation for how autoregressive models can generalize sequence modeling across domains (text, audio, etc.).

---

## Challenges and Learnings

- Debugging embedding dimensions and ensuring compatibility with linear layers highlighted the importance of careful tensor shape management.
- Experimenting with hierarchical layers emphasized the trade-offs between model depth and computational efficiency.

---

## Future Directions

- Experiment with dilated convolutions to further align the model with the original Wavenet architecture.
- Extend the sampling logic to handle constrained vocabularies or specific generation rules.
- Explore alternative optimization techniques (e.g., Adam or RMSprop) for faster convergence.

---

This lecture deepened my understanding of autoregressive sequence modeling and introduced the foundational concepts behind Wavenet. By combining practical implementation with theoretical insights from the research paper, I was able to bridge the gap between abstract architecture design and real-world application.
