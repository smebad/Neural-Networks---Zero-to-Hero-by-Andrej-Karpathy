# Lecture 7: Building GPT from Scratch

## Introduction
In this project, I implemented a **Transformer-based GPT model** from scratch using **PyTorch**. This is part of **Andrej Karpathy's Neural Networks: Zero to Hero** series. The goal was to understand and build a **character-level language model** that can generate text, using **self-attention** and **transformer blocks**.

## Concepts Learned
By implementing this model, I learned:
- **Tokenization**: Mapping characters to integer values.
- **Embedding Layers**: Converting tokens into dense vector representations.
- **Self-Attention**: Enabling the model to focus on different parts of the input sequence.
- **Multi-Head Attention**: Using multiple attention heads to capture different features.
- **Transformer Blocks**: Stacking multiple self-attention and feedforward layers.
- **Residual Connections**: Improving gradient flow for better training.
- **Layer Normalization**: Stabilizing training and improving convergence.
- **Dropout Regularization**: Preventing overfitting by randomly disabling neurons.
- **Cross-Entropy Loss**: Measuring how well the model predicts the next token.
- **AdamW Optimizer**: Optimizing the model parameters efficiently.

---

## **Understanding the Code**
### 1. **Data Preprocessing**
- Reads the input text file (`input.txt`).
- Creates a **vocabulary of unique characters** and **integer mappings** (`stoi` & `itos`).
- Encodes the text into integer sequences using **PyTorch tensors**.
- Splits data into **90% training** and **10% validation**.

### 2. **Model Architecture**
The model consists of:
1. **Token Embeddings**: Converts input tokens into dense vectors.
2. **Positional Embeddings**: Provides positional information since transformers are order-agnostic.
3. **Transformer Blocks** (stacked `n_layer` times):
   - **Multi-Head Self-Attention**: Captures dependencies across the sequence.
   - **Feedforward Layer**: Applies transformations after attention.
   - **Layer Normalization & Residual Connections**: Stabilizes training.
4. **Final Linear Layer**: Predicts the next character in the sequence.

### 3. **Training Process**
- Uses **cross-entropy loss** to measure model performance.
- Trained using the **AdamW optimizer** with **learning rate scheduling**.
- Evaluates loss every `eval_interval` iterations.
- Uses **gradient clipping** and **dropout** to prevent overfitting.

### 4. **Generating Text**
- **Autoregressive generation**: The model predicts the next token iteratively.
- Uses **softmax** to sample the next token based on probability distribution.

---

## **Hyperparameters for CPU Training**
Since training a Transformer on a **CPU (Intel Core i7)** is challenging, I optimized the **hyperparameters**:

| Hyperparameter | Value | Reason |
|---------------|---------|--------|
| `batch_size`  | **32**  | Reduces memory usage. |
| `block_size`  | **128** | Shorter sequences for faster training. |
| `n_embd`      | **32**  | Fewer embedding dimensions to reduce computation. |
| `n_head`      | **4**   | Fewer attention heads for efficiency. |
| `n_layer`     | **4**   | Fewer transformer blocks to reduce processing. |
| `dropout`     | **0.2** | Prevents overfitting while keeping efficiency. |
| `learning_rate` | **3e-4** | Optimized for training stability. |

---

## **Why GPUs Perform Better - Tested on google colab**
While this model can run on a CPU, **GPUs** are significantly better because:
✅ **Parallel Computation**: GPUs handle matrix multiplications much faster.  
✅ **Faster Training**: More efficient batch processing and backpropagation.  
✅ **Optimized Tensor Operations**: CUDA-based acceleration speeds up operations.  
✅ **Handles Larger Models**: Can train deeper networks with larger embeddings.

On **GPUs**, I could increase:
- `batch_size` (e.g., 128 or 256)
- `block_size` (e.g., 256 or 512)
- `n_layer` (e.g., 8 or more)
- `n_head` (e.g., 8 or more)
- `n_embd` (e.g., 64 or 128)

---

## **Final Thoughts**
This project helped me **understand the inner workings of GPT-style models**. I implemented **self-attention, multi-head attention, residual connections, and transformer blocks from scratch**.  
While running on a **CPU** required reducing parameters, the core **concepts of transformer-based text generation** remain the same.

**Next Steps**:
- Train on a **larger dataset** (e.g., Wikipedia, Books dataset).
- Implement **byte-pair encoding (BPE)** for better tokenization.
- Train using **GPU acceleration** for faster results.

---

## **References**
- **Andrej Karpathy's Zero to Hero**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLpDlXwq2U3XlQzyy78LFCeRGXWdpsZt-D)
- **Original Transformer Paper**: ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- **PyTorch Documentation**: [pytorch.org/docs](https://pytorch.org/docs)
