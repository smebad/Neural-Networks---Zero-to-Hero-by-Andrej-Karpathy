# Micrograd from Scratch - Lecture 1

This repository contains the implementation and code from Lecture 1 of Andrej Karpathy's "Neural Networks: Zero to Hero" series. In this lecture, we laid the groundwork for understanding automatic differentiation by building a basic computational graph and implementing the core concepts of derivatives and backpropagation.

---

## What We Did

1. **Explored the Fundamentals of Derivatives**  
   - We computed numerical derivatives using the definition of derivatives:
     \[
     \text{slope} = \frac{f(x + h) - f(x)}{h}
     \]
   - Implemented derivative calculations for both simple and composite functions.

2. **Built a Custom `Value` Class**  
   - Designed a Python class to represent values in a computational graph. Each value could:
     - Store its data.
     - Track gradients for backpropagation.
     - Record its parent nodes and operation (`+`, `*`, etc.).

3. **Implemented Core Operations**  
   - Implemented support for operations like addition (`+`), multiplication (`*`), subtraction (`-`), division (`/`), and exponentiation (`**`).
   - Enabled each operation to contribute to gradient computation through backward functions.

4. **Visualized the Computational Graph**  
   - Used `graphviz` to create a visualization of the computational graph, showing the relationships between operations and gradients.

5. **Introduced Backpropagation**  
   - Created a method to compute gradients (partial derivatives) for all nodes in the graph using reverse-mode automatic differentiation.

---

## Fundamentals Learned

1. **Numerical Derivatives**  
   - Learned how to approximate the derivative of a function at a point using small perturbations (`h`).

2. **Computational Graphs**  
   - Understood how to represent complex functions as graphs where nodes are operations or inputs, and edges represent dependencies.

3. **Gradients and Backpropagation**  
   - Grasped how to compute the derivative of the output with respect to any input efficiently using the chain rule.

4. **Automatic Differentiation**  
   - Explored how gradients can be computed programmatically for any differentiable function.

---

## What Helped Me Understand the Concepts

1. **Breaking Down the Math**  
   - Step-by-step implementation of simple operations like addition and multiplication helped solidify the understanding of derivatives.

2. **Visualizing the Computational Graph**  
   - Using `graphviz` to see how values flow through the graph and how gradients are propagated made abstract concepts tangible.

3. **Hands-on Coding**  
   - Writing the `Value` class and implementing operations from scratch provided a deeper understanding of how frameworks like PyTorch and TensorFlow handle gradients internally.

---

## Key Takeaways

- Numerical derivatives are an excellent starting point but are inefficient and prone to errors due to the choice of `h`.
- Reverse-mode automatic differentiation, as implemented in this project, is the backbone of modern deep learning frameworks.
- Visualization of computation graphs is a powerful tool for understanding how backpropagation works.

---

