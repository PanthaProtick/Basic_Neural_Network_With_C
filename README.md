# Basic Neural Network With C

🧠 Built a Neural Network from Scratch — in C!

This project is a fully functional neural network written entirely in C, with no Python, no PyTorch, and no machine learning framework hidden underneath.

Since C was my first programming language, I wanted to challenge myself by building something computationally heavy in the language I know best. The result is a hands-on neural network pipeline that loads data, trains dynamically, evaluates performance, and supports early stopping.

## Project Highlights

- Forward pass and backpropagation implemented from scratch
- ReLU activation and Mean Squared Error loss
- He initialization using Box-Muller Gaussian sampling
- Stochastic Gradient Descent with manual weight and bias updates
- Linked-layer architecture built with structs and pointers
- CSV loading, parsing, shuffling, and train/test splitting
- Feature scaling based on training data only
- Early stopping with best-weight snapshotting and restoration
- Model evaluation using MSE and $R^2$

## What I Built Myself

- Network structure, layers, nodes, and connections
- Forward propagation logic
- Backpropagation and gradient calculation
- ReLU activation and derivative
- He-style weight initialization
- Manual weight and bias updates during training

## What Was Built With AI Assistance

- CSV data pipeline
- Dataset shuffling and splitting
- Train/test split logic with no data leakage
- Feature scaling and label scaling
- Early stopping and best-model restoration
- Evaluation metrics for train and test sets

## What I Learned

- The real math behind forward propagation and backpropagation
- Why initialization matters for deep learning stability
- Why GPUs and parallelism exist for this kind of work
- How to structure a full ML workflow without a library
- How AI tools can speed up development while still leaving the core learning experience intact

## Why This Project Matters

I started with hardcoded data and ended with a model that can train dynamically on any compatible CSV file.

It is not perfect, but it is largely mine — and that makes it one of the projects I am most proud of.

## How It Works

1. Load a CSV dataset
2. Shuffle the rows
3. Split the data into training and test sets
4. Scale features and labels using training statistics only
5. Build the neural network architecture
6. Train with stochastic gradient descent
7. Track validation loss with early stopping
8. Restore the best model weights
9. Evaluate the final model using MSE and $R^2$

## Technologies Used

- C
- Standard C libraries
- Dynamic memory management
- Basic numerical methods

## Note

This project was intentionally built without machine learning libraries to focus on understanding the fundamentals and the math behind the model.

## Closing Thoughts

This was a challenging build, but also a deeply rewarding one.

It taught me a lot about machine learning, low-level programming, and the value of learning by doing.

Built from scratch. Learned step by step. Proudly in C.
