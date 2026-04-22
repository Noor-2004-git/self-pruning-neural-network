# Self-Pruning Neural Network for CIFAR-10

A PyTorch implementation of a dynamically self-pruning neural network built for CIFAR-10 image classification.

This project was developed as part of the Tredence AI Engineering Internship Case Study. The goal was to design a neural network capable of pruning its own connections during training using learnable gate parameters and sparsity regularization.

---

## Project Overview

Traditional neural network pruning is usually performed after training. In this project, pruning is integrated directly into the training process.

Each weight in the network is paired with a learnable gate parameter. During the forward pass, gate values are generated using a sigmoid transformation and multiplied element-wise with the weights.

Connections with gate values close to zero become effectively inactive, allowing the model to automatically learn a sparse architecture during training.

---

## Core Idea

The effective weight used during the forward pass is:

$$
\text{effective weight} = \text{weight} \times \sigma(\text{gate score})
$$

The model is trained using the following objective:

$$\mathrm{Total\ Loss}=\mathrm{Classification\ Loss}+\lambda\cdot\mathrm{Sparsity\ Loss}$$

where:

* Classification Loss improves prediction accuracy
* Sparsity Loss encourages unnecessary connections to move toward zero
* λ controls the sparsity vs accuracy tradeoff

---

## Features

* Custom `PrunableLinear` layer implemented from scratch
* Learnable gate mechanism for dynamic pruning
* L1 sparsity regularization on gate activations
* Custom training loop with combined loss function
* Sparsity measurement and analysis
* Comparison across multiple lambda values
* Gate distribution visualization using matplotlib

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## Model Architecture

The network uses fully connected prunable layers:

```text id="v4u4ud"
Input (3072)
   ↓
PrunableLinear(3072 → 1024)
   ↓
ReLU + Dropout
   ↓
PrunableLinear(1024 → 512)
   ↓
ReLU + Dropout
   ↓
PrunableLinear(512 → 256)
   ↓
ReLU + Dropout
   ↓
PrunableLinear(256 → 10)
```

---

## Dataset

The model is trained on the CIFAR-10 dataset provided by `torchvision.datasets`.

CIFAR-10 contains 60,000 color images across 10 classes.

---

## Installation

Clone the repository:

```bash id="ev57hf"
git clone https://github.com/your-username/self-pruning-neural-network.git
cd self-pruning-neural-network
```

Install dependencies:

```bash id="1a8pk3"
pip install torch torchvision matplotlib numpy
```

---

## Running the Project

Run the training script:

```bash id="s8vqyv"
python self_pruning_network.py
```

The script will:

* train the model for different lambda values
* compute sparsity levels
* evaluate test accuracy
* generate gate distribution plots

---

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 1e-5   | 58.95             | 8.66              |
| 1e-4   | 58.43             | 37.91              |
| 1e-3   | 53.50             | 58.57              |

---

## Gate Distribution Visualization

The final gate distribution plot for the best-performing model:

![Gate Distribution]
<img width="871" height="541" alt="image" src="https://github.com/user-attachments/assets/f81f28d9-6ed0-4ab9-983a-b2dc6b929bfb" />


A successful pruning strategy should produce:

* a large concentration of gate values near zero
* a smaller cluster of important active connections away from zero

---

## Key Insights

* Lower lambda values preserve more connections and generally improve accuracy
* Higher lambda values increase sparsity by aggressively pruning weak connections
* The model learns which connections are important during training instead of relying on post-training pruning

---

## Future Improvements

* Structured neuron pruning
* Convolutional self-pruning layers
* Dynamic thresholding strategies
* Sparse inference optimization
* Integration with vectorized sparse operations

---

## Author

Developed for the Tredence AI Engineering Internship Case Study.
# self-pruning-neural-network
