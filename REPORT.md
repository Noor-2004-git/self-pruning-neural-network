# Self-Pruning Neural Network Report

## Overview

This project implements a self-pruning neural network for CIFAR-10 image classification.

Instead of pruning weights after training, the network learns during training which connections are important and which can be removed.

Each weight is paired with a learnable gate parameter. During the forward pass, the effective weight is computed as:

$$
\text{effective weight} = \text{weight} \times \sigma(\text{gate score})
$$

where the sigmoid function converts gate scores into values between 0 and 1.

If a gate value becomes very close to zero, the corresponding connection contributes very little to the output and is effectively pruned.

---

## Why L1 Regularization Encourages Sparsity

The sparsity loss is computed using the L1 norm of the gate values.

L1 regularization encourages sparsity because it penalizes every active gate linearly. As a result, unnecessary connections are pushed toward zero during training.

Important connections remain active because removing them would increase the classification loss, while less useful connections are suppressed to reduce the sparsity penalty.

This creates a balance between maintaining good classification performance and reducing the number of active connections in the network.

---

## Total Loss Function

The model is trained using the following objective:

$$\mathrm{Total\ Loss}=\mathrm{Classification\ Loss}+\lambda\cdot\mathrm{Sparsity\ Loss}$$

where:

* Classification Loss helps the model improve prediction accuracy
* Sparsity Loss encourages the network to remove unnecessary connections
* λ controls the tradeoff between accuracy and sparsity

---

## Experimental Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 1e-5   | 58.95             | 8.66               |
| 1e-4   | 58.43             | 37.91              |
| 1e-3   | 53.50             | 58.57              |

---

## Analysis

Smaller lambda values prioritize classification accuracy and produce lower sparsity levels.

Larger lambda values apply stronger pressure on gate values, which increases sparsity by pruning more connections. However, aggressive pruning can reduce classification accuracy.

The results demonstrate the expected tradeoff between model compactness and predictive performance.

---

## Gate Distribution Plot

The histogram below shows the distribution of gate values for the best-performing model.

A successful self-pruning model should show:

* a large concentration of gate values near zero
* a smaller cluster of important active connections away from zero
 [image in another file named gate_distribution_lambda_0.0001.png]


## Conclusion

This project demonstrates how a neural network can dynamically prune its own connections during training using learnable gates and sparsity regularization.

The approach successfully balances classification performance with model sparsity, resulting in a more compact and efficient neural network.

