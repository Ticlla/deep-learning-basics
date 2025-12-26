# Level 2: Building Your First Neural Network

> Deep dive into how neural networks work: architecture, feedforward, backpropagation, and training.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Network Architecture](#2-network-architecture)
3. [Weights and Biases](#3-weights-and-biases)
4. [Sigmoid Activation](#4-sigmoid-activation)
5. [Feedforward Computation](#5-feedforward-computation)
6. [Cost Function](#6-cost-function)
7. [Backpropagation](#7-backpropagation)
8. [Stochastic Gradient Descent](#8-stochastic-gradient-descent)
9. [Training Process](#9-training-process)
10. [Code Reference](#10-code-reference)
11. [Exercises](#11-exercises)

---

## 1. Overview

A neural network is a computational model inspired by biological neurons. It learns to recognize patterns by adjusting internal parameters (weights and biases) based on examples.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     NEURAL NETWORK OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INPUT              HIDDEN              OUTPUT                         │
│   (784 pixels)       (30 features)       (10 probabilities)            │
│                                                                         │
│      ○                   ○                   ○  → P(0) = 0.01          │
│      ○                   ○                   ○  → P(1) = 0.02          │
│      ○       W¹, b¹      ○       W², b²      ○  → P(2) = 0.95  ← MAX   │
│     ...    ───────►     ...    ───────►     ...                        │
│      ○                   ○                   ○  → P(9) = 0.01          │
│                                                                         │
│   "What is           "I see             "It's a 2!"                    │
│    this digit?"       curves..."                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Learning Process

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  1. FEEDFORWARD        2. COMPUTE ERROR       3. BACKPROPAGATE        │
│  ─────────────         ──────────────         ───────────────         │
│  Pass input through    Compare output         Calculate how each      │
│  network to get        to correct answer      weight contributed      │
│  prediction                                   to the error            │
│                                                                        │
│       ○───►○───►○            ○                    ○◄───○◄───○         │
│                          prediction               gradients            │
│                             ↓                                          │
│                         - target                                       │
│                             ↓                                          │
│                           error                                        │
│                                                                        │
│  4. UPDATE WEIGHTS                                                     │
│  ────────────────                                                      │
│  Adjust weights to reduce error:  w_new = w_old - η × gradient        │
│                                                                        │
│  5. REPEAT for thousands of examples until network learns!            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Network Architecture

### Layer Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NETWORK [784, 30, 10]                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 0              LAYER 1              LAYER 2                      │
│  INPUT                HIDDEN               OUTPUT                       │
│  ────────             ──────               ──────                       │
│                                                                         │
│  784 neurons          30 neurons           10 neurons                   │
│  (no weights)         (learns features)    (predictions)               │
│                                                                         │
│     a⁰                    a¹                   a²                       │
│    (784,1)              (30,1)               (10,1)                     │
│                                                                         │
│           ────W¹────►          ────W²────►                             │
│           (30×784)             (10×30)                                  │
│                                                                         │
│           ────b¹────►          ────b²────►                             │
│           (30×1)               (10×1)                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why These Sizes?

| Layer | Neurons | Reason |
|-------|---------|--------|
| Input | 784 | One per pixel (28×28 = 784) |
| Hidden | 30 | Enough to learn features, not too slow |
| Output | 10 | One per digit class (0-9) |

### Neuron Model

```
                    SINGLE NEURON
    ┌─────────────────────────────────────────┐
    │                                         │
    │   inputs (from previous layer)          │
    │      ↓                                  │
    │   x₁ ──w₁──┐                           │
    │   x₂ ──w₂──┼──► Σ ──(+b)──► σ ──► output
    │   x₃ ──w₃──┘    │                │     │
    │   ...           │                │     │
    │                 │                │     │
    │            weighted          sigmoid   │
    │              sum            activation │
    │                                         │
    │   z = w₁x₁ + w₂x₂ + w₃x₃ + ... + b    │
    │   a = σ(z)                              │
    │                                         │
    └─────────────────────────────────────────┘
```

---

## 3. Weights and Biases

### What Are Weights?

Weights determine **how much each input influences the output**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEIGHT INTERPRETATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Large positive weight (+2.5):  Input strongly ACTIVATES the neuron    │
│  Large negative weight (-2.5):  Input strongly INHIBITS the neuron     │
│  Small weight (~0):             Input has little effect                │
│                                                                         │
│  Example: A hidden neuron that detects "horizontal lines"              │
│                                                                         │
│      Input Image        Weights (28×28)        Result                  │
│      ┌─────────┐        ┌─────────┐                                    │
│      │─────────│        │+++++++++│           High                     │
│      │         │   ×    │         │     =     activation               │
│      │         │        │         │           (line detected!)         │
│      └─────────┘        └─────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What Are Biases?

Biases control **how easily a neuron activates**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BIAS INTERPRETATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  z = Σ(wᵢxᵢ) + b                                                       │
│                                                                         │
│  Positive bias (+5):  Neuron activates easily (even with weak input)   │
│  Negative bias (-5):  Neuron is "skeptical" (needs strong input)       │
│  Zero bias (0):       Neutral threshold                                │
│                                                                         │
│              Sigmoid Output                                             │
│         1 ┤         ╭────────                                          │
│           │        ╱                                                    │
│       0.5 ┤───────╳────────   ← b shifts this curve left/right        │
│           │      ╱                                                      │
│         0 ┤─────╯                                                       │
│           └──────────────────                                          │
│              -10   0   +10   z                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Parameter Count

```
Network [784, 30, 10]:

Layer 1 (input → hidden):
  Weights W¹: 30 × 784 = 23,520 parameters
  Biases b¹:  30 × 1   = 30 parameters
  
Layer 2 (hidden → output):
  Weights W²: 10 × 30  = 300 parameters
  Biases b²:  10 × 1   = 10 parameters

TOTAL: 23,520 + 30 + 300 + 10 = 23,860 learnable parameters
```

### Weight Matrix Dimensions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEIGHT MATRIX SHAPES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  W[layer] has shape: (neurons_out, neurons_in)                         │
│                                                                         │
│  W¹ shape: (30, 784)                                                   │
│                                                                         │
│       ┌─ 784 columns (one per input pixel) ─┐                          │
│       ↓                                     ↓                          │
│  30  ╔═══════════════════════════════════════╗                         │
│  rows║  w₀,₀   w₀,₁   w₀,₂  ...  w₀,₇₈₃    ║ ← weights to neuron 0   │
│  (one║  w₁,₀   w₁,₁   w₁,₂  ...  w₁,₇₈₃    ║ ← weights to neuron 1   │
│  per ║  ...    ...    ...   ...  ...        ║                         │
│  out)║  w₂₉,₀  w₂₉,₁  w₂₉,₂ ... w₂₉,₇₈₃   ║ ← weights to neuron 29  │
│      ╚═══════════════════════════════════════╝                         │
│                                                                         │
│  Each ROW contains all weights connecting to ONE hidden neuron         │
│  (This is why we can visualize each row as a 28×28 "filter")          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Sigmoid Activation

### The Sigmoid Function

```
                    1
σ(z) = ─────────────────
         1 + e^(-z)
```

### Properties

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SIGMOID PROPERTIES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Output range: (0, 1) - interpretable as probability                │
│                                                                         │
│  2. Smooth & differentiable - needed for gradient descent              │
│                                                                         │
│  3. Key values:                                                         │
│     σ(-∞) → 0     (very negative input → output near 0)                │
│     σ(0)  = 0.5   (zero input → output exactly 0.5)                    │
│     σ(+∞) → 1     (very positive input → output near 1)                │
│                                                                         │
│  4. Derivative: σ'(z) = σ(z) × (1 - σ(z))                              │
│     Maximum at z=0: σ'(0) = 0.25                                       │
│     Vanishes for large |z| (vanishing gradient problem!)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Visualization

```
    Sigmoid σ(z)                    Derivative σ'(z)
    
  1 ┤         ╭─────────          0.25┤      ╭╮
    │        ╱                        │     ╱  ╲
    │       ╱                         │    ╱    ╲
0.5 ┤──────╳                          │   ╱      ╲
    │     ╱                           │  ╱        ╲
    │    ╱                            │ ╱          ╲
  0 ┤───╯                           0 ┼╱            ╲───
    └────────────────────             └────────────────────
      -5    0    +5   z                 -5    0    +5   z
```

---

## 5. Feedforward Computation

### The Algorithm

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEEDFORWARD STEP BY STEP                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Start with input a⁰ = x (the image, 784×1)                            │
│                                                                         │
│  For each layer l = 1, 2, ..., L:                                      │
│                                                                         │
│      zˡ = Wˡ · aˡ⁻¹ + bˡ     ← weighted sum                           │
│      aˡ = σ(zˡ)               ← apply activation                       │
│                                                                         │
│  Output: aᴸ (the prediction, 10×1)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Concrete Example

```
Network [784, 30, 10] with an image of "5":

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT LAYER                                                            │
│  ───────────                                                            │
│  a⁰ = x = [0.0, 0.0, 0.3, 0.9, ..., 0.0]ᵀ   (784×1 image vector)      │
│                                                                         │
│                     ↓                                                   │
│            z¹ = W¹ · a⁰ + b¹                                           │
│            (30×784)·(784×1) + (30×1) = (30×1)                          │
│                     ↓                                                   │
│            a¹ = σ(z¹)                        (30×1 hidden activations) │
│                                                                         │
│  HIDDEN LAYER                                                           │
│  ────────────                                                           │
│  a¹ = [0.2, 0.8, 0.1, 0.9, ..., 0.3]ᵀ       (30×1)                    │
│                                                                         │
│                     ↓                                                   │
│            z² = W² · a¹ + b²                                           │
│            (10×30)·(30×1) + (10×1) = (10×1)                            │
│                     ↓                                                   │
│            a² = σ(z²)                        (10×1 output activations) │
│                                                                         │
│  OUTPUT LAYER                                                           │
│  ────────────                                                           │
│  a² = [0.01, 0.02, 0.05, 0.08, 0.03, 0.75, 0.02, 0.02, 0.01, 0.01]ᵀ   │
│         ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑       │
│        P(0)  P(1)  P(2)  P(3)  P(4)  P(5)  P(6)  P(7)  P(8)  P(9)     │
│                                       │                                 │
│                              Highest! ← Prediction = 5                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Cost Function

### Quadratic Cost (Mean Squared Error)

```
         1
C = ─────── Σ ‖y - a‖²
      2n    x

Where:
  n = number of training examples
  y = target output (one-hot vector)
  a = actual output (network prediction)
  ‖·‖ = Euclidean norm
```

### Example Calculation

```
Target y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]ᵀ   (digit 5)
Output a = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.05, 0.05, 0.05, 0.05]ᵀ

Error vector (y - a):
  [0-0.1, 0-0.1, 0-0.1, 0-0.1, 0-0.1, 1-0.4, 0-0.05, ...]
= [-0.1, -0.1, -0.1, -0.1, -0.1, 0.6, -0.05, -0.05, -0.05, -0.05]

‖y - a‖² = 0.01 + 0.01 + 0.01 + 0.01 + 0.01 + 0.36 + 0.0025×4
         = 0.05 + 0.36 + 0.01
         = 0.42

Cost for this example: C = 0.42 / 2 = 0.21
```

---

## 7. Backpropagation

### The Four Fundamental Equations

```
╔═════════════════════════════════════════════════════════════════════════╗
║                    THE BACKPROPAGATION EQUATIONS                        ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  (BP1)  δᴸ = ∇ₐC ⊙ σ'(zᴸ)                                              ║
║         │                                                               ║
║         └─► Error at OUTPUT layer                                       ║
║             (how much output affects cost × how much z affects output)  ║
║                                                                         ║
║  (BP2)  δˡ = (Wˡ⁺¹)ᵀ · δˡ⁺¹ ⊙ σ'(zˡ)                                   ║
║         │                                                               ║
║         └─► Error at layer l in terms of layer l+1                      ║
║             (propagate error BACKWARD through weights)                  ║
║                                                                         ║
║  (BP3)  ∂C/∂bˡⱼ = δˡⱼ                                                   ║
║         │                                                               ║
║         └─► Gradient for biases = the error itself!                     ║
║                                                                         ║
║  (BP4)  ∂C/∂wˡⱼₖ = aˡ⁻¹ₖ · δˡⱼ                                          ║
║         │                                                               ║
║         └─► Gradient for weights = input activation × error             ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝

Key: ⊙ = element-wise (Hadamard) product
```

### Visual Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BACKPROPAGATION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│           FORWARD PASS (compute activations)                            │
│           ══════════════════════════════════►                           │
│                                                                         │
│      a⁰ ────W¹────► a¹ ────W²────► a² ────────► Cost C                 │
│     input         hidden         output                                 │
│                                                                         │
│                                                                         │
│           ◄══════════════════════════════════                           │
│           BACKWARD PASS (compute gradients)                             │
│                                                                         │
│                                                                         │
│  Step 1: Compute output error                                           │
│          δ² = (a² - y) ⊙ σ'(z²)                                        │
│                     ↑                                                   │
│                     │                                                   │
│  Step 2: Propagate backward                                             │
│          δ¹ = (W²)ᵀ · δ² ⊙ σ'(z¹)                                      │
│                     ↑                                                   │
│                     │                                                   │
│  Step 3: Compute gradients                                              │
│          ∂C/∂W² = δ² · (a¹)ᵀ                                           │
│          ∂C/∂b² = δ²                                                    │
│          ∂C/∂W¹ = δ¹ · (a⁰)ᵀ                                           │
│          ∂C/∂b¹ = δ¹                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why It Works: Chain Rule

```
Cost depends on output:           C = f(a²)
Output depends on z:              a² = σ(z²)
z depends on weights:             z² = W² · a¹ + b²

By chain rule:
∂C/∂W² = ∂C/∂a² · ∂a²/∂z² · ∂z²/∂W²
         \_____/   \_____/   \_____/
            │         │         │
            │         │         └── = a¹ᵀ (input to this layer)
            │         └──────────── = σ'(z²) (sigmoid derivative)
            └────────────────────── = (a² - y) (cost derivative)

Combined: ∂C/∂W² = (a² - y) ⊙ σ'(z²) · (a¹)ᵀ = δ² · (a¹)ᵀ
```

---

## 8. Stochastic Gradient Descent

### The Update Rule

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRADIENT DESCENT UPDATE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  w_new = w_old - η × ∂C/∂w                                             │
│  b_new = b_old - η × ∂C/∂b                                             │
│                                                                         │
│  Where η (eta) is the LEARNING RATE                                     │
│                                                                         │
│  Intuition: Move in the opposite direction of the gradient             │
│             (gradient points "uphill", we want to go "downhill")       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mini-Batch SGD

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MINI-BATCH APPROACH                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Instead of computing gradient over ALL 50,000 training examples:      │
│                                                                         │
│  1. Shuffle the training data                                           │
│                                                                         │
│  2. Split into mini-batches (e.g., 10 examples each)                   │
│     [batch_1] [batch_2] [batch_3] ... [batch_5000]                     │
│                                                                         │
│  3. For each batch:                                                     │
│     - Compute gradient for each example in batch                        │
│     - Average the gradients                                             │
│     - Update weights once                                               │
│                                                                         │
│  Benefits:                                                              │
│  ✓ Faster: Don't wait for all 50,000 examples                          │
│  ✓ Regularization: Noise helps escape local minima                     │
│  ✓ Memory: Only need to hold batch_size examples                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Learning Rate Effect

```
         Cost                              Cost                    Cost
           │                                │                        │
           │  η too HIGH                    │  η just RIGHT          │  η too LOW
           │  (unstable)                    │  (converges)           │  (slow)
           │                                │                        │
           │    ╱╲  ╱╲                     │ ╲                      │ ╲
           │   ╱  ╲╱  ╲   oscillates!      │  ╲                     │  ╲
           │  ╱        ╲                   │   ╲                    │   ╲
           │ ╱                             │    ╲__                 │    ╲
           │╱                              │       ╲___             │     ╲___
           └──────────────                 └──────────────          └──────────────
                 epochs                          epochs                   epochs
```

---

## 9. Training Process

### Complete Training Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING ALGORITHM                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOR epoch = 1 to num_epochs:                                          │
│  │                                                                      │
│  │   1. Shuffle training data                                          │
│  │                                                                      │
│  │   2. Split into mini-batches                                        │
│  │                                                                      │
│  │   3. FOR each mini_batch:                                           │
│  │      │                                                               │
│  │      │   Initialize: nabla_w = 0, nabla_b = 0                       │
│  │      │                                                               │
│  │      │   FOR each (x, y) in mini_batch:                             │
│  │      │   │                                                          │
│  │      │   │   • Feedforward: compute all a and z                     │
│  │      │   │   • Backprop: compute δ for each layer                   │
│  │      │   │   • Accumulate: nabla_w += ∂C/∂w, nabla_b += ∂C/∂b      │
│  │      │   │                                                          │
│  │      │   END FOR                                                    │
│  │      │                                                               │
│  │      │   Update weights:                                            │
│  │      │   w = w - (η/batch_size) × nabla_w                           │
│  │      │   b = b - (η/batch_size) × nabla_b                           │
│  │      │                                                               │
│  │      END FOR                                                        │
│  │                                                                      │
│  │   4. Optionally: evaluate on test set, print progress               │
│  │                                                                      │
│  END FOR                                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Typical Training Results

```
Network [784, 30, 10], η=3.0, batch_size=10:

Epoch  0: 9098 / 10000 = 90.98%  │████████████████████░░░│
Epoch  1: 9199 / 10000 = 91.99%  │█████████████████████░░│
Epoch  2: 9319 / 10000 = 93.19%  │█████████████████████░░│
Epoch  3: 9332 / 10000 = 93.32%  │█████████████████████░░│
Epoch  4: 9382 / 10000 = 93.82%  │█████████████████████░░│
Epoch  5: 9345 / 10000 = 93.45%  │█████████████████████░░│
Epoch  6: 9415 / 10000 = 94.15%  │█████████████████████░░│
Epoch  7: 9392 / 10000 = 93.92%  │█████████████████████░░│
Epoch  8: 9423 / 10000 = 94.23%  │██████████████████████░│
Epoch  9: 9431 / 10000 = 94.31%  │██████████████████████░│

Final accuracy: ~94%
```

---

## 10. Code Reference

### Creating a Network

```python
import network

# Create network with 784 input, 30 hidden, 10 output neurons
net = network.Network([784, 30, 10])

# Access weights and biases
print(net.weights[0].shape)  # (30, 784) - input to hidden
print(net.weights[1].shape)  # (10, 30)  - hidden to output
print(net.biases[0].shape)   # (30, 1)   - hidden layer
print(net.biases[1].shape)   # (10, 1)   - output layer
```

### Feedforward

```python
def feedforward(self, a):
    """Return the output of the network if 'a' is input."""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a
```

### Training

```python
import mnist_loader

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# Create and train network
net = network.Network([784, 30, 10])
net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)
```

### Making Predictions

```python
# Get a test image
x, y = test_data[0]

# Feedforward to get output
output = net.feedforward(x)

# Prediction is the index of highest activation
prediction = np.argmax(output)
print(f"Predicted: {prediction}, Actual: {y}")
```

---

## 11. Exercises

### Exercise 1: Experiment with Hidden Layer Size

Try different hidden layer sizes and observe the effect:

```python
for hidden_size in [10, 30, 100, 300]:
    net = network.Network([784, hidden_size, 10])
    net.SGD(training_data, epochs=5, mini_batch_size=10, eta=3.0, test_data=test_data)
```

**Questions:**
- How does accuracy change with hidden layer size?
- How does training time change?
- Is bigger always better?

### Exercise 2: Learning Rate Exploration

Try different learning rates:

```python
for eta in [0.1, 1.0, 3.0, 10.0, 30.0]:
    net = network.Network([784, 30, 10])
    net.SGD(training_data, epochs=5, mini_batch_size=10, eta=eta, test_data=test_data)
```

**Questions:**
- What happens with very small η?
- What happens with very large η?
- What's the "sweet spot"?

### Exercise 3: Manual Feedforward

Implement feedforward without using the class method:

```python
def my_feedforward(weights, biases, x):
    """Compute network output for input x."""
    a = x
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = 1.0 / (1.0 + np.exp(-z))  # sigmoid
    return a

# Test it
x, y = training_data[0]
my_output = my_feedforward(net.weights, net.biases, x)
net_output = net.feedforward(x)
print(f"Match: {np.allclose(my_output, net_output)}")
```

### Exercise 4: Visualize Training Progress

Modify SGD to track accuracy over epochs and plot it:

```python
import matplotlib.pyplot as plt

accuracies = []
for epoch in range(30):
    net.SGD(training_data, epochs=1, mini_batch_size=10, eta=3.0)
    accuracy = net.evaluate(test_data) / len(test_data)
    accuracies.append(accuracy)

plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.show()
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LEVEL 2 KEY CONCEPTS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ARCHITECTURE                                                           │
│  • Layers: input → hidden → output                                     │
│  • Weights connect layers, biases shift activations                    │
│  • [784, 30, 10] has 23,860 parameters                                 │
│                                                                         │
│  FEEDFORWARD                                                            │
│  • z = W·a + b (weighted sum)                                          │
│  • a = σ(z) (apply sigmoid)                                            │
│  • Repeat for each layer                                               │
│                                                                         │
│  BACKPROPAGATION                                                        │
│  • Compute error δ at output layer                                     │
│  • Propagate error backward through layers                             │
│  • Compute gradients ∂C/∂w and ∂C/∂b                                   │
│                                                                         │
│  TRAINING (SGD)                                                         │
│  • Split data into mini-batches                                        │
│  • Update: w = w - η × gradient                                        │
│  • Repeat for many epochs                                              │
│                                                                         │
│  RESULT: ~94% accuracy on MNIST with simple network!                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps: Level 3

Level 3 will improve our network with:

1. **Cross-Entropy Cost** - Faster learning, no vanishing gradients
2. **L2 Regularization** - Prevent overfitting
3. **Better Initialization** - Xavier/He initialization
4. **Learning Rate Schedules** - Adaptive learning rates

These techniques can push accuracy from 94% to 98%+!

