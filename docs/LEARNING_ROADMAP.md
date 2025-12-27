# Neural Networks and Deep Learning - Learning Roadmap

> A structured learning path from fundamentals to advanced concepts, following Michael Nielsen's book "Neural Networks and Deep Learning".

---

## 📚 Overview

This roadmap guides you through the codebase progressively, building understanding layer by layer (pun intended). Each module builds upon the previous one.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LEARNING PROGRESSION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 1: Foundations                                                    │
│  └── mnist_loader.py → Understanding data                                │
│                                                                          │
│  Level 2: Basic Neural Network                                           │
│  └── network.py → Feedforward, backpropagation, SGD                      │
│                                                                          │
│  Level 3: Improved Techniques                                            │
│  └── network2.py → Cost functions, regularization, initialization        │
│                                                                          │
│  Level 4: Deep Learning & CNNs                                           │
│  └── network3.py + conv.py → Convolutional networks, dropout, GPU        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Level 1: Foundations - Understanding the Data

**File:** `src/mnist_loader.py`  
**Book Chapter:** Chapter 1 (Introduction)  
**Estimated Time:** 1-2 hours  
**📖 Documentation:** [level1_data_understanding.md](level1_data_understanding.md)  
**🔬 Exploration Script:** `src/level1_explore_mnist.py`

### Learning Objectives
- [ ] Understand the MNIST dataset structure
- [ ] Learn how images are represented as vectors
- [ ] Understand one-hot encoding for labels

### Key Concepts

#### 1.1 The MNIST Dataset
```
MNIST = Modified National Institute of Standards and Technology

- 70,000 grayscale images of handwritten digits (0-9)
- Image size: 28 × 28 pixels = 784 pixels total
- Training set: 50,000 images
- Validation set: 10,000 images  
- Test set: 10,000 images
```

#### 1.2 Data Representation
```
Image (28×28) → Flattened Vector (784×1)

Each pixel value: 0.0 (white) to 1.0 (black)

Example: A "7" image becomes:
[0.0, 0.0, 0.1, 0.9, 0.8, ..., 0.0]  (784 values)
```

#### 1.3 Label Encoding (One-Hot)
```
Digit 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Digit 7 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

### Code to Study

```python
# mnist_loader.py - Key functions

def load_data():
    """Returns raw MNIST data as numpy arrays"""
    # training_data[0]: 50,000 images (each 784 pixels)
    # training_data[1]: 50,000 labels (digits 0-9)

def load_data_wrapper():
    """Returns data formatted for neural network training"""
    # Reshapes images to (784, 1) column vectors
    # Converts labels to one-hot vectors (10, 1)

def vectorized_result(j):
    """Converts digit j to one-hot vector"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
```

### Exercises
1. Load the data and visualize some digits using matplotlib
2. Print the shape of training inputs and outputs
3. Understand why we reshape to (784, 1) instead of (784,)

---

## 🧠 Level 2: Basic Neural Network

**File:** `src/network.py`  
**Book Chapter:** Chapters 1-2  
**Estimated Time:** 4-6 hours  
**📖 Documentation:** [level2_neural_network.md](level2_neural_network.md)  
**🔬 Exploration Script:** `src/level2_neural_network.py`

### Learning Objectives
- [x] Understand feedforward neural networks
- [x] Implement the sigmoid activation function
- [x] Understand and implement backpropagation
- [x] Implement stochastic gradient descent (SGD)

### Key Concepts

#### 2.1 Network Architecture
```
INPUT LAYER          HIDDEN LAYER(S)        OUTPUT LAYER
   (784)                (30-100)               (10)
    
    ○                      ○                    ○  → P(digit=0)
    ○                      ○                    ○  → P(digit=1)
    ○     ─────────→       ○      ─────────→    ○  → P(digit=2)
   ...      weights       ...       weights    ...
    ○                      ○                    ○  → P(digit=9)
   
  pixels              hidden neurons         predictions
```

#### 2.2 Sigmoid Activation Function
```
σ(z) = 1 / (1 + e^(-z))

Properties:
- Output range: (0, 1)
- Smooth, differentiable
- σ'(z) = σ(z) × (1 - σ(z))
```

#### 2.3 Feedforward Computation
```
For each layer:
    z = w · a + b       (weighted input)
    a' = σ(z)           (activation)

Where:
    w = weights matrix
    a = previous layer activations  
    b = bias vector
```

#### 2.4 Backpropagation Algorithm
```
Purpose: Compute gradients ∂C/∂w and ∂C/∂b efficiently

Steps:
1. Feedforward: compute all activations
2. Output error: δᴸ = ∇ₐC ⊙ σ'(zᴸ)
3. Backpropagate: δˡ = (wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(zˡ)
4. Gradients: ∂C/∂b = δ, ∂C/∂w = δ · aˡ⁻¹ᵀ
```

#### 2.5 Stochastic Gradient Descent (SGD)
```
For each epoch:
    1. Shuffle training data
    2. Divide into mini-batches
    3. For each mini-batch:
        - Compute gradients via backprop
        - Update: w → w - (η/m) × ∇w
        - Update: b → b - (η/m) × ∇b

Where:
    η = learning rate
    m = mini-batch size
```

### Code to Study

```python
# network.py - Key methods

class Network:
    def __init__(self, sizes):
        """sizes = [784, 30, 10] creates a 3-layer network"""
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """Return network output for input a"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def backprop(self, x, y):
        """Return (∇b, ∇w) for a single training example"""
        # Forward pass: store all z's and activations
        # Backward pass: compute deltas layer by layer
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train using mini-batch stochastic gradient descent"""
```

### Exercises
1. Create a network with `Network([784, 30, 10])`
2. Train it: `net.SGD(training_data, 30, 10, 3.0, test_data=test_data)`
3. Experiment with different architectures: `[784, 100, 10]`, `[784, 30, 30, 10]`
4. Try different learning rates: 0.1, 1.0, 3.0, 10.0
5. Manually trace through backprop for a tiny network `[2, 2, 1]`

### Expected Results
```
Epoch 0: 9038 / 10000
Epoch 1: 9182 / 10000
...
Epoch 29: 9492 / 10000  (~95% accuracy)
```

---

## ⚡ Level 3: Improved Techniques

**File:** `src/network2.py`  
**Book Chapter:** Chapters 3-4  
**Estimated Time:** 4-6 hours

### Learning Objectives
- [ ] Understand the cross-entropy cost function
- [ ] Implement L2 regularization
- [ ] Understand better weight initialization
- [ ] Learn about the vanishing gradient problem

### Key Concepts

#### 3.1 Cost Functions

**Quadratic Cost (Mean Squared Error)**
```
C = (1/2n) Σ ||y - a||²

Problem: Learning slowdown when σ(z) ≈ 0 or 1
         because σ'(z) appears in gradient
```

**Cross-Entropy Cost**
```
C = -(1/n) Σ [y·ln(a) + (1-y)·ln(1-a)]

Advantage: Gradient is (a - y), no σ'(z) term!
           Faster learning when predictions are wrong
```

#### 3.2 L2 Regularization (Weight Decay)
```
C = C₀ + (λ/2n) Σ w²

Purpose: Prevent overfitting by penalizing large weights

Update rule:
w → (1 - ηλ/n)w - (η/m) × ∇w
      ↑
  "weight decay" factor
```

#### 3.3 Weight Initialization
```
Old (network.py):
    w ~ N(0, 1)         ← Can cause saturation

New (network2.py):
    w ~ N(0, 1/√n_in)   ← Keeps activations reasonable

Where n_in = number of inputs to the neuron
```

#### 3.4 Overfitting vs Underfitting
```
┌────────────────────────────────────────────────────┐
│                                                    │
│  Training Accuracy    ████████████████  98%        │
│  Test Accuracy        ██████████        75%        │
│                       ↑                            │
│                    OVERFITTING!                    │
│                                                    │
│  Solutions:                                        │
│  • More training data                              │
│  • Regularization (L2)                             │
│  • Dropout (Level 4)                               │
│  • Early stopping                                  │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Code to Study

```python
# network2.py - Key improvements

class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        """Cross-entropy cost"""
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        """Output error - note: no σ'(z)!"""
        return (a - y)

class Network:
    def default_weight_initializer(self):
        """Xavier-like initialization"""
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """With L2 regularization"""
        # Weight update includes decay term:
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
```

### Exercises
1. Compare training with QuadraticCost vs CrossEntropyCost
2. Train without regularization, observe overfitting
3. Add regularization (λ=0.1, 1.0, 5.0), observe effect
4. Compare default_weight_initializer vs large_weight_initializer
5. Monitor both training and validation accuracy

### Expected Results
```
With cross-entropy + regularization:
Epoch 29: ~97-98% accuracy on test data
```

---

## 🚀 Level 4: Deep Learning & Convolutional Networks

**Files:** `src/network3.py`, `src/conv.py`  
**Book Chapter:** Chapters 5-6  
**Estimated Time:** 6-10 hours

### Learning Objectives
- [ ] Understand convolutional neural networks (CNNs)
- [ ] Learn about pooling layers
- [ ] Implement dropout regularization
- [ ] Understand the softmax output layer
- [ ] Run experiments on GPU (optional)

### Key Concepts

#### 4.1 Why CNNs for Images?
```
Fully Connected Problem:
- 784 inputs × 100 hidden = 78,400 weights (just first layer!)
- Ignores spatial structure of images
- No translation invariance

CNN Solution:
- Local receptive fields (5×5 filters)
- Weight sharing across image
- Much fewer parameters
- Built-in translation invariance
```

#### 4.2 Convolutional Layer
```
Input Image (28×28)
       ↓
   ┌─────┐
   │ 5×5 │ Filter (learns features)
   └─────┘
       ↓
Feature Map (24×24)    ← 28-5+1 = 24

Multiple filters → Multiple feature maps
20 filters → 20 feature maps
```

#### 4.3 Pooling Layer
```
Feature Map (24×24)
       ↓
   Max Pooling (2×2)
       ↓
Pooled Map (12×12)     ← 24/2 = 12

Takes maximum value in each 2×2 region
→ Reduces size, adds translation invariance
```

#### 4.4 CNN Architecture Example
```
INPUT           CONV+POOL        CONV+POOL         FC          OUTPUT
(28×28×1)    →  (12×12×20)   →   (4×4×40)     →  (100)    →   (10)
 
784 pixels     20 filters       40 filters      100 neurons   10 classes
               5×5, pool 2×2    5×5, pool 2×2    + dropout     softmax
```

#### 4.5 Dropout
```
During Training:
- Randomly "drop" neurons with probability p
- Prevents co-adaptation of neurons
- Like training ensemble of networks

During Testing:
- Use all neurons
- Scale outputs by (1-p)
```

#### 4.6 Activation Functions
```
Sigmoid:    σ(z) = 1/(1+e^(-z))     Range: (0, 1)
Tanh:       tanh(z)                  Range: (-1, 1)  
ReLU:       max(0, z)                Range: [0, ∞)   ← Preferred!

ReLU Advantages:
- No vanishing gradient (for z > 0)
- Computationally efficient
- Sparse activations
```

#### 4.7 Softmax Output Layer
```
softmax(zⱼ) = e^(zⱼ) / Σₖ e^(zₖ)

Properties:
- All outputs sum to 1
- Interpretable as probabilities
- Used with cross-entropy loss
```

### Code to Study

```python
# network3.py - Layer types

class ConvPoolLayer:
    """Convolutional layer followed by max-pooling"""
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        # filter_shape: (num_filters, num_input_maps, filter_height, filter_width)
        # image_shape: (batch_size, num_input_maps, image_height, image_width)

class FullyConnectedLayer:
    """Dense layer with optional dropout"""
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):

class SoftmaxLayer:
    """Output layer with softmax activation"""
    def __init__(self, n_in, n_out, p_dropout=0.0):
```

```python
# conv.py - Experiment functions

def shallow(n=3, epochs=60):
    """Baseline: FC only (no convolution)"""
    # ~97.5% accuracy

def basic_conv(n=3, epochs=60):
    """One conv layer + FC"""
    # ~98.5% accuracy
    
def dbl_conv(activation_fn=sigmoid):
    """Two conv layers + FC"""
    # ~99% accuracy

def double_fc_dropout(p0, p1, p2, repetitions):
    """Two conv + two FC with dropout"""
    # ~99.5% accuracy (state of the art for this architecture)
```

### Architecture Progression
```
┌─────────────────────────────────────────────────────────────────┐
│  Architecture                              Test Accuracy        │
├─────────────────────────────────────────────────────────────────┤
│  [784] → [100] → [10]                      ~97.5%               │
│  (fully connected only)                                         │
├─────────────────────────────────────────────────────────────────┤
│  [28×28] → Conv(20) → [100] → [10]         ~98.5%               │
│  (one conv layer)                                               │
├─────────────────────────────────────────────────────────────────┤
│  [28×28] → Conv(20) → Conv(40) → [100] → [10]    ~99%          │
│  (two conv layers)                                              │
├─────────────────────────────────────────────────────────────────┤
│  + ReLU activation                         ~99.2%               │
├─────────────────────────────────────────────────────────────────┤
│  + Expanded training data                  ~99.4%               │
├─────────────────────────────────────────────────────────────────┤
│  + Dropout + 1000 hidden neurons           ~99.6%               │
└─────────────────────────────────────────────────────────────────┘
```

### Exercises
1. Run `shallow()` to establish baseline
2. Run `basic_conv()` and compare
3. Run `dbl_conv(activation_fn=ReLU)` 
4. Generate expanded data with `expand_mnist.py`
5. Run `double_fc_dropout(0.5, 0.5, 0.5, 3)` for best results
6. Use `ensemble()` to combine multiple networks

---

## 📊 Summary: Concept Dependencies

```
                    ┌─────────────────┐
                    │  Linear Algebra │
                    │  (vectors,      │
                    │   matrices)     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │    Calculus       │         │   Probability     │
    │  (derivatives,    │         │  (distributions,  │
    │   chain rule)     │         │   Bayes)          │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Feedforward   │
                    │   Networks      │
                    │   (network.py)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│ Cost Functions│   │ Regularization  │   │ Initialization│
│ (cross-entropy│   │ (L2, dropout)   │   │ (Xavier)      │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Improved      │
                    │   Networks      │
                    │   (network2.py) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  Convolution      │         │  Activation Fns   │
    │  (local features) │         │  (ReLU, softmax)  │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │      CNNs       │
                    │  (network3.py)  │
                    │   (conv.py)     │
                    └─────────────────┘
```

---

## 🔧 Quick Reference: Running the Code

### Level 1-2: Basic Network
```python
import mnist_loader
import network

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create and train network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

### Level 3: Improved Network
```python
import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)
```

### Level 4: Convolutional Network
```python
import network3
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

training_data, validation_data, test_data = network3.load_data_shared()

net = network3.Network([
    ConvPoolLayer(image_shape=(10, 1, 28, 28), 
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(10, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)
], mini_batch_size=10)

net.SGD(training_data, 60, 10, 0.03, validation_data, test_data)
```

---

## 📖 Additional Resources

- **Book:** [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) by Michael Nielsen
- **3Blue1Brown:** [Neural Networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (visual explanations)
- **Stanford CS231n:** [Convolutional Neural Networks](http://cs231n.stanford.edu/) (more advanced)

---

## ✅ Progress Tracker

| Level | Topic | Status |
|-------|-------|--------|
| 1 | Data Loading (mnist_loader.py) | ✅ Complete |
| 2 | Basic Network (network.py) | ✅ Complete |
| 3 | Improved Network (network2.py) | ✅ Complete |
| 4 | CNNs (network3.py, conv.py) | 🔄 In Progress |

---

*Last updated: December 2024*

