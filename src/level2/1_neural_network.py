"""
Level 2: Building Your First Neural Network
=============================================

This script takes you through the fundamentals of neural networks
step by step, from architecture to training.

Topics covered:
1. Network Architecture (layers, neurons, weights, biases)
2. Sigmoid Activation Function
3. Feedforward Computation
4. Backpropagation Algorithm
5. Stochastic Gradient Descent
6. Training and Evaluation

Run from project root:
    python src/level2/1_neural_network.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import network
from utils import level2_picture, reset_run_timestamp


def section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# =============================================================================
# SECTION 1: Network Architecture
# =============================================================================

def explain_architecture():
    """
    Explain the structure of a neural network.
    """
    section_header("SECTION 1: Network Architecture")
    
    print("""
    A neural network is organized in LAYERS:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   INPUT LAYER        HIDDEN LAYER        OUTPUT LAYER              │
    │   (784 neurons)      (30 neurons)        (10 neurons)              │
    │                                                                     │
    │       ○                   ○                   ○  → P(digit=0)       │
    │       ○                   ○                   ○  → P(digit=1)       │
    │       ○        ───►       ○        ───►       ○  → P(digit=2)       │
    │      ...      weights    ...      weights    ...                   │
    │       ○                   ○                   ○  → P(digit=9)       │
    │                                                                     │
    │    (pixels)           (features)          (predictions)            │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    Network [784, 30, 10] means:
    - Input layer:  784 neurons (one per pixel)
    - Hidden layer: 30 neurons  (learned features)
    - Output layer: 10 neurons  (one per digit 0-9)
    """)
    
    # Create a network and examine its structure
    print("Let's create a network and examine its structure:\n")
    print(">>> net = network.Network([784, 30, 10])")
    
    net = network.Network([784, 30, 10])
    
    print(f"\nNetwork properties:")
    print(f"  Number of layers: {net.num_layers}")
    print(f"  Layer sizes: {net.sizes}")
    print(f"  Number of weight matrices: {len(net.weights)}")
    print(f"  Number of bias vectors: {len(net.biases)}")
    
    print("\n--- Weight Matrices ---")
    for i, w in enumerate(net.weights):
        print(f"  weights[{i}] shape: {w.shape}")
        print(f"    - Connects layer {i} ({net.sizes[i]} neurons)")
        print(f"    - To layer {i+1} ({net.sizes[i+1]} neurons)")
        print(f"    - Total parameters: {w.size:,}")
    
    print("\n--- Bias Vectors ---")
    for i, b in enumerate(net.biases):
        print(f"  biases[{i}] shape: {b.shape}")
        print(f"    - For layer {i+1} ({net.sizes[i+1]} neurons)")
    
    # Total parameters
    total_weights = sum(w.size for w in net.weights)
    total_biases = sum(b.size for b in net.biases)
    print(f"\n📊 TOTAL PARAMETERS: {total_weights + total_biases:,}")
    print(f"   - Weights: {total_weights:,}")
    print(f"   - Biases: {total_biases:,}")
    
    return net


def visualize_weights(net):
    """
    Visualize the initial random weights.
    """
    print("\n--- Visualizing Initial Weights ---\n")
    
    # First layer weights: (30, 784) - each row is a filter for one hidden neuron
    w1 = net.weights[0]
    
    print(f"First layer weights shape: {w1.shape}")
    print("Each row (30 total) represents what pattern a hidden neuron looks for.")
    print("Reshaping each row to 28x28 shows the 'filter' for each neuron.\n")
    
    # Show first 16 hidden neurons' weight patterns
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Initial Random Weights (First 16 Hidden Neurons)\nBefore Training", 
                 fontsize=14)
    
    for idx, ax in enumerate(axes.flatten()):
        weights_2d = w1[idx].reshape(28, 28)
        im = ax.imshow(weights_2d, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f"Neuron {idx}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = level2_picture("initial_weights")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    print("\nNote: These are RANDOM - no meaningful patterns yet!")
    print("After training, they will learn useful features.")


# =============================================================================
# SECTION 2: Sigmoid Activation Function
# =============================================================================

def explain_sigmoid():
    """
    Explain the sigmoid activation function.
    """
    section_header("SECTION 2: Sigmoid Activation Function")
    
    print("""
    The sigmoid function "squashes" any input to a value between 0 and 1:
    
                    1
    σ(z) = ─────────────
            1 + e^(-z)
    
    Properties:
    - Output range: (0, 1) - like a probability!
    - σ(0) = 0.5
    - σ(large positive) → 1
    - σ(large negative) → 0
    - Smooth and differentiable everywhere
    """)
    
    # Plot sigmoid
    z = np.linspace(-10, 10, 1000)
    sig = network.sigmoid(z)
    sig_prime = network.sigmoid_prime(z)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sigmoid function
    ax = axes[0]
    ax.plot(z, sig, 'b-', linewidth=2, label='σ(z)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('z (weighted input)', fontsize=12)
    ax.set_ylabel('σ(z) (activation)', fontsize=12)
    ax.set_title('Sigmoid Function: σ(z) = 1/(1+e⁻ᶻ)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Mark key points
    ax.plot(0, 0.5, 'ro', markersize=10)
    ax.annotate('σ(0) = 0.5', xy=(0, 0.5), xytext=(2, 0.6),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    # Sigmoid derivative
    ax = axes[1]
    ax.plot(z, sig_prime, 'r-', linewidth=2, label="σ'(z)")
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('z (weighted input)', fontsize=12)
    ax.set_ylabel("σ'(z) (derivative)", fontsize=12)
    ax.set_title("Sigmoid Derivative: σ'(z) = σ(z)(1-σ(z))", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark maximum
    ax.plot(0, 0.25, 'ro', markersize=10)
    ax.annotate("Max at z=0\nσ'(0) = 0.25", xy=(0, 0.25), xytext=(2, 0.2),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    fig_path = level2_picture("sigmoid")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    
    # Show code
    print("\nCode implementation:")
    print("""
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(z):
        # σ'(z) = σ(z) * (1 - σ(z))
        return sigmoid(z) * (1 - sigmoid(z))
    """)
    
    # Demo values
    print("\nExample values:")
    test_values = [-5, -2, 0, 2, 5]
    for v in test_values:
        print(f"  σ({v:2}) = {network.sigmoid(v):.4f}  |  σ'({v:2}) = {network.sigmoid_prime(v):.4f}")


# =============================================================================
# SECTION 3: Feedforward Computation
# =============================================================================

def explain_feedforward():
    """
    Explain how data flows through the network.
    """
    section_header("SECTION 3: Feedforward Computation")
    
    print("""
    Feedforward = computing the output given an input.
    
    For each layer:
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   z = W · a + b         (weighted input)                          │
    │   a' = σ(z)             (activation - apply sigmoid)              │
    │                                                                    │
    │   Where:                                                           │
    │   - W: weight matrix (connects previous to current layer)         │
    │   - a: activations from previous layer                            │
    │   - b: biases for current layer                                   │
    │   - σ: sigmoid function                                           │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
    """)
    
    # Load one image and run feedforward
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    x, y = training_data[0]
    actual_digit = np.argmax(y)
    
    print(f"Let's trace feedforward for one image (digit {actual_digit}):\n")
    
    net = network.Network([784, 30, 10])
    
    print("Step-by-step computation:")
    print("-" * 50)
    
    # Manual feedforward with printing
    a = x  # Start with input
    print(f"Input a⁰ shape: {a.shape}")
    
    for layer_idx, (w, b) in enumerate(zip(net.weights, net.biases)):
        print(f"\n--- Layer {layer_idx + 1} ---")
        print(f"W{layer_idx+1} shape: {w.shape}")
        print(f"b{layer_idx+1} shape: {b.shape}")
        
        z = np.dot(w, a) + b
        print(f"z{layer_idx+1} = W·a + b, shape: {z.shape}")
        
        a = network.sigmoid(z)
        print(f"a{layer_idx+1} = σ(z), shape: {a.shape}")
        
        if layer_idx == len(net.weights) - 1:
            print(f"\nOutput activations (probabilities):")
            for digit, prob in enumerate(a.flatten()):
                bar = "█" * int(prob * 50)
                print(f"  P(digit={digit}): {prob:.4f} {bar}")
    
    prediction = np.argmax(a)
    print(f"\n🎯 Network predicts: {prediction}")
    print(f"   Actual label: {actual_digit}")
    print(f"   Correct: {'✓' if prediction == actual_digit else '✗'}")
    print("\n(Note: Random weights = random prediction. Training will fix this!)")
    
    # Show feedforward code
    print("\nCode implementation:")
    print("""
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    """)
    
    return net, x, y


# =============================================================================
# SECTION 4: Backpropagation Algorithm
# =============================================================================

def explain_backpropagation():
    """
    Explain the backpropagation algorithm.
    """
    section_header("SECTION 4: Backpropagation Algorithm")
    
    print("""
    Backpropagation = computing gradients for learning.
    
    Goal: Find how much each weight/bias affects the error,
          so we can adjust them to reduce the error.
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    THE 4 BACKPROP EQUATIONS                        ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  (BP1) δᴸ = ∇ₐC ⊙ σ'(zᴸ)         Output layer error              ║
    ║                                                                    ║
    ║  (BP2) δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(zˡ)  Error in terms of next layer ║
    ║                                                                    ║
    ║  (BP3) ∂C/∂bˡ = δˡ                 Gradient for biases            ║
    ║                                                                    ║
    ║  (BP4) ∂C/∂wˡ = δˡ (aˡ⁻¹)ᵀ         Gradient for weights           ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    Where:
    - δ: error (how much this layer contributed to the final error)
    - ⊙: element-wise multiplication (Hadamard product)
    - ∇ₐC: derivative of cost with respect to output activations
    """)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      BACKPROP FLOW                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   FORWARD PASS (left → right):                                      │
    │   input → hidden → output → cost                                    │
    │                                                                     │
    │   BACKWARD PASS (right → left):                                     │
    │   cost → δ_output → δ_hidden → gradients                           │
    │                                                                     │
    │   Step 1: Compute output error δᴸ                                  │
    │   Step 2: Propagate error backward to get δ for each layer         │
    │   Step 3: Compute gradients ∂C/∂w and ∂C/∂b                        │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    # Visualize the gradient flow
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Draw layers
    layer_x = [0.1, 0.4, 0.7]
    layer_names = ['Input\n(784)', 'Hidden\n(30)', 'Output\n(10)']
    
    for i, (x, name) in enumerate(zip(layer_x, layer_names)):
        # Draw neurons
        n_neurons = [5, 4, 3][i]  # Simplified
        for j in range(n_neurons):
            y = 0.2 + j * 0.15
            circle = plt.Circle((x, y), 0.03, fill=True, 
                               color=['lightblue', 'lightgreen', 'salmon'][i])
            ax.add_patch(circle)
        ax.text(x, 0.95, name, ha='center', fontsize=12)
    
    # Forward arrows (blue)
    ax.annotate('', xy=(0.35, 0.5), xytext=(0.15, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.45, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(0.25, 0.55, 'Forward\n(compute a)', ha='center', fontsize=10, color='blue')
    ax.text(0.55, 0.55, 'Forward\n(compute a)', ha='center', fontsize=10, color='blue')
    
    # Backward arrows (red)
    ax.annotate('', xy=(0.45, 0.3), xytext=(0.65, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(0.15, 0.3), xytext=(0.35, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.55, 0.22, 'Backward\n(compute δ)', ha='center', fontsize=10, color='red')
    ax.text(0.25, 0.22, 'Backward\n(compute δ)', ha='center', fontsize=10, color='red')
    
    # Cost at the end
    ax.text(0.85, 0.5, 'Cost\nC', ha='center', fontsize=14, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.annotate('', xy=(0.8, 0.5), xytext=(0.73, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Backpropagation: Forward Pass (blue) → Backward Pass (red)', fontsize=14)
    
    plt.tight_layout()
    fig_path = level2_picture("backprop_flow")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    
    # Show code
    print("\nCode implementation (simplified):")
    print("""
    def backprop(self, x, y):
        # Forward pass - store activations and z values
        activations = [x]
        zs = []
        a = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        
        # Backward pass
        # (BP1) Output error
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta                                    # (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].T)         # (BP4)
        
        # (BP2) Propagate error backward
        for l in range(2, num_layers):
            delta = np.dot(weights[-l+1].T, delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta                                # (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)   # (BP4)
        
        return nabla_b, nabla_w
    """)


# =============================================================================
# SECTION 5: Stochastic Gradient Descent
# =============================================================================

def explain_sgd():
    """
    Explain the SGD training algorithm.
    """
    section_header("SECTION 5: Stochastic Gradient Descent (SGD)")
    
    print("""
    SGD is HOW we use gradients to improve the network.
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    GRADIENT DESCENT IDEA                           ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║   Cost function C is like a landscape with hills and valleys.     ║
    ║   We want to find the LOWEST point (minimum cost).                ║
    ║                                                                    ║
    ║   Gradient tells us the direction of steepest ASCENT.             ║
    ║   So we go in the OPPOSITE direction (negative gradient).         ║
    ║                                                                    ║
    ║   Update rule:                                                     ║
    ║       w_new = w_old - η * ∂C/∂w                                   ║
    ║       b_new = b_old - η * ∂C/∂b                                   ║
    ║                                                                    ║
    ║   Where η (eta) is the LEARNING RATE.                             ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("""
    Why "Stochastic"? Why "Mini-Batch"?
    ────────────────────────────────────
    
    - BATCH gradient descent: Use ALL 50,000 images per update (SLOW)
    - STOCHASTIC GD: Use 1 image per update (NOISY)
    - MINI-BATCH GD: Use small batches (e.g., 10 images) per update (BEST)
    
    Mini-batch benefits:
    ✓ Faster than full batch (don't need all data)
    ✓ Less noisy than single samples
    ✓ Allows parallelization (GPU-friendly)
    ✓ Provides regularization effect
    """)
    
    print("""
    Training Loop (per epoch):
    ──────────────────────────
    
    1. Shuffle training data (randomize order)
    2. Split into mini-batches (e.g., 10 images each)
    3. For each mini-batch:
       a. Compute gradients using backprop
       b. Average gradients across batch
       c. Update weights: w -= η * avg_gradient
    4. Repeat for all epochs
    """)
    
    # Visualize SGD
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost landscape
    ax = axes[0]
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple bowl
    
    ax.contour(X, Y, Z, levels=20, cmap='viridis')
    
    # SGD path
    path = [(2.5, 2.5)]
    eta = 0.3
    for _ in range(10):
        x, y = path[-1]
        grad_x, grad_y = 2*x, 2*y
        new_x = x - eta * grad_x + np.random.normal(0, 0.2)  # Add noise
        new_y = y - eta * grad_y + np.random.normal(0, 0.2)
        path.append((new_x, new_y))
    
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=8, linewidth=2)
    ax.plot(0, 0, 'g*', markersize=20, label='Minimum')
    ax.set_xlabel('Weight w₁')
    ax.set_ylabel('Weight w₂')
    ax.set_title('SGD on Cost Landscape\n(finding minimum cost)')
    ax.legend()
    
    # Learning rate comparison
    ax = axes[1]
    epochs = np.arange(30)
    
    # Simulated cost curves
    cost_high_lr = 2 * np.exp(-0.5 * epochs) + np.random.normal(0, 0.1, 30) + 0.5 * (np.sin(epochs) > 0)
    cost_good_lr = 2 * np.exp(-0.3 * epochs) + np.random.normal(0, 0.02, 30)
    cost_low_lr = 2 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.01, 30)
    
    ax.plot(epochs, cost_high_lr, 'r-', label='η=3.0 (too high - unstable)', linewidth=2)
    ax.plot(epochs, cost_good_lr, 'g-', label='η=0.3 (good)', linewidth=2)
    ax.plot(epochs, cost_low_lr, 'b-', label='η=0.01 (too low - slow)', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.set_title('Effect of Learning Rate η')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = level2_picture("sgd")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    
    # Show code
    print("\nCode implementation:")
    print("""
    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                           for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
    
    def update_mini_batch(self, mini_batch, eta):
        # Accumulate gradients
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in ...]
            nabla_w = [nw + dnw for nw, dnw in ...]
        
        # Update weights and biases
        self.weights = [w - (eta/batch_size)*nw for w, nw in ...]
        self.biases = [b - (eta/batch_size)*nb for b, nb in ...]
    """)


# =============================================================================
# SECTION 6: Training and Evaluation
# =============================================================================

def train_and_evaluate():
    """
    Train the network and watch it learn!
    """
    section_header("SECTION 6: Training and Evaluation")
    
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    
    print(f"Training set: {len(training_data):,} images")
    print(f"Test set: {len(test_data):,} images")
    
    print("\n" + "─" * 50)
    print("Creating network [784, 30, 10]...")
    print("─" * 50)
    
    net = network.Network([784, 30, 10])
    
    print("""
    Training parameters:
    - Epochs: 10
    - Mini-batch size: 10
    - Learning rate (η): 3.0
    """)
    
    print("Starting training...")
    print("-" * 50)
    
    # Train the network
    net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0,
            test_data=test_data)
    
    print("-" * 50)
    
    # Final evaluation
    correct = net.evaluate(test_data)
    accuracy = 100 * correct / len(test_data)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Test accuracy: {correct} / {len(test_data)} = {accuracy:.2f}%")
    
    # Visualize some predictions
    visualize_predictions(net, test_data)
    
    # Visualize learned weights
    visualize_learned_weights(net)
    
    return net


def visualize_predictions(net, test_data):
    """
    Show some example predictions.
    """
    print("\n--- Sample Predictions ---\n")
    
    # Get random samples
    indices = np.random.choice(len(test_data), 20, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle("Network Predictions on Test Data", fontsize=14)
    
    correct = 0
    for ax, idx in zip(axes.flatten(), indices):
        x, y = test_data[idx]
        prediction = np.argmax(net.feedforward(x))
        actual = y
        
        ax.imshow(x.reshape(28, 28), cmap='gray')
        
        if prediction == actual:
            color = 'green'
            correct += 1
        else:
            color = 'red'
        
        ax.set_title(f"Pred: {prediction}, True: {actual}", 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = level2_picture("predictions")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    print(f"   Sample accuracy: {correct}/20 ({100*correct/20:.0f}%)")


def visualize_learned_weights(net):
    """
    Visualize what the hidden neurons learned.
    """
    print("\n--- Learned Weight Patterns ---\n")
    
    w1 = net.weights[0]
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Learned Weight Patterns (First 16 Hidden Neurons)\nAfter Training", 
                 fontsize=14)
    
    for idx, ax in enumerate(axes.flatten()):
        weights_2d = w1[idx].reshape(28, 28)
        im = ax.imshow(weights_2d, cmap='RdBu')
        ax.set_title(f"Neuron {idx}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = level2_picture("learned_weights")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    print("\nNotice: The learned weights show patterns - these are")
    print("features the network learned to recognize digits!")


# =============================================================================
# MAIN
# =============================================================================

def main(interactive: bool = False):
    # Start a fresh run (creates new folder with timestamp)
    reset_run_timestamp()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "LEVEL 2: BUILDING YOUR FIRST NEURAL NETWORK" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")
    
    def wait(msg: str) -> None:
        if interactive:
            input(msg)
        else:
            print(msg.replace("Press Enter to", "Continuing to"))
    
    # Section 1: Architecture
    net = explain_architecture()
    visualize_weights(net)
    wait("\nPress Enter to continue to Sigmoid function...")
    
    # Section 2: Sigmoid
    explain_sigmoid()
    wait("\nPress Enter to continue to Feedforward...")
    
    # Section 3: Feedforward
    explain_feedforward()
    wait("\nPress Enter to continue to Backpropagation...")
    
    # Section 4: Backpropagation
    explain_backpropagation()
    wait("\nPress Enter to continue to SGD...")
    
    # Section 5: SGD
    explain_sgd()
    wait("\nPress Enter to train the network...")
    
    # Section 6: Training
    net = train_and_evaluate()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🎉 LEVEL 2 COMPLETE! 🎉" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("""
    You've learned:
    
    ✅ Network architecture (layers, weights, biases)
    ✅ Sigmoid activation function
    ✅ Feedforward computation
    ✅ Backpropagation algorithm
    ✅ Stochastic Gradient Descent
    ✅ Training and evaluation
    
    → Ready for Level 3: Improved Techniques! 🚀
       (Cross-entropy, regularization, better initialization)
    """)


if __name__ == "__main__":
    main()

