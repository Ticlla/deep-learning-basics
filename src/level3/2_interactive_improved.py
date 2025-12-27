"""
Level 3: Interactive Digit Recognition (Improved Network)
==========================================================

Draw a digit with your mouse and watch the IMPROVED neural network predict it!

Features:
- Cross-Entropy Cost Function (faster learning)
- L2 Regularization (better generalization)
- Better Weight Initialization (1/√n)
- Larger hidden layer [784, 100, 10]
- Model persistence (saves/loads trained model)

Run from src/ directory:
    python level3/2_interactive_improved.py

Expected accuracy: ~97-98% (vs ~94% from Level 2)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tkinter as tk
from tkinter import ttk
import network2
import mnist_loader
from PIL import Image, ImageDraw, ImageOps
import os
import json


class ImprovedDigitRecognizer:
    """Interactive digit recognition with Level 3 improvements."""
    
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "level3_improved.json"
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Level 3: Improved Neural Network")
        self.root.configure(bg='#0d1117')
        
        # Canvas size (10x MNIST for better drawing)
        self.canvas_size = 280
        self.mnist_size = 28
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Load/train network
        self.net = self.load_or_train_network()
        
        # Setup UI
        self.setup_ui()
        
    def load_or_train_network(self):
        """Load trained network or train a new one."""
        # Ensure models directory exists
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        if self.MODEL_PATH.exists():
            print(f"✓ Loading saved model from {self.MODEL_PATH}")
            net = network2.load(str(self.MODEL_PATH))
            print("✓ Model loaded successfully!")
            
            # Quick test
            _, _, test_data = mnist_loader.load_data_wrapper()
            accuracy = net.accuracy(test_data) / len(test_data) * 100
            print(f"✓ Model accuracy: {accuracy:.2f}%")
            self.model_accuracy = accuracy
            return net
        
        # Train new model with Level 3 improvements
        print("\n" + "=" * 60)
        print("  Training new model with Level 3 improvements...")
        print("=" * 60)
        print("\nImprovements:")
        print("  ✓ Cross-Entropy Cost Function")
        print("  ✓ L2 Regularization (λ=5.0)")
        print("  ✓ Better Weight Initialization (1/√n)")
        print("  ✓ Larger hidden layer [784, 100, 10]")
        print("\nThis will take ~2-3 minutes...\n")
        
        # Load data
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        
        # Create improved network
        net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
        
        # Train with regularization
        net.SGD(
            training_data,
            epochs=30,
            mini_batch_size=10,
            eta=0.5,
            lmbda=5.0,
            evaluation_data=test_data,
            monitor_evaluation_accuracy=True
        )
        
        # Calculate final accuracy
        accuracy = net.accuracy(test_data) / len(test_data) * 100
        self.model_accuracy = accuracy
        
        print(f"\n🎯 Final accuracy: {accuracy:.2f}%")
        
        # Save the trained model
        net.save(str(self.MODEL_PATH))
        print(f"✓ Model saved to {self.MODEL_PATH}")
        print("  (Next time will load instantly!)")
        
        return net
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#0d1117')
        main_frame.pack(padx=20, pady=20)
        
        # Title
        title = tk.Label(
            main_frame,
            text="✏️ Draw a Digit (0-9)",
            font=('Helvetica', 24, 'bold'),
            fg='#58a6ff',
            bg='#0d1117'
        )
        title.pack(pady=(0, 5))
        
        # Subtitle with improvements
        subtitle = tk.Label(
            main_frame,
            text=f"Level 3 Network: ~{getattr(self, 'model_accuracy', 97):.1f}% accuracy",
            font=('Helvetica', 12),
            fg='#8b949e',
            bg='#0d1117'
        )
        subtitle.pack(pady=(0, 15))
        
        # Content frame (canvas + prediction)
        content = tk.Frame(main_frame, bg='#0d1117')
        content.pack()
        
        # Left side: Drawing canvas
        canvas_container = tk.Frame(content, bg='#0d1117')
        canvas_container.pack(side=tk.LEFT, padx=(0, 20))
        
        canvas_frame = tk.Frame(canvas_container, bg='#30363d', padx=3, pady=3)
        canvas_frame.pack()
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='#010409',
            cursor='crosshair',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Improvements panel under canvas
        improvements_frame = tk.Frame(canvas_container, bg='#161b22', padx=10, pady=8)
        improvements_frame.pack(fill=tk.X, pady=(10, 0))
        
        improvements_title = tk.Label(
            improvements_frame,
            text="Level 3 Improvements:",
            font=('Helvetica', 9, 'bold'),
            fg='#58a6ff',
            bg='#161b22'
        )
        improvements_title.pack(anchor='w')
        
        improvements = [
            "✓ Cross-Entropy Cost",
            "✓ L2 Regularization (λ=5.0)",
            "✓ 1/√n Weight Init",
            "✓ [784, 100, 10] Architecture"
        ]
        
        for imp in improvements:
            lbl = tk.Label(
                improvements_frame,
                text=imp,
                font=('Courier', 8),
                fg='#3fb950',
                bg='#161b22'
            )
            lbl.pack(anchor='w')
        
        # Right side: Prediction display
        pred_frame = tk.Frame(content, bg='#0d1117')
        pred_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Prediction label
        pred_title = tk.Label(
            pred_frame,
            text="Prediction:",
            font=('Helvetica', 14),
            fg='#8b949e',
            bg='#0d1117'
        )
        pred_title.pack()
        
        # Big prediction number
        self.pred_label = tk.Label(
            pred_frame,
            text="?",
            font=('Helvetica', 100, 'bold'),
            fg='#3fb950',
            bg='#0d1117',
            width=3
        )
        self.pred_label.pack(pady=5)
        
        # Confidence label
        self.confidence_label = tk.Label(
            pred_frame,
            text="",
            font=('Helvetica', 14),
            fg='#8b949e',
            bg='#0d1117'
        )
        self.confidence_label.pack()
        
        # Confidence bars
        self.confidence_frame = tk.Frame(pred_frame, bg='#0d1117')
        self.confidence_frame.pack(fill=tk.X, pady=15)
        
        self.confidence_bars = []
        self.confidence_labels = []
        self.digit_labels = []
        
        # Style for progress bars
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor='#21262d',
            background='#3fb950',
            bordercolor='#30363d',
            lightcolor='#3fb950',
            darkcolor='#3fb950'
        )
        style.configure(
            "gray.Horizontal.TProgressbar",
            troughcolor='#21262d',
            background='#6e7681',
            bordercolor='#30363d',
            lightcolor='#6e7681',
            darkcolor='#6e7681'
        )
        
        for i in range(10):
            row = tk.Frame(self.confidence_frame, bg='#0d1117')
            row.pack(fill=tk.X, pady=2)
            
            # Digit label
            digit_label = tk.Label(
                row,
                text=f"{i}:",
                font=('Courier', 11, 'bold'),
                fg='#8b949e',
                bg='#0d1117',
                width=2
            )
            digit_label.pack(side=tk.LEFT)
            self.digit_labels.append(digit_label)
            
            # Progress bar
            bar = ttk.Progressbar(
                row,
                length=150,
                mode='determinate',
                maximum=100,
                style="gray.Horizontal.TProgressbar"
            )
            bar.pack(side=tk.LEFT, padx=5)
            
            # Percentage label
            pct_label = tk.Label(
                row,
                text="0%",
                font=('Courier', 10),
                fg='#8b949e',
                bg='#0d1117',
                width=6
            )
            pct_label.pack(side=tk.LEFT)
            
            self.confidence_bars.append(bar)
            self.confidence_labels.append(pct_label)
        
        # Buttons
        btn_frame = tk.Frame(main_frame, bg='#0d1117')
        btn_frame.pack(pady=15)
        
        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Clear Canvas",
            font=('Helvetica', 12),
            command=self.clear_canvas,
            bg='#f85149',
            fg='white',
            activebackground='#da3633',
            activeforeground='white',
            padx=20,
            pady=8,
            relief=tk.FLAT,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="Draw a digit on the black canvas. Prediction updates in real-time!",
            font=('Helvetica', 10),
            fg='#6e7681',
            bg='#0d1117'
        )
        instructions.pack(pady=(10, 0))
        
    def start_draw(self, event):
        """Start drawing."""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """Draw line while mouse is moving."""
        if self.drawing:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                fill='white',
                width=22,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=22
            )
            
            self.last_x = event.x
            self.last_y = event.y
            
            # Predict in real-time
            self.predict()
    
    def stop_draw(self, event):
        """Stop drawing."""
        self.drawing = False
        self.predict()
        
    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="?", fg='#3fb950')
        self.confidence_label.config(text="")
        
        # Reset confidence bars
        for i in range(10):
            self.confidence_bars[i]['value'] = 0
            self.confidence_bars[i].configure(style="gray.Horizontal.TProgressbar")
            self.confidence_labels[i].config(text="0%", fg='#8b949e')
            self.digit_labels[i].config(fg='#8b949e')
    
    def predict(self):
        """Run prediction on current drawing."""
        # Resize to 28x28
        img_small = self.image.resize(
            (self.mnist_size, self.mnist_size),
            Image.Resampling.LANCZOS
        )
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img_small) / 255.0
        
        # Flatten to (784, 1) column vector
        input_vector = img_array.reshape(784, 1)
        
        # Feedforward through network
        output = self.net.feedforward(input_vector)
        
        # Get prediction
        prediction = np.argmax(output)
        confidence = output[prediction][0] * 100
        
        # Update prediction display
        self.pred_label.config(text=str(prediction))
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Color based on confidence
        if confidence > 80:
            self.pred_label.config(fg='#3fb950')  # Green
            self.confidence_label.config(fg='#3fb950')
        elif confidence > 50:
            self.pred_label.config(fg='#d29922')  # Orange
            self.confidence_label.config(fg='#d29922')
        else:
            self.pred_label.config(fg='#f85149')  # Red
            self.confidence_label.config(fg='#f85149')
        
        # Update confidence bars
        for i in range(10):
            pct = output[i][0] * 100
            self.confidence_bars[i]['value'] = pct
            self.confidence_labels[i].config(text=f"{pct:.1f}%")
            
            # Highlight the predicted digit
            if i == prediction:
                self.confidence_bars[i].configure(style="green.Horizontal.TProgressbar")
                self.confidence_labels[i].config(fg='#3fb950')
                self.digit_labels[i].config(fg='#3fb950')
            else:
                self.confidence_bars[i].configure(style="gray.Horizontal.TProgressbar")
                self.confidence_labels[i].config(fg='#8b949e')
                self.digit_labels[i].config(fg='#8b949e')


def main():
    print("\n" + "=" * 60)
    print("  🧠 Level 3: Improved Neural Network")
    print("=" * 60)
    print("\nStarting application...")
    
    root = tk.Tk()
    root.resizable(False, False)
    app = ImprovedDigitRecognizer(root)
    
    print("\n" + "=" * 60)
    print("  ✓ Draw on the canvas to test the network!")
    print("=" * 60 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    main()

