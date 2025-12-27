"""
Level 2: Interactive Digit Recognition
========================================

Draw a digit with your mouse and watch the neural network predict it in real-time!

Run from src/ directory:
    python level2/2_interactive_digit_recognition.py

Requirements:
    - tkinter (built-in with Python)
    - Trained network (trains automatically if needed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tkinter as tk
from tkinter import ttk
import network
import mnist_loader
from PIL import Image, ImageDraw, ImageOps


class DigitRecognizer:
    """Interactive digit recognition application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Neural Network Digit Recognizer")
        self.root.configure(bg='#1e1e1e')
        
        # Canvas size (10x MNIST for better drawing)
        self.canvas_size = 280
        self.mnist_size = 28
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create PIL image for drawing (we'll resize this to 28x28)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Load/train network
        self.net = self.load_or_train_network()
        
        # Setup UI
        self.setup_ui()
        
    def load_or_train_network(self):
        """Load trained network or train a new one."""
        print("Loading MNIST data...")
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        
        print("Creating and training network [784, 30, 10]...")
        print("This may take ~30 seconds...\n")
        
        net = network.Network([784, 30, 10])
        net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)
        
        print("\n✓ Network ready!")
        return net
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(padx=20, pady=20)
        
        # Title
        title = tk.Label(
            main_frame,
            text="✏️ Draw a Digit (0-9)",
            font=('Helvetica', 24, 'bold'),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        title.pack(pady=(0, 15))
        
        # Content frame (canvas + prediction)
        content = tk.Frame(main_frame, bg='#1e1e1e')
        content.pack()
        
        # Left side: Drawing canvas
        canvas_frame = tk.Frame(content, bg='#333333', padx=3, pady=3)
        canvas_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='crosshair',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Right side: Prediction display
        pred_frame = tk.Frame(content, bg='#1e1e1e')
        pred_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Prediction label
        pred_title = tk.Label(
            pred_frame,
            text="Prediction:",
            font=('Helvetica', 14),
            fg='#888888',
            bg='#1e1e1e'
        )
        pred_title.pack()
        
        # Big prediction number
        self.pred_label = tk.Label(
            pred_frame,
            text="?",
            font=('Helvetica', 120, 'bold'),
            fg='#00ff88',
            bg='#1e1e1e',
            width=3
        )
        self.pred_label.pack(pady=10)
        
        # Confidence bars
        self.confidence_frame = tk.Frame(pred_frame, bg='#1e1e1e')
        self.confidence_frame.pack(fill=tk.X, pady=10)
        
        self.confidence_bars = []
        self.confidence_labels = []
        
        for i in range(10):
            row = tk.Frame(self.confidence_frame, bg='#1e1e1e')
            row.pack(fill=tk.X, pady=1)
            
            # Digit label
            digit_label = tk.Label(
                row,
                text=f"{i}:",
                font=('Courier', 10),
                fg='#888888',
                bg='#1e1e1e',
                width=2
            )
            digit_label.pack(side=tk.LEFT)
            
            # Progress bar
            bar = ttk.Progressbar(
                row,
                length=150,
                mode='determinate',
                maximum=100
            )
            bar.pack(side=tk.LEFT, padx=5)
            
            # Percentage label
            pct_label = tk.Label(
                row,
                text="0%",
                font=('Courier', 10),
                fg='#888888',
                bg='#1e1e1e',
                width=5
            )
            pct_label.pack(side=tk.LEFT)
            
            self.confidence_bars.append(bar)
            self.confidence_labels.append(pct_label)
        
        # Buttons
        btn_frame = tk.Frame(main_frame, bg='#1e1e1e')
        btn_frame.pack(pady=15)
        
        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Clear",
            font=('Helvetica', 14),
            command=self.clear_canvas,
            bg='#ff4444',
            fg='white',
            padx=20,
            pady=10
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="Draw a digit on the black canvas. The network predicts in real-time!",
            font=('Helvetica', 10),
            fg='#666666',
            bg='#1e1e1e'
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
                width=20,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=20
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
        self.pred_label.config(text="?")
        
        # Reset confidence bars
        for i in range(10):
            self.confidence_bars[i]['value'] = 0
            self.confidence_labels[i].config(text="0%")
    
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
        
        # Color based on confidence
        if confidence > 80:
            self.pred_label.config(fg='#00ff88')  # Green
        elif confidence > 50:
            self.pred_label.config(fg='#ffaa00')  # Orange
        else:
            self.pred_label.config(fg='#ff4444')  # Red
        
        # Update confidence bars
        for i in range(10):
            pct = output[i][0] * 100
            self.confidence_bars[i]['value'] = pct
            self.confidence_labels[i].config(text=f"{pct:.0f}%")
            
            # Highlight the predicted digit
            if i == prediction:
                self.confidence_labels[i].config(fg='#00ff88')
            else:
                self.confidence_labels[i].config(fg='#888888')


def main():
    print("\n" + "=" * 60)
    print("  🧠 Interactive Digit Recognition")
    print("=" * 60)
    print("\nStarting application...")
    print("A window will open where you can draw digits!\n")
    
    root = tk.Tk()
    app = DigitRecognizer(root)
    
    print("\n" + "=" * 60)
    print("  Draw on the canvas to test the neural network!")
    print("=" * 60)
    
    root.mainloop()


if __name__ == "__main__":
    main()


