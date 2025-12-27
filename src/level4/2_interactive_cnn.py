#!/usr/bin/env python3
"""
Level 4: Interactive CNN Digit Recognition
==========================================

Draw digits with your mouse and see real-time predictions from the trained CNN!
Demonstrates the power of Convolutional Neural Networks.

Run from project root:
    cd /home/alcidesticlla/Documents/MOOC/mniels/neural-networks-and-deep-learning/src
    python3 level4/2_interactive_cnn.py

Features:
- Draw digits with mouse
- Real-time CNN predictions with confidence
- Visualize what the CNN "sees" (preprocessed input)
- Compare with Level 3 accuracy (~97.7% → 99.3%!)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import data loader for testing
import mnist_loader


# =============================================================================
# CNN MODEL (same as training script)
# =============================================================================

class AdvancedCNN(nn.Module):
    """Advanced CNN with BatchNorm + Dropout (~99.3% accuracy)"""
    def __init__(self, dropout_rate=0.5):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40 * 4 * 4, 100)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 40 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def get_probabilities(self, x):
        """Get softmax probabilities for visualization"""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.view(-1, 1, 28, 28)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 40 * 4 * 4)
            x = F.relu(self.fc1(x))
            # No dropout during inference
            x = self.fc2(x)
            return F.softmax(x, dim=1)


# =============================================================================
# INTERACTIVE GUI
# =============================================================================

class CNNDigitRecognizer:
    """Interactive digit recognizer using trained CNN"""
    
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "level4_cnn.pt"
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Level 4: CNN Digit Recognition")
        self.root.configure(bg='#1a1a2e')
        
        # Drawing settings
        self.canvas_size = 280  # 10x the 28x28 input
        self.brush_size = 18
        
        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Load the CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
        # Bind events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
    
    def load_model(self):
        """Load trained CNN model"""
        model = AdvancedCNN()
        
        if self.MODEL_PATH.exists():
            print(f"✓ Loading CNN model from {self.MODEL_PATH}")
            model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            model.to(self.device)
            model.eval()
            print(f"✓ Model loaded! Using device: {self.device}")
            
            # Test accuracy
            _, _, test_data = mnist_loader.load_data_wrapper()
            test_x = torch.tensor([x.flatten() for x, _ in test_data], dtype=torch.float32)
            test_y = torch.tensor([y for _, y in test_data], dtype=torch.long)
            
            with torch.no_grad():
                test_x = test_x.to(self.device)
                outputs = model(test_x)
                predictions = outputs.argmax(dim=1).cpu()
                accuracy = (predictions == test_y).float().mean() * 100
            
            self.model_accuracy = accuracy.item()
            print(f"✓ Model accuracy: {self.model_accuracy:.2f}%")
        else:
            print(f"⚠ Model not found at {self.MODEL_PATH}")
            print("  Run: python3 level4/1_cnn_pytorch.py first!")
            self.model_accuracy = 0.0
        
        return model
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Big.TLabel', font=('Helvetica', 72, 'bold'))
        style.configure('Info.TLabel', font=('Helvetica', 11))
        
        # Header
        header = ttk.Label(main_frame, 
                          text="🧠 CNN Digit Recognition (Level 4)",
                          style='Title.TLabel')
        header.pack(pady=(0, 5))
        
        # Accuracy label
        acc_text = f"Model Accuracy: {self.model_accuracy:.2f}% (vs ~97.7% in Level 3)"
        acc_label = ttk.Label(main_frame, text=acc_text, style='Info.TLabel')
        acc_label.pack(pady=(0, 10))
        
        # Architecture info
        arch_text = "Architecture: Conv(20)→Pool→Conv(40)→Pool→FC(100)→Dropout→Output(10)"
        arch_label = ttk.Label(main_frame, text=arch_text, font=('Courier', 9))
        arch_label.pack(pady=(0, 10))
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Drawing canvas
        left_frame = ttk.LabelFrame(content_frame, text="Draw a digit (0-9)", padding="5")
        left_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.canvas = tk.Canvas(left_frame, 
                               width=self.canvas_size, 
                               height=self.canvas_size,
                               bg='black',
                               cursor='circle')
        self.canvas.pack()
        
        # Clear button
        clear_btn = ttk.Button(left_frame, text="🗑️ Clear", command=self.clear_canvas)
        clear_btn.pack(pady=10)
        
        # Right: Prediction
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Prediction display
        pred_frame = ttk.LabelFrame(right_frame, text="CNN Prediction", padding="10")
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prediction_label = ttk.Label(pred_frame, text="?", style='Big.TLabel')
        self.prediction_label.pack()
        
        self.confidence_label = ttk.Label(pred_frame, text="Draw something!", style='Info.TLabel')
        self.confidence_label.pack()
        
        # Probability bars
        prob_frame = ttk.LabelFrame(right_frame, text="Class Probabilities", padding="5")
        prob_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prob_bars = []
        self.prob_labels = []
        
        for i in range(10):
            row_frame = ttk.Frame(prob_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            label = ttk.Label(row_frame, text=f"{i}:", width=2)
            label.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(row_frame, length=150, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            prob_label = ttk.Label(row_frame, text="0.0%", width=8)
            prob_label.pack(side=tk.LEFT)
            
            self.prob_bars.append(bar)
            self.prob_labels.append(prob_label)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(right_frame, text="What CNN sees (28×28)", padding="5")
        preview_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.preview_canvas = tk.Canvas(preview_frame, width=112, height=112, bg='black')
        self.preview_canvas.pack()
        
        # Info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        info_text = ("💡 Level 4 Improvements:\n"
                    "• Convolutional layers detect local patterns\n"
                    "• Max pooling adds translation invariance\n"
                    "• BatchNorm accelerates training\n"
                    "• Dropout prevents overfitting")
        info_label = ttk.Label(info_frame, text=info_text, font=('Helvetica', 9))
        info_label.pack()
    
    def paint(self, event):
        """Paint on canvas when mouse is dragged"""
        x, y = event.x, event.y
        r = self.brush_size
        
        # Draw on canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        
        # Draw on PIL image
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def on_release(self, event):
        """Predict when mouse is released"""
        self.predict()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.prediction_label.config(text="?")
        self.confidence_label.config(text="Draw something!")
        
        for bar, label in zip(self.prob_bars, self.prob_labels):
            bar['value'] = 0
            label.config(text="0.0%")
        
        self.preview_canvas.delete('all')
    
    def preprocess_image(self):
        """
        Preprocess drawn image to 28x28 for CNN.
        
        Uses MNIST-style centering for better accuracy:
        - Finds bounding box of the digit
        - Crops and resizes to fit in 20x20 box
        - Centers in 28x28 image
        """
        img_array = np.array(self.image)
        
        # Find bounding box of non-zero pixels
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        
        if not rows.any() or not cols.any():
            # Empty image
            return np.zeros((28, 28), dtype=np.float32)
        
        # Get bounding box
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop to bounding box
        cropped = self.image.crop((cmin, rmin, cmax + 1, rmax + 1))
        
        # Calculate size to fit in 20x20 box while preserving aspect ratio
        w, h = cropped.size
        if w > h:
            new_w = 20
            new_h = max(1, int(20 * h / w))
        else:
            new_h = 20
            new_w = max(1, int(20 * w / h))
        
        # Resize to fit in 20x20
        cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create 28x28 image and paste centered
        centered = Image.new('L', (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        centered.paste(cropped, (paste_x, paste_y))
        
        # Apply slight blur for anti-aliasing (matches MNIST style)
        centered = centered.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert to numpy and normalize
        img_array = np.array(centered, dtype=np.float32) / 255.0
        
        return img_array
    
    def update_preview(self, img_array):
        """Update the preview canvas with what the CNN sees"""
        # Scale up for display
        img_display = Image.fromarray((img_array * 255).astype(np.uint8))
        img_display = img_display.resize((112, 112), Image.Resampling.NEAREST)
        
        self.preview_photo = ImageTk.PhotoImage(img_display)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(56, 56, image=self.preview_photo)
    
    def predict(self):
        """Make prediction with CNN"""
        # Preprocess
        img_array = self.preprocess_image()
        
        # Update preview
        self.update_preview(img_array)
        
        # Check if empty
        if img_array.sum() < 0.1:
            return
        
        # Convert to tensor
        input_tensor = torch.tensor(img_array, dtype=torch.float32).view(1, 1, 28, 28)
        input_tensor = input_tensor.to(self.device)
        
        # Get probabilities
        probs = self.model.get_probabilities(input_tensor)
        probs = probs.cpu().numpy()[0]
        
        # Get prediction
        prediction = probs.argmax()
        confidence = probs[prediction] * 100
        
        # Update display
        self.prediction_label.config(text=str(prediction))
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Update probability bars
        for i, (bar, label) in enumerate(zip(self.prob_bars, self.prob_labels)):
            prob = probs[i] * 100
            bar['value'] = prob
            label.config(text=f"{prob:.1f}%")


def main():
    """Main entry point"""
    print("=" * 60)
    print("  Level 4: Interactive CNN Digit Recognition")
    print("=" * 60)
    
    root = tk.Tk()
    app = CNNDigitRecognizer(root)
    
    print("\n🖌️  Draw digits with your mouse!")
    print("   Press Clear to start over.\n")
    
    root.mainloop()


if __name__ == "__main__":
    main()

