# MNIST Digit Classifier — Multilayer Perceptron (MLP)

A PyTorch implementation of a Multilayer Perceptron (MLP) trained on the full MNIST dataset to classify handwritten digits (0–9).

---

## What This Program Does

1. Downloads and loads the MNIST dataset (70,000 handwritten digit images)
2. Builds a neural network (MLP) with two hidden layers
3. Trains the model using Cross-Entropy Loss and the Adam optimizer
4. Evaluates accuracy on the test set
5. Plots a Confusion Matrix to visualize prediction performance

---

## Dataset — MNIST

| Property        | Details                          |
|-----------------|----------------------------------|
| Full Form       | Modified National Institute of Standards and Technology |
| Training Images | 60,000                           |
| Test Images     | 10,000                           |
| Image Size      | 28 × 28 pixels (grayscale)       |
| Classes         | 10 (digits 0 through 9)          |

---

## Model Architecture

```
Input Image: 28 × 28 pixels
        ↓
   Flatten → 784 values
        ↓
 Linear Layer (784 → 256)
        ↓
      ReLU
        ↓
 Linear Layer (256 → 128)
        ↓
      ReLU
        ↓
 Linear Layer (128 → 10)
        ↓
 Output: 10 scores (one per digit)
```

| Layer           | Input Size | Output Size | Activation |
|-----------------|------------|-------------|------------|
| Flatten         | 28 × 28    | 784         | —          |
| Hidden Layer 1  | 784        | 256         | ReLU       |
| Hidden Layer 2  | 256        | 128         | ReLU       |
| Output Layer    | 128        | 10          | —          |

> The output layer produces **logits** (raw scores). The highest score index is the predicted digit.

---

## Requirements

```
torch
torchvision
scikit-learn
seaborn
matplotlib
```

Install with:
```bash
pip install torch torchvision scikit-learn seaborn matplotlib
```

---

## How to Run

```bash
python mnist_mlp.py
```

The MNIST dataset will be **automatically downloaded** into a `./data` folder on first run.

---

## Code Walkthrough

### 1. Device Setup
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Uses GPU (CUDA) if available, otherwise falls back to CPU. GPU training is significantly faster.

---

### 2. Data Loading
```python
train_loader = DataLoader(MNIST(..., train=True), batch_size=64, shuffle=True)
test_loader  = DataLoader(MNIST(..., train=False), batch_size=64)
```
- `transforms.ToTensor()` converts PIL images to PyTorch tensors and scales pixel values from [0, 255] to [0.0, 1.0]
- `DataLoader` feeds data in **batches of 64** instead of loading all images at once
- `shuffle=True` randomizes training order each epoch to prevent the model from memorizing sequence

---

### 3. Model Definition
```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
).to(device)
```
- `nn.Sequential` stacks layers in order
- `nn.Flatten()` converts the 2D image (28×28) into a 1D vector of 784 values
- `nn.Linear(in, out)` is a fully connected layer — every input connects to every output
- `nn.ReLU()` applies `f(x) = max(0, x)` — adds non-linearity so the network can learn complex patterns

---

### 4. Loss Function & Optimizer
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- **CrossEntropyLoss** measures how wrong the model's predictions are (lower = better)
- **Adam (Adaptive Moment Estimation)** updates model weights to minimize the loss
- **Learning Rate (lr=0.001)** controls the size of each weight update step

---

### 5. Training Loop
```python
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        loss = criterion(model(images), labels)
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Backpropagation
        optimizer.step()        # Update weights
```
Each batch goes through this cycle:

```
Forward Pass → Compute Loss → Zero Gradients → Backpropagation → Update Weights
```

- **Epoch**: One full pass through the entire training dataset
- **Backpropagation**: Computes how much each weight contributed to the error
- `zero_grad()` must be called each step because PyTorch accumulates gradients by default

---

### 6. Evaluation
```python
model.eval()
with torch.no_grad():
    predicted = outputs.argmax(1)
```
- `model.eval()` switches off training-specific behavior
- `torch.no_grad()` disables gradient tracking (not needed during evaluation — saves memory)
- `argmax(1)` picks the index of the highest score among the 10 output values — that index is the predicted digit

---

### 7. Confusion Matrix
```python
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```
A 10×10 grid where:
- **Rows** = Actual digit
- **Columns** = Predicted digit
- **Diagonal cells** = Correct predictions
- **Off-diagonal cells** = Mistakes (e.g., model predicted 9 but it was actually 4)

---

## Key Terms Glossary

| Term            | Full Form / Meaning                                              |
|-----------------|------------------------------------------------------------------|
| MLP             | Multilayer Perceptron                                            |
| MNIST           | Modified National Institute of Standards and Technology          |
| ReLU            | Rectified Linear Unit — activation function: `max(0, x)`        |
| Adam            | Adaptive Moment Estimation — optimizer algorithm                 |
| CUDA            | Compute Unified Device Architecture — NVIDIA's GPU platform      |
| GPU             | Graphics Processing Unit                                         |
| CPU             | Central Processing Unit                                          |
| Epoch           | One full pass through the entire training dataset                |
| Batch           | A small subset of data processed together                        |
| Loss            | A number measuring how wrong the model's predictions are         |
| Logits          | Raw output scores before applying Softmax                        |
| Softmax         | Converts raw scores to probabilities that sum to 1               |
| Backpropagation | Algorithm to compute gradients of the loss w.r.t. weights        |
| Gradient        | Direction and magnitude to adjust each weight                    |
| Learning Rate   | Controls the size of each weight update step                     |
| DataLoader      | PyTorch utility to load and batch data efficiently               |

---

## Expected Output

```
Training for 5 epochs...
Test Accuracy: ~97.5%
```

A confusion matrix heatmap is displayed showing prediction performance across all 10 digit classes.
