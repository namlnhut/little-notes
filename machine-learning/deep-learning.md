# Deep Learning

Deep learning uses multi-layered neural networks to learn hierarchical representations of data.

## Neural Network Architectures

### Feedforward Neural Networks
- Multilayer Perceptrons (MLP)
- Activation functions (ReLU, Sigmoid, Tanh)
- Backpropagation

### Convolutional Neural Networks (CNN)
- Image classification
- Object detection
- Image segmentation

### Recurrent Neural Networks (RNN)
- Sequence modeling
- Time series prediction
- Natural language processing
- LSTM and GRU variants

### Transformers
- Attention mechanisms
- BERT, GPT architectures
- Modern NLP applications

## Example: Simple Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Training Techniques

- Gradient descent optimization
- Batch normalization
- Dropout regularization
- Learning rate scheduling
- Data augmentation
