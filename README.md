# Incremental CNN Text Classification

## Description
This project implements an incremental learning approach for text classification using Convolutional Neural Networks (CNN). The system is designed to handle dynamic class changes and concept drift in streaming text data, making it suitable for real-world applications where new categories may emerge or existing ones may become obsolete over time.

Key Features:
- Dynamic class handling (addition/removal of classes during training)
- Real-time visualization of learning metrics
- Custom tokenization for text processing
- Concept drift simulation
- Memory-efficient data streaming
- Interactive visualization of model performance

## Architecture
The system consists of several key components:
- **DynamicCNN**: A CNN architecture that can adapt its output layer for changing number of classes
- **TextDataStream**: Simulates streaming text data with concept drift
- **LearningVisualizer**: Real-time visualization of training metrics
- **Custom Tokenizer**: Efficient text preprocessing without external dependencies

## Requirements
```
torch
datasets
matplotlib
numpy
scikit-learn
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/incremental-cnn-classifier.git
cd incremental-cnn-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training:
```bash
python main.py
```

## Visualization
The system provides real-time visualization of four key metrics:
- Training Loss
- Classification Accuracy
- Current Class Distribution
- Number of Active Classes

## Project Structure
```
incremental-cnn-classifier/
├── main.py              # Main script containing all components
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Model Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Embedding Dimension**: 100
- **Vocabulary Size**: 10,000
- **Max Sequence Length**: 200
- **Learning Rate**: 0.001
- **Batch Size**: 32

## Data
The project uses the AG News dataset, which includes:
- 4 classes of news articles
- Training set: 120,000 samples
- Test set: 7,600 samples

## Features
1. **Incremental Learning**
   - Dynamic class addition/removal
   - Adaptive output layer
   - Continuous learning from streaming data

2. **Concept Drift Handling**
   - Automatic detection of class changes
   - Smooth transition between class configurations
   - Maintains performance during distribution shifts

3. **Real-time Monitoring**
   - Loss tracking
   - Accuracy visualization
   - Class distribution monitoring
   - Active class tracking

## Performance
The model adapts to changing class distributions while maintaining classification performance. Key metrics:
- Real-time accuracy tracking
- Dynamic loss visualization
- Class distribution monitoring
- Concept drift visualization

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Future Improvements
- [ ] Add support for custom datasets
- [ ] Implement more sophisticated concept drift detection
- [ ] Add model checkpointing
- [ ] Enhance visualization options
- [ ] Add cross-validation support
- [ ] Implement early stopping
- [ ] Add support for different architectures

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this code in your research, please cite:
```
@misc{incremental-cnn-classifier,
  author = {Imen Ferjani},
  title = {Incremental CNN Text Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/incremental-cnn-classifier}
}
```

## Contact
- Imen Ferjani
- Email: imene.ferjani@enit.utm.tn
- GitHub: [@imenFerjani](https://github.com/yourusername)