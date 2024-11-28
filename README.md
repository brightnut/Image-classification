Cancer Cell Classification Using Transfer Learning
Overview
This project implements a deep learning model to classify cancer cells as either benign or malignant using pre-trained VGG16 and VGG19 models with transfer learning. The model achieves high accuracy and demonstrates the potential of leveraging pre-trained models for medical image classification tasks.

Key Features
Utilizes VGG16 and VGG19 architectures with ImageNet pre-trained weights.
Applies transfer learning for efficient training on a limited dataset.
Processes high-resolution images (224x224) for optimal performance.
Achieves best test accuracy with the frozen VGG16 model: 88.8%.
Includes test accuracy for VGG19: 82.8%.
Project Structure
bash
Copy code
Cancer_Cell_Classification/
│
├── train/                   # Training dataset (benign/malignant folders)
├── test/                    # Test dataset (benign/malignant folders)
├── Cancer_Cell_Classification.ipynb  # Main notebook with code
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
Dataset
The dataset consists of images of cancer cells divided into two categories:

Benign
Malignant
Each image is resized to 224x224 pixels for compatibility with VGG models.

Model Description
VGG16 and VGG19
Both models are loaded without their top layers (include_top=False).
Pre-trained weights from ImageNet are used.
Additional custom layers for classification:
Dense layer with 64 units, ReLU activation.
Dense layer with 32 units, softmax activation.
Output layer with 2 units, sigmoid activation for binary classification.
Optimizer and Loss Function
Optimizer: Adam
Loss Function: Mean Squared Error.

Preprocess the dataset.
Train and evaluate the models.
Save the trained model.
Performance
Model	Test Accuracy
VGG16	88%
VGG19	84%
Future Improvements
Fine-tuning deeper layers of VGG16 and VGG19 for domain-specific learning.
Incorporating data augmentation to improve generalization.
Exploring other architectures like ResNet, Inception, or EfficientNet.
Contributing
Contributions are welcome! Feel free to fork this repository, make enhancements, and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Pre-trained VGG models: Keras Applications
Dataset: 50 images.
Deep learning framework: TensorFlow/Keras.
