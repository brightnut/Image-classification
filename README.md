Skin Cancer Classification with CNN and Transfer Learning

This project classifies skin cancer images as either benign or malignant using deep learning models, specifically VGG16 and VGG19, with transfer learning for feature extraction. Ensemble classifiers, Random Forest and Decision Tree, are then applied to the extracted features to enhance prediction accuracy.

Table of Contents
Overview
Dataset
Installation
Project Structure
Model Architecture
Training and Evaluation
Results
Future Improvements
Acknowledgments
Overview
The main objective is to classify skin cancer images into two categories:

Benign (non-cancerous)
Malignant (cancerous)
This model utilizes pretrained CNN architectures (VGG16 and VGG19) with frozen layers to leverage feature extraction capabilities. Final classification is performed with Random Forest and Decision Tree models.

Dataset
Images should be organized in the following folder structure:

bash
Copy code
Skin_Cancer/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
Train and Test Data: Place images in the train and test folders. Ensure images are organized by class.
Installation
To run this project, make sure you have the following libraries installed:

bash
Copy code
pip install numpy pandas matplotlib opencv-python Pillow seaborn scikit-learn tensorflow keras
Project Structure
trainpath: Path to the training data folder.
testpath: Path to the testing data folder.
Key Files:
main.py: Contains all code for data loading, preprocessing, model training, and evaluation.
README.md: Project documentation.
Model Architecture
Data Loading and Preprocessing:

Images are loaded from directories, resized to 224x224 pixels, and labeled.
Labels are encoded as 0 (benign) and 1 (malignant).
Transfer Learning:

VGG16 and VGG19 models pretrained on ImageNet are used as feature extractors.
Feature extraction layers are frozen to retain pretrained weights.
Additional Dense layers are added to allow binary classification.
Feature-Based Classification:

Extracted features are fed to ensemble classifiers (Random Forest and Decision Tree) for final classification.
Training and Evaluation
Training: The models are trained with MSE loss and accuracy as metrics.
Validation: 20% of the training data is used as validation during training.
Evaluation: Accuracy scores for the training and test sets are calculated and printed.
Results
Random Forest and Decision Tree classifiers achieved high accuracy scores on the test data.
Training accuracy is near 1.0, and test accuracy is approximately 0.97, demonstrating effective generalization on unseen data.
Future Improvements
Experiment with fine-tuning pretrained layers in VGG16 and VGG19 for improved performance.
Explore alternative evaluation metrics (e.g., precision, recall, F1-score) to handle potential class imbalances.
Test additional classifiers or ensemble methods to optimize the prediction accuracy further.
Acknowledgments
This project utilizes pretrained models from the Keras library, specifically VGG16 and VGG19 architectures.
ImageNet dataset is used as the initial training dataset for transfer learning.
