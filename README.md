# ClarityX-Ophthalmologist-at-Home

# EfficientNetB0 for Retina Fundus Classification

## Overview
This project utilizes **EfficientNetB0**, a pre-trained convolutional neural network, for **retina fundus image classification**. The model is fine-tuned on a medical dataset to improve accuracy in detecting retinal diseases.

## Why EfficientNetB0?
EfficientNetB0 is a lightweight and highly efficient deep learning model trained on **ImageNet**. It provides a good balance of **accuracy and computational efficiency**, making it suitable for medical image analysis.

### Key Features:
- **Pre-trained on ImageNet**: Leverages transfer learning for faster training.
- **Optimized for accuracy & efficiency**: Uses compound scaling to balance width, depth, and resolution.
- **Lightweight architecture**: Suitable for deployment on various platforms.

## Model Architecture
The model consists of:
1. **Input Layer**: Takes in **256x256x3** RGB images.
2. **EfficientNetB0 Backbone**: Pre-trained on ImageNet with `include_top=False`.
3. **Batch Normalization**: Stabilizes training by normalizing feature maps.
4. **MaxPooling & Global Average Pooling**: Reduces dimensionality and extracts global features.
5. **Fully Connected Layer (512 neurons, ReLU activation)**: Learns high-level patterns.
6. **Dropout (0.4)**: Prevents overfitting by randomly deactivating neurons.
7. **Output Layer (Softmax Activation)**: Classifies images into `len(class_names)` categories.

## Training Strategy
- **Fine-tuning**: The base EfficientNetB0 model is made **trainable**, allowing it to learn medical-specific features.
- **Transfer Learning**: Initially, only custom layers are trained, then the entire model is fine-tuned.
- **Optimization**: Uses an appropriate optimizer (e.g., Adam) with a learning rate scheduler.

## Why Use Softmax in the Output Layer?
The softmax activation function is used for **multi-class classification**, ensuring that the output probabilities sum to 1, making interpretation easier.

## Dataset: ImageNet & Retina Fundus Data
- **ImageNet**: Pre-training dataset with **14M+ images across 1,000 categories**.
- **Retina Fundus Dataset**: Domain-specific images for medical classification.

## Running the Model
1. Install dependencies:
   ```sh
   pip install tensorflow keras numpy matplotlib
   ```
2. Train the model:
   ```python
   model.fit(train_dataset, epochs=20, validation_data=val_dataset)
   ```
3. Evaluate:
   ```python
   model.evaluate(test_dataset)
   ```

## Future Improvements
- Experiment with **EfficientNetB1-B7** for better accuracy.
- Implement **data augmentation** to improve generalization.
- Optimize for **edge devices** (e.g., TensorFlow Lite, ONNX).

## Conclusion
EfficientNetB0 provides a **robust foundation** for retina fundus classification, leveraging **transfer learning and fine-tuning** to achieve high performance in medical imaging tasks.



# **Dataset:**
https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets?select=G1020


https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification


https://www.kaggle.com/datasets/mariaherrerot/aptos2019

# **Research Papers:**
https://hal.science/hal-03974553/document


https://ebrary.net/202815/engineering/deep_learning_approach_predict_grade_glaucoma_fundus_images_through_constitutional_neural_networks





