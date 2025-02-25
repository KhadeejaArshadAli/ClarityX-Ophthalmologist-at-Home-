# ClarityX-Ophthalmologist-at-Home
# Image Clustering and Classification using KMeans and EfficientNetB0
![Model](modelfinal.png)

## Overview
This project involves clustering image data using **KMeans**, training a deep learning model with **EfficientNetB0**, and deploying the model through a **Flask** web application.

## Project Workflow
1. **Image Clustering with KMeans**
   - Used **KMeans clustering** to group similar images.
   - Applied **Elbow Method** and **Silhouette Analysis** to determine the optimal number of clusters.
   - Evaluated the clustering performance using **silhouette scores**.

2. **Training with EfficientNetB0**
   - Used a **pre-trained EfficientNetB0** model for feature extraction and classification.
   - Performed **data preprocessing**, **augmentation**, and **fine-tuning** for improved accuracy.
   - Validated and tested the model to ensure robust performance.

3. **Flask Application for Deployment**
   - Created a **Flask-based web app** to serve the trained model.
   - Allows users to upload images and get predictions in real-time.
   - Ensured smooth integration with front-end UI.

## Technologies Used
- **Machine Learning & Deep Learning**: `KMeans`, `EfficientNetB0`
- **Libraries**: `scikit-learn`, `TensorFlow`, `Keras`
- **Data Processing**: `NumPy`, `Pandas`, `Matplotlib`
- **Web Framework**: `Flask`

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/` to access the application.

## Results
- Achieved **optimal clustering** for dataset categorization.
- Trained **EfficientNetB0** with high accuracy.
- Successfully deployed a **Flask-based web interface** for image classification.

## Future Enhancements
- Improve clustering techniques with **advanced feature extraction**.
- Optimize the Flask application with **faster inference time**.
- Add a **database** to store user queries and predictions.

## Contributors
- **Khadeeja Arshad Ali -B20102057**
- **Nabiha Faisal -B20102130**
- **Saad Shariq Siddique -B20102141**
- **M.Noor Sheikh -B20102107**

## License
This project is licensed under the **MIT License**.




# **Dataset:**
https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets?select=G1020


https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification


https://www.kaggle.com/datasets/mariaherrerot/aptos2019

# **Research Papers:**
https://hal.science/hal-03974553/document


https://ebrary.net/202815/engineering/deep_learning_approach_predict_grade_glaucoma_fundus_images_through_constitutional_neural_networks





