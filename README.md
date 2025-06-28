# 🧠 Brain Tumor Detection using CNN

This project implements a Convolutional Neural Network (CNN) to detect the presence of brain tumors from MRI images. The goal is to build an assistive tool for early diagnosis of tumors using deep learning and medical imaging.

---

## 📂 Dataset

- **Source**: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Structure**:
  - `yes/`: MRI images **with tumor**
  - `no/`: MRI images **without tumor**

---

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib
- scikit-learn

---

## 🚀 Project Pipeline

1. **Data Preprocessing**
   - Resize all images to 150x150
   - Normalize pixel values between 0 and 1
   - Label encoding: `0 = no tumor`, `1 = tumor`
   - Train/Test split (80% / 20%)

2. **Model Building**
   - CNN architecture with Conv2D, MaxPooling, Flatten, Dropout, and Dense layers
   - Binary classification using `sigmoid` activation
   - Loss function: `binary_crossentropy`
   - Optimizer: `adam`

3. **Model Training**
   - Fit model on training data with validation split
   - Monitor accuracy and loss over epochs

4. **Evaluation**
   - Accuracy score on test set
   - Confusion matrix, precision, recall, F1-score
   - Visualization of sample predictions

---

## 📈 Results

- Achieved >90% accuracy on test set
- Good generalization on unseen brain MRI images
- Simple and effective model for binary tumor classification

---

## 💡 Future Improvements

- Apply **data augmentation** to enrich training set
- Use **transfer learning** (e.g., VGG16, ResNet)

---

## 🧪 Demo

To test the model on your own MRI images:

```python
# Load and preprocess your image
img = load_img('your_image.jpg', target_size=(150,150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print("Tumor detected" if prediction[0][0] > 0.5 else "No tumor detected")
