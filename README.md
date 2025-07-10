# 🫀 Heart Disease ML Pipeline

A comprehensive end-to-end machine learning pipeline for heart disease prediction using the **UCI Heart Disease dataset**. This project covers everything from data preprocessing to deployment, with a focus on building interpretable and accurate models, and providing real-time predictions through a Streamlit web interface.

---

## 📌 Project Overview

This project aims to:

- Analyze and preprocess medical data related to heart disease  
- Select the most relevant features and apply dimensionality reduction  
- Build, evaluate, and optimize multiple machine learning models  
- Perform clustering to explore hidden patterns  
- Deploy an interactive UI using Streamlit  
- Provide public access to the model via **Ngrok**  

---

## 📁 Project Structure

```
├── app.py # Streamlit web app
├── Heart Disease.ipynb # Full training notebook (EDA → model)
├── RandomForest_pipeline.pkl # Saved pipeline model
├── README.md # This file                
```

---

## 🧪 Features & Techniques

### ✅ Data Preprocessing
- Missing value handling  
- One-hot encoding for categorical variables  
- Feature scaling (StandardScaler, MinMaxScaler)  
- Exploratory Data Analysis (EDA)

### ✅ Dimensionality Reduction
- PCA with explained variance visualization

### ✅ Feature Selection
- Chi-Square Test  
- Recursive Feature Elimination (RFE)  
- Feature importance via Random Forest

### ✅ Supervised Learning Models
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)

### ✅ Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix, ROC Curve, AUC Score

### ✅ Unsupervised Learning
- K-Means Clustering (with Elbow Method)  
- Hierarchical Clustering (with Dendrogram)

### ✅ Model Optimization
- GridSearchCV  
- RandomizedSearchCV

### ✅ Deployment
- Real-time predictions via **Streamlit**  
- Public web access using **Ngrok**

---

## 🚀 How to Run the Project

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

### 🌐 Deploy Using Ngrok

```bash
ngrok http 8501
```

Copy the generated public URL and share it to access the app online.

---

## 📊 Dataset

- **Name:** Heart Disease UCI Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## 📂 Output & Deliverables

- Cleaned & preprocessed dataset  
- PCA-transformed features and visualization  
- Top selected features  
- Trained classification and clustering models  
- Evaluation reports and metrics  
- Optimized model with hyperparameters  
- Saved model in `.pkl` format  
- Interactive Streamlit application  
- GitHub repository with full source code and documentation  

---

## 🧠 Future Enhancements

- Add deep learning models (e.g., using Keras or TensorFlow)  
- Integrate CI/CD for automated deployment  
- Build Docker containers for scalable deployment  
- Extend app for multi-disease predictions

---

## 📬 Contact

For questions or collaborations, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/ziad-attia-4b1843241/).

---

> Built with ❤️ by Ziad Attia
