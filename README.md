# 🌸 Iris Flower Prediction App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A professional, end-to-end Machine Learning web application tailored for the **Iris Flower Prediction** project. This app allows users to input flower measurements and instantly predicts the species using a trained **Random Forest Classifier**.

---

##  Project Structure
```text
Iris-Flower-Prediction/
├── app.py               #  Main Streamlit application
├── train_model.py       #  ML pipeline (Training & Saving Model)
├── requirements.txt     #  Project dependencies
├── iris_model.joblib    #  Trained Model File
├── feature_names.joblib #  Feature Metadata
├── target_names.joblib  #  Target Class Metadata
└── README.md            #  Documentation
```

##  Features
- ** Home Page**: Overview of the Iris dataset and project goals.
- ** Prediction Page**: Real-time prediction with interactive sliders.
  - **Confidence Scores**: Visual bar chart showing probability for each species.
  - **History Logging**: Tracks your recent predictions in the session.
- ** Visualizations**:
  - Interactive Exploratory Data Analysis (EDA).
  - Feature distribution plots & Correlation heatmaps.
  - Feature Importance analysis from the Random Forest model.
- **About**: Technical details and developer info.

##  Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: Scikit-Learn (Random Forest)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

##  How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/urvashivankar/Iris-Flower-Prediction.git
cd Iris-Flower-Prediction
```

### 2. Install Dependencies
Ensure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the Model
If you want to regenerate the model file:
```bash
python train_model.py
```

### 4. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Deployment
This app is ready for deployment on **Streamlit Cloud**:
1. Push this code to a GitHub repository.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository.
4. Choose `app.py` as the main file and click **Deploy**.


