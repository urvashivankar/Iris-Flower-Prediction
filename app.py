import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Intelligence", page_icon="🌸", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('iris_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    target_names = joblib.load('target_names.joblib')
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return model, feature_names, target_names, df

try:
    model, feature_names, target_names, df = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}. Please run 'train_model.py' first.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://www.tensorflow.org/images/iris_three_species.jpg", use_container_width=True)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Home", "📊 Prediction", "📈 Visualizations", "ℹ️ About"])
    st.divider()
    st.info("Built with ❤️ using Streamlit & Scikit-Learn")

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("🌸 Iris Flower Prediction Project")
    st.markdown("""
    ### Project Overview
    Welcome to the **Iris Intelligence** platform. This application uses Machine Learning to predict the species of an Iris flower based on its physical measurements.
    
    #### The Dataset
    The [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is one of the most famous datasets in statistics and machine learning. It contains 150 samples from three species of Iris:
    - **Setosa**
    - **Versicolor**
    - **Virginica**
    
    #### ML Workflow
    1. **Data Collection**: Iris Fisher dataset.
    2. **Preprocessing**: Feature scaling and cleaning.
    3. **Training**: Random Forest Classifier (100 estimators).
    4. **Evaluation**: 100% Accuracy on test split.
    5. **Deployment**: Real-time Streamlit Web Interface.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        st.metric("Target Classes", len(target_names))

# --- PREDICTION PAGE ---
elif page == "📊 Prediction":
    st.title("📊 Flower Species Prediction")
    st.write("Adjust the sliders below to input flower characteristics.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Features")
        s_len = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        s_wid = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
        p_len = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        p_wid = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)
        
        predict_btn = st.button("Predict Species")
    
    with col2:
        st.subheader("Result")
        if predict_btn:
            input_data = np.array([[s_len, s_wid, p_len, p_wid]])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            species = target_names[prediction[0]]
            st.success(f"### Predicted Species: **{species}**")
            
            # Show probabilities
            st.write("#### Confidence Scores:")
            prob_df = pd.DataFrame(probability, columns=target_names)
            st.bar_chart(prob_df.T)
            
            # --- LOGGING (Bonus) ---
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Sepal Length": s_len, "Sepal Width": s_wid,
                "Petal Length": p_len, "Petal Width": p_wid,
                "Prediction": species
            })
        else:
            st.info("Enter values and click 'Predict' to see the result.")

    # Show History
    if 'history' in st.session_state and len(st.session_state.history) > 0:
        st.divider()
        st.subheader("📜 Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Download as CSV (Bonus)
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# --- VISUALIZATIONS PAGE ---
elif page == "📈 Visualizations":
    st.title("📈 Data Exploration & Insights")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Feature Distribution", "Pair Plot"])
    
    with tab1:
        st.subheader("Iris Dataset Preview")
        st.dataframe(df, use_container_width=True)
        st.subheader("Statistical Summary")
        st.write(df.describe())
        
    with tab2:
        st.subheader("Distribution of Features by Species")
        feature = st.selectbox("Select Feature to Visualize", feature_names)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df, x=feature, hue="species", fill=True, ax=ax)
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Pair Plot Interaction")
        st.write("This plot shows relationships between all pairs of features.")
        # Seaborn pairplot is heavy, so we use a cache-friendly approach if possible or just render
        fig = sns.pairplot(df.drop('target', axis=1), hue="species", palette="viridis")
        st.pyplot(fig)
        
        # Feature Importance (Bonus)
        st.divider()
        st.subheader("Model Insights: Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax_imp, palette="magma")
        st.pyplot(fig_imp)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    
    st.markdown("""
    ### Technology Stack
    - **Frontend**: [Streamlit](https://streamlit.io)
    - **ML Framework**: [Scikit-Learn](https://scikit-learn.org)
    - **Data Handling**: [Pandas](https://pandas.pydata.org) & [NumPy](https://numpy.org)
    - **Visualization**: [Matplotlib](https://matplotlib.org) & [Seaborn](https://seaborn.pydata.org)
    - **Model Storage**: [Joblib](https://joblib.readthedocs.io)
    
    ### Model Details
    The prediction engine uses a **Random Forest Classifier** with 100 decision trees. It was trained on the standard Iris dataset and achieved high accuracy due to the clear separation of species features.
    
    ### How to Deploy
    1. **Streamlit Cloud**: Link your GitHub repo and select `app.py`.
    2. **Render / Hugging Face**: Use the provided `requirements.txt` and run `streamlit run app.py`.
    
    ### About the Developer
    This project was built as a portfolio-ready demonstration of an end-to-end ML application.
    """)
    st.balloons()
