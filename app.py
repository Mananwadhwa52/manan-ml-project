import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prediction import StudentPerformancePredictor
from src.visualization import PerformanceVisualizer
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.markdown("### Machine Learning-based Academic Performance Predictor")
st.markdown("---")

@st.cache_resource
def load_predictor():
    if os.path.exists('models/best_model.pkl'):
        return StudentPerformancePredictor('models/best_model.pkl')
    return None

def train_model():
    with st.spinner("Training models... This may take a moment."):
        preprocessor = DataPreprocessor('data/student_performance.csv')
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, X_test, y_train, y_test)
        trainer.save_model('models/best_model.pkl')
        
        return trainer, preprocessor

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🤖 Train Model", "🔮 Make Prediction", "📈 Analytics"])

with tab1:
    st.header("Dataset Overview")
    
    if os.path.exists('data/student_performance.csv'):
        df = pd.read_csv('data/student_performance.csv')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            pass_count = len(df[df['final_result'] == 'Pass'])
            st.metric("Pass Count", pass_count)
        with col3:
            fail_count = len(df[df['final_result'] == 'Fail'])
            st.metric("Fail Count", fail_count)
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['final_result'].value_counts())
        with col2:
            st.write("**Percentage Distribution:**")
            st.write(df['final_result'].value_counts(normalize=True) * 100)
    else:
        st.error("Dataset not found! Please ensure 'data/student_performance.csv' exists.")

with tab2:
    st.header("Train Machine Learning Models")
    
    st.write("""
    This module trains multiple ML algorithms and compares their performance:
    - Decision Tree Classifier
    - Random Forest Classifier
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    """)
    
    if st.button("🚀 Start Training", type="primary"):
        trainer, preprocessor = train_model()
        
        st.success(f"✅ Training completed! Best Model: **{trainer.best_model_name}** with **{trainer.best_accuracy*100:.2f}%** accuracy")
        
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame({
            'Model': list(trainer.results.keys()),
            'Accuracy (%)': [trainer.results[m]['accuracy']*100 for m in trainer.results.keys()],
            'Precision (%)': [trainer.results[m]['precision']*100 for m in trainer.results.keys()],
            'Recall (%)': [trainer.results[m]['recall']*100 for m in trainer.results.keys()],
            'F1-Score (%)': [trainer.results[m]['f1_score']*100 for m in trainer.results.keys()]
        }).round(2)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        st.bar_chart(comparison_df.set_index('Model')['Accuracy (%)'])
    
    if os.path.exists('models/best_model.pkl'):
        st.info("✅ Trained model is available and ready for predictions!")

with tab3:
    st.header("Predict Student Performance")
    
    predictor = load_predictor()
    
    if predictor is None:
        st.warning("⚠️ No trained model found. Please train the model first in the 'Train Model' tab.")
    else:
        st.success(f"Using model: **{predictor.model_name}** (Accuracy: **{predictor.accuracy*100:.2f}%**)")
        
        st.subheader("Enter Student Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            study_hours = st.slider("Study Hours (per day)", 0, 10, 5)
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            internal_marks = st.slider("Internal Marks", 0, 100, 70)
            assignment_marks = st.slider("Assignment Marks", 0, 100, 70)
        
        with col2:
            previous_marks = st.slider("Previous Marks", 0, 100, 70)
            participation = st.slider("Class Participation (1-10)", 1, 10, 7)
            project_marks = st.slider("Project Marks", 0, 100, 70)
        
        if st.button("🔮 Predict Performance", type="primary"):
            result, prediction = predictor.predict_single_student(
                study_hours, attendance, internal_marks, assignment_marks,
                previous_marks, participation, project_marks
            )
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if result == "Pass":
                st.success(f"## ✅ Prediction: **{result}**")
                st.balloons()
                st.write("The student is predicted to **PASS**. Keep up the good work!")
            else:
                st.error(f"## ❌ Prediction: **{result}**")
                st.write("The student is predicted to **FAIL**. Immediate intervention required!")
                
                st.subheader("📋 Recommendations")
                recommendations = []
                if study_hours < 4:
                    recommendations.append("• Increase daily study hours (currently low)")
                if attendance < 75:
                    recommendations.append("• Improve attendance (currently below 75%)")
                if internal_marks < 60:
                    recommendations.append("• Focus on improving internal exam performance")
                if assignment_marks < 60:
                    recommendations.append("• Complete assignments with better quality")
                if participation < 6:
                    recommendations.append("• Increase class participation")
                
                for rec in recommendations:
                    st.warning(rec)

with tab4:
    st.header("Analytics & Insights")
    
    if os.path.exists('data/student_performance.csv'):
        df = pd.read_csv('data/student_performance.csv')
        
        st.subheader("Feature Correlation Analysis")
        
        from sklearn.preprocessing import LabelEncoder
        df_encoded = df.copy()
        le = LabelEncoder()
        df_encoded['final_result_encoded'] = le.fit_transform(df['final_result'])
        
        feature_columns = ['study_hours', 'attendance', 'internal_marks', 
                          'assignment_marks', 'previous_marks', 'participation', 
                          'project_marks', 'final_result_encoded']
        
        correlation = df_encoded[feature_columns].corr()
        
        st.write("**Correlation with Final Result:**")
        result_corr = correlation['final_result_encoded'].sort_values(ascending=False)
        st.bar_chart(result_corr[:-1])
        
        st.subheader("Pass vs Fail Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pass Students - Average Scores**")
            pass_students = df[df['final_result'] == 'Pass']
            st.dataframe(pass_students[['study_hours', 'attendance', 'internal_marks', 
                                       'assignment_marks']].mean().round(2))
        
        with col2:
            st.write("**Fail Students - Average Scores**")
            fail_students = df[df['final_result'] == 'Fail']
            st.dataframe(fail_students[['study_hours', 'attendance', 'internal_marks', 
                                       'assignment_marks']].mean().round(2))
        
        predictor = load_predictor()
        if predictor:
            st.subheader("Weak Students Identification")
            
            if st.button("🔍 Identify Weak Students"):
                weak_students = predictor.identify_weak_students('data/student_performance.csv')
                
                if len(weak_students) > 0:
                    st.warning(f"Found **{len(weak_students)}** students predicted to fail")
                    st.dataframe(weak_students[['study_hours', 'attendance', 'internal_marks', 
                                               'assignment_marks', 'predicted_label']], 
                               use_container_width=True)
                else:
                    st.success("No weak students identified!")

st.sidebar.title("About")
st.sidebar.info("""
**Student Performance Prediction System**

This application uses Machine Learning to predict student academic performance based on various factors:

- Study Hours
- Attendance
- Internal Marks
- Assignment Marks
- Previous Marks
- Class Participation
- Project Marks

**Algorithms Used:**
- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors

**Developed for:** Minor Project
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Start:**")
st.sidebar.markdown("1. View dataset in 'Dataset Overview'")
st.sidebar.markdown("2. Train models in 'Train Model'")
st.sidebar.markdown("3. Make predictions in 'Make Prediction'")
st.sidebar.markdown("4. Analyze results in 'Analytics'")
