import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prediction import StudentPerformancePredictor
from src.visualization import PerformanceVisualizer

def main():
    print("="*80)
    print(" "*20 + "STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*80)
    
    data_path = 'data/student_performance.csv'
    
    print("\n[STEP 1] DATA PREPROCESSING")
    print("-"*80)
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    print("\n[STEP 2] DATA VISUALIZATION")
    print("-"*80)
    visualizer = PerformanceVisualizer()
    print("\nGenerating visualizations...")
    visualizer.plot_class_distribution(preprocessor.df)
    visualizer.plot_correlation_matrix(preprocessor.df)
    
    print("\n[STEP 3] MODEL TRAINING")
    print("-"*80)
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    print("\n[STEP 4] SAVING BEST MODEL")
    print("-"*80)
    trainer.save_model('models/best_model.pkl')
    
    print("\n[STEP 5] GENERATING MODEL VISUALIZATIONS")
    print("-"*80)
    visualizer.plot_model_comparison(trainer.results)
    
    for model_name in trainer.results.keys():
        y_pred = trainer.results[model_name]['predictions']
        visualizer.plot_confusion_matrix(y_test, y_pred, model_name)
    
    if hasattr(trainer.best_model, 'feature_importances_'):
        feature_names = ['study_hours', 'attendance', 'internal_marks', 
                        'assignment_marks', 'previous_marks', 'participation', 
                        'project_marks']
        visualizer.plot_feature_importance(trainer.best_model, feature_names, 
                                          trainer.best_model_name)
    
    print("\n[STEP 6] MAKING PREDICTIONS")
    print("-"*80)
    predictor = StudentPerformancePredictor('models/best_model.pkl')
    
    print("\n--- Example Prediction 1: Strong Student ---")
    predictor.predict_with_details(
        study_hours=7, 
        attendance=92, 
        internal_marks=88, 
        assignment_marks=85, 
        previous_marks=86, 
        participation=9, 
        project_marks=89
    )
    
    print("\n--- Example Prediction 2: Weak Student ---")
    predictor.predict_with_details(
        study_hours=2, 
        attendance=58, 
        internal_marks=44, 
        assignment_marks=40, 
        previous_marks=48, 
        participation=3, 
        project_marks=42
    )
    
    print("\n[STEP 7] IDENTIFYING WEAK STUDENTS")
    print("-"*80)
    weak_students = predictor.identify_weak_students(data_path)
    
    print("\n" + "="*80)
    print(" "*25 + "PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 All results saved in 'outputs/' directory")
    print("💾 Best model saved in 'models/' directory")
    print(f"🏆 Best Model: {trainer.best_model_name}")
    print(f"✅ Accuracy: {trainer.best_accuracy*100:.2f}%")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
