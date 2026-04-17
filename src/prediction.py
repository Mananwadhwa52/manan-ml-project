import pandas as pd
import numpy as np
import pickle

class StudentPerformancePredictor:
    def __init__(self, model_path='../models/best_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.model_name = None
        self.accuracy = None
        self.load_model()
        
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.model_name = data['model_name']
            self.accuracy = data['accuracy']
            
            print(f"✅ Model loaded successfully!")
            print(f"Model: {self.model_name}")
            print(f"Accuracy: {self.accuracy*100:.2f}%")
        except FileNotFoundError:
            print(f"❌ Model file not found at {self.model_path}")
            print("Please train the model first using model_training.py")
    
    def predict_single_student(self, study_hours, attendance, internal_marks, 
                              assignment_marks, previous_marks, participation, 
                              project_marks):
        features = np.array([[study_hours, attendance, internal_marks, 
                            assignment_marks, previous_marks, participation, 
                            project_marks]])
        
        prediction = self.model.predict(features)[0]
        
        result = "Pass" if prediction == 1 else "Fail"
        
        return result, prediction
    
    def predict_batch(self, data_df):
        feature_columns = ['study_hours', 'attendance', 'internal_marks', 
                          'assignment_marks', 'previous_marks', 'participation', 
                          'project_marks']
        
        X = data_df[feature_columns]
        predictions = self.model.predict(X)
        
        data_df['predicted_result'] = predictions
        data_df['predicted_label'] = data_df['predicted_result'].apply(
            lambda x: 'Pass' if x == 1 else 'Fail'
        )
        
        return data_df
    
    def predict_with_details(self, study_hours, attendance, internal_marks, 
                            assignment_marks, previous_marks, participation, 
                            project_marks):
        result, prediction = self.predict_single_student(
            study_hours, attendance, internal_marks, assignment_marks, 
            previous_marks, participation, project_marks
        )
        
        print("\n" + "="*60)
        print("STUDENT PERFORMANCE PREDICTION")
        print("="*60)
        print(f"\nInput Features:")
        print(f"  Study Hours: {study_hours}")
        print(f"  Attendance: {attendance}%")
        print(f"  Internal Marks: {internal_marks}")
        print(f"  Assignment Marks: {assignment_marks}")
        print(f"  Previous Marks: {previous_marks}")
        print(f"  Participation: {participation}/10")
        print(f"  Project Marks: {project_marks}")
        
        print(f"\n{'='*60}")
        print(f"PREDICTION: {result}")
        print(f"{'='*60}")
        
        if result == "Fail":
            print("\n⚠️ WARNING: Student is predicted to FAIL")
            print("\nRecommendations:")
            if study_hours < 4:
                print("  • Increase daily study hours (currently low)")
            if attendance < 75:
                print("  • Improve attendance (currently below 75%)")
            if internal_marks < 60:
                print("  • Focus on improving internal exam performance")
            if assignment_marks < 60:
                print("  • Complete assignments with better quality")
            if participation < 6:
                print("  • Increase class participation")
        else:
            print("\n✅ Student is predicted to PASS")
            print("Keep up the good work!")
        
        return result, prediction
    
    def identify_weak_students(self, data_path):
        df = pd.read_csv(data_path)
        df_with_predictions = self.predict_batch(df)
        
        weak_students = df_with_predictions[df_with_predictions['predicted_result'] == 0]
        
        print("\n" + "="*60)
        print(f"WEAK STUDENTS IDENTIFICATION")
        print("="*60)
        print(f"\nTotal Students: {len(df_with_predictions)}")
        print(f"Predicted to Pass: {len(df_with_predictions[df_with_predictions['predicted_result'] == 1])}")
        print(f"Predicted to Fail: {len(weak_students)}")
        print(f"Failure Rate: {len(weak_students)/len(df_with_predictions)*100:.2f}%")
        
        if len(weak_students) > 0:
            print(f"\n⚠️ Students predicted to fail:")
            print(weak_students[['study_hours', 'attendance', 'internal_marks', 
                                'assignment_marks', 'predicted_label']].to_string(index=True))
        
        return weak_students

if __name__ == "__main__":
    predictor = StudentPerformancePredictor()
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    print("\n--- Example 1: Good Student ---")
    predictor.predict_with_details(
        study_hours=7, 
        attendance=90, 
        internal_marks=85, 
        assignment_marks=88, 
        previous_marks=82, 
        participation=9, 
        project_marks=87
    )
    
    print("\n--- Example 2: Weak Student ---")
    predictor.predict_with_details(
        study_hours=2, 
        attendance=55, 
        internal_marks=42, 
        assignment_marks=38, 
        previous_marks=45, 
        participation=3, 
        project_marks=40
    )
    
    print("\n--- Identifying Weak Students from Dataset ---")
    predictor.identify_weak_students('../data/student_performance.csv')
