import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        
    def initialize_models(self):
        print("Initializing machine learning models...")
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        print(f"Initialized {len(self.models)} models: {', '.join(self.models.keys())}")
        
    def train_model(self, model_name, model, X_train, y_train):
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"{model_name} training completed!")
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        print(f"\nEvaluating {model_name}...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        print(f"\nConfusion Matrix:\n{cm}")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            self.best_model_name = model_name
        
        return self.results[model_name]
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        self.initialize_models()
        
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            self.train_model(model_name, model, X_train, y_train)
            self.evaluate_model(model_name, model, X_test, y_test)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        self.display_comparison()
        
    def display_comparison(self):
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy']*100 for m in self.results.keys()],
            'Precision': [self.results[m]['precision']*100 for m in self.results.keys()],
            'Recall': [self.results[m]['recall']*100 for m in self.results.keys()],
            'F1-Score': [self.results[m]['f1_score']*100 for m in self.results.keys()]
        })
        
        comparison_df = comparison_df.round(2)
        print(comparison_df.to_string(index=False))
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name} with {self.best_accuracy*100:.2f}% accuracy")
        
        return comparison_df
    
    def save_model(self, model_path='../models/best_model.pkl'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'accuracy': self.best_accuracy
            }, f)
        
        print(f"\n✅ Best model ({self.best_model_name}) saved to {model_path}")
        
    def load_model(self, model_path='../models/best_model.pkl'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.best_model = data['model']
        self.best_model_name = data['model_name']
        self.best_accuracy = data['accuracy']
        
        print(f"✅ Model loaded: {self.best_model_name} (Accuracy: {self.best_accuracy*100:.2f}%)")
        return self.best_model

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor('../data/student_performance.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, X_test, y_train, y_test)
    trainer.save_model()
