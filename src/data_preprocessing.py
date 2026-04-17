import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        return self.df
    
    def check_missing_values(self):
        print("\nChecking for missing values...")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(f"Missing values:\n{missing[missing > 0]}")
        return missing
    
    def handle_missing_values(self):
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            for column in self.df.columns:
                if self.df[column].dtype in ['float64', 'int64']:
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                else:
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            print("Missing values handled!")
        return self.df
    
    def encode_labels(self):
        print("\nEncoding target variable...")
        if 'final_result' in self.df.columns:
            self.df['final_result_encoded'] = self.label_encoder.fit_transform(self.df['final_result'])
            print(f"Encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        return self.df
    
    def get_statistics(self):
        print("\nDataset Statistics:")
        print(self.df.describe())
        
        if 'final_result' in self.df.columns:
            print("\nClass Distribution:")
            print(self.df['final_result'].value_counts())
            print(f"\nPercentage Distribution:")
            print(self.df['final_result'].value_counts(normalize=True) * 100)
        
        return self.df.describe()
    
    def prepare_features(self):
        print("\nPreparing features for training...")
        feature_columns = ['study_hours', 'attendance', 'internal_marks', 
                          'assignment_marks', 'previous_marks', 'participation', 
                          'project_marks']
        
        X = self.df[feature_columns]
        y = self.df['final_result_encoded']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
        X, y = self.prepare_features()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess(self):
        self.load_data()
        self.check_missing_values()
        self.handle_missing_values()
        self.encode_labels()
        self.get_statistics()
        return self.split_data()

if __name__ == "__main__":
    preprocessor = DataPreprocessor('../data/student_performance.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    print("\nData preprocessing completed successfully!")
