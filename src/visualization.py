import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

class PerformanceVisualizer:
    def __init__(self):
        self.output_dir = '../outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_class_distribution(self, df, save=True):
        plt.figure(figsize=(10, 6))
        
        counts = df['final_result'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        
        plt.subplot(1, 2, 1)
        counts.plot(kind='bar', color=colors, edgecolor='black')
        plt.title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        plt.xlabel('Result', fontsize=12)
        plt.ylabel('Number of Students', fontsize=12)
        plt.xticks(rotation=0)
        
        for i, v in enumerate(counts):
            plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/class_distribution.png")
        
        plt.show()
        
    def plot_feature_distributions(self, df, save=True):
        feature_columns = ['study_hours', 'attendance', 'internal_marks', 
                          'assignment_marks', 'previous_marks', 'participation', 
                          'project_marks']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(feature_columns):
            axes[idx].hist(df[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(feature_columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/feature_distributions.png")
        
        plt.show()
        
    def plot_correlation_matrix(self, df, save=True):
        feature_columns = ['study_hours', 'attendance', 'internal_marks', 
                          'assignment_marks', 'previous_marks', 'participation', 
                          'project_marks', 'final_result_encoded']
        
        plt.figure(figsize=(12, 10))
        
        correlation = df[feature_columns].corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/correlation_matrix.png")
        
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fail', 'Pass'], 
                   yticklabels=['Fail', 'Pass'],
                   cbar_kws={"shrink": 0.8})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
        
        plt.show()
        
    def plot_model_comparison(self, results_dict, save=True):
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        data = {metric: [results_dict[model][metric]*100 for model in models] 
                for metric in metrics}
        
        x = np.arange(len(models))
        width = 0.2
        
        plt.figure(figsize=(14, 8))
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            plt.bar(x + i*width, data[metric], width, label=metric.replace('_', ' ').title(), 
                   color=color, edgecolor='black')
        
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xticks(x + width*1.5, models, rotation=15, ha='right')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 105)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/model_comparison.png")
        
        plt.show()
        
    def plot_feature_importance(self, model, feature_names, model_name, save=True):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            
            plt.bar(range(len(importances)), importances[indices], 
                   color='teal', edgecolor='black', alpha=0.7)
            
            plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance', fontsize=12)
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save:
                filename = f'{self.output_dir}/feature_importance_{model_name.replace(" ", "_").lower()}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {filename}")
            
            plt.show()
        else:
            print(f"⚠️ {model_name} does not support feature importance")
            
    def plot_pass_fail_by_features(self, df, save=True):
        feature_columns = ['study_hours', 'attendance', 'internal_marks', 
                          'assignment_marks', 'previous_marks', 'participation', 
                          'project_marks']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(feature_columns):
            df.boxplot(column=col, by='final_result', ax=axes[idx])
            axes[idx].set_title(f'{col.replace("_", " ").title()}')
            axes[idx].set_xlabel('Result')
            axes[idx].set_ylabel('Value')
            plt.sca(axes[idx])
            plt.xticks([1, 2], ['Fail', 'Pass'])
        
        for idx in range(len(feature_columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Distribution by Pass/Fail', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/pass_fail_by_features.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/pass_fail_by_features.png")
        
        plt.show()

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor('../data/student_performance.csv')
    preprocessor.load_data()
    preprocessor.encode_labels()
    
    visualizer = PerformanceVisualizer()
    
    print("\nGenerating visualizations...")
    visualizer.plot_class_distribution(preprocessor.df)
    visualizer.plot_feature_distributions(preprocessor.df)
    visualizer.plot_correlation_matrix(preprocessor.df)
    visualizer.plot_pass_fail_by_features(preprocessor.df)
    
    print("\n✅ All visualizations generated successfully!")
