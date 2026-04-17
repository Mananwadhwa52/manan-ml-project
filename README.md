# Student Performance Prediction Using Machine Learning

A comprehensive machine learning project that predicts student academic performance based on various academic factors.

## 📋 Project Overview

This system uses supervised machine learning algorithms to predict whether a student will **Pass** or **Fail** based on their academic data. It helps educational institutions identify weak students early and take preventive measures.

## 🎯 Objectives

- Predict student final performance in advance
- Identify weak students early for intervention
- Analyze factors affecting academic performance
- Provide data-driven insights for better academic planning
- Improve overall academic results

## 📊 Dataset Features

The system uses the following input parameters:

| Feature | Description |
|---------|-------------|
| `study_hours` | Daily study time (hours) |
| `attendance` | Attendance percentage (0-100%) |
| `internal_marks` | Internal/sessional exam marks (0-100) |
| `assignment_marks` | Assignment scores (0-100) |
| `previous_marks` | Previous exam marks (0-100) |
| `participation` | Class participation score (1-10) |
| `project_marks` | Project evaluation marks (0-100) |

**Output:** `final_result` (Pass/Fail)

## 🤖 Machine Learning Algorithms

The project implements and compares multiple algorithms:

1. **Decision Tree Classifier** (Primary)
   - Easy to interpret
   - Good accuracy
   - Works well with small datasets

2. **Random Forest Classifier**
   - Ensemble method
   - Reduces overfitting
   - Higher accuracy

3. **Logistic Regression**
   - Simple and fast
   - Good baseline model

4. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Non-parametric approach

## 🏗️ System Architecture

```
Input Data → Data Preprocessing → Model Training → Model Evaluation → Prediction → Output
```

### Detailed Workflow:

1. **Data Collection** - Load student academic data
2. **Data Preprocessing** - Clean, normalize, and encode data
3. **Train/Test Split** - 80% training, 20% testing
4. **Model Training** - Train multiple ML algorithms
5. **Model Evaluation** - Compare performance metrics
6. **Prediction** - Predict student performance
7. **Visualization** - Generate insights and reports

## 💻 Technology Stack

- **Language:** Python 3.8+
- **Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical visualization
  - `streamlit` - Web interface

## 📁 Project Structure

```
manan-project-ml/
│
├── data/
│   └── student_performance.csv      # Dataset
│
├── src/
│   ├── data_preprocessing.py        # Data preprocessing module
│   ├── model_training.py            # Model training module
│   ├── prediction.py                # Prediction module
│   └── visualization.py             # Visualization module
│
├── models/
│   └── best_model.pkl               # Saved trained model
│
├── outputs/
│   └── (generated visualizations)   # Charts and graphs
│
├── main.py                          # Main execution script
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🚀 Installation & Setup

### 1. Clone or Download the Project

```bash
cd manan-project-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Dataset

Ensure `data/student_performance.csv` exists with proper data.

## 🎮 Usage

### Option 1: Command Line Interface

Run the complete pipeline:

```bash
python main.py
```

This will:
- Load and preprocess data
- Train all models
- Compare performance
- Save the best model
- Generate visualizations
- Make sample predictions

### Option 2: Web Interface (Streamlit)

Launch the interactive web application:

```bash
streamlit run app.py
```

Features:
- 📊 **Dataset Overview** - View and analyze dataset
- 🤖 **Train Model** - Train and compare ML models
- 🔮 **Make Prediction** - Predict individual student performance
- 📈 **Analytics** - View insights and identify weak students

### Option 3: Individual Modules

#### Data Preprocessing
```bash
cd src
python data_preprocessing.py
```

#### Model Training
```bash
cd src
python model_training.py
```

#### Make Predictions
```bash
cd src
python prediction.py
```

#### Generate Visualizations
```bash
cd src
python visualization.py
```

## 📈 Model Evaluation Metrics

The system evaluates models using:

- **Accuracy** - Overall correctness
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Detailed prediction breakdown

### Expected Results:
- Accuracy: ~90-95%
- Precision: ~88-93%
- Recall: ~87-92%
- F1-Score: ~88-92%

## 🎯 Use Cases

### For Teachers:
1. Upload student academic data
2. System identifies students at risk of failing
3. Provide targeted support and intervention
4. Monitor improvement over time

### For Institutions:
1. Early warning system for academic performance
2. Data-driven decision making
3. Resource allocation for student support
4. Improve overall pass rates

## 📊 Sample Output

```
Student 1 → Pass (Confidence: 95%)
Student 2 → Fail (Confidence: 88%)
Student 3 → Pass (Confidence: 92%)
```

### Recommendations for Weak Students:
- Increase daily study hours
- Improve attendance (target: >75%)
- Focus on internal exam preparation
- Complete assignments with better quality
- Increase class participation

## ✅ Advantages

- ✓ Simple and easy to implement
- ✓ Fast prediction (real-time)
- ✓ Useful for teachers and administrators
- ✓ Data-driven decision making
- ✓ Scalable to large datasets
- ✓ Early intervention possible
- ✓ Multiple algorithm comparison
- ✓ Visual insights and reports

## ⚠️ Limitations

- Small dataset may reduce accuracy
- Only academic factors considered (no social/economic factors)
- Requires historical data for training
- No real-time data integration
- Binary classification only (Pass/Fail)

## 🔮 Future Enhancements

1. **Deep Learning Models** - Implement neural networks
2. **GUI Dashboard** - Advanced web dashboard with more features
3. **Recommendation System** - Personalized study recommendations
4. **Subject-wise Prediction** - Predict performance per subject
5. **Attendance Tracker** - Real-time attendance monitoring
6. **Mobile Application** - Android/iOS app
7. **Multi-class Classification** - Grade prediction (A, B, C, D, F)
8. **Integration** - Connect with school management systems
9. **Real-time Alerts** - Notify teachers about at-risk students
10. **Historical Trends** - Track student progress over time

## 📚 Module Breakdown

### Module 1: Data Collection
- Load dataset from CSV
- Validate data format
- Handle file errors

### Module 2: Data Preprocessing
- Handle missing values
- Normalize/scale features
- Encode categorical variables
- Split train/test data

### Module 3: Model Training
- Initialize ML algorithms
- Train models on training data
- Hyperparameter tuning
- Save trained models

### Module 4: Prediction
- Load trained model
- Accept input features
- Generate predictions
- Provide confidence scores

### Module 5: Evaluation
- Calculate accuracy metrics
- Generate confusion matrix
- Compare model performance
- Select best model

### Module 6: Visualization
- Class distribution charts
- Feature correlation heatmap
- Model comparison graphs
- Confusion matrices
- Feature importance plots

## 🧪 Testing

### Test with Sample Data:

**Good Student:**
```python
study_hours = 7
attendance = 92
internal_marks = 88
assignment_marks = 85
previous_marks = 86
participation = 9
project_marks = 89
# Expected: Pass
```

**Weak Student:**
```python
study_hours = 2
attendance = 58
internal_marks = 44
assignment_marks = 40
previous_marks = 48
participation = 3
project_marks = 42
# Expected: Fail
```

## 📖 Dataset Information

The dataset contains 100 student records with balanced Pass/Fail distribution.

**Source:** Custom generated dataset based on real academic patterns

**Format:** CSV (Comma-Separated Values)

**Size:** ~100 records

**Features:** 7 input features + 1 target variable

## 🤝 Contributing

This is a minor project for educational purposes. Suggestions and improvements are welcome!

## 📄 License

This project is created for academic purposes as a minor project.

## 👨‍💻 Author

**Minor Project - Student Performance Prediction**

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the code comments
3. Test with sample data
4. Verify all dependencies are installed

## 🎓 Academic Use

This project is suitable for:
- Minor/Major projects
- Machine Learning coursework
- Data Science assignments
- Educational demonstrations
- Research purposes

## 📝 Report Sections

For academic report writing, this project covers:

1. **Introduction** - Problem statement and objectives
2. **Literature Review** - ML algorithms overview
3. **Methodology** - System design and architecture
4. **Implementation** - Code and modules
5. **Results** - Model performance and accuracy
6. **Conclusion** - Findings and future work
7. **References** - Libraries and resources used

## 🔗 References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Streamlit Documentation: https://streamlit.io/
- Machine Learning Concepts: Various academic resources

---

**Last Updated:** 2024

**Version:** 1.0

**Status:** ✅ Complete and Ready to Use
