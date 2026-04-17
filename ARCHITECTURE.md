# System Architecture - Student Performance Prediction

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                        │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │  Streamlit Web   │              │  Command Line    │        │
│  │  Application     │              │  Interface       │        │
│  └──────────────────┘              └──────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Prediction   │  │ Visualization│  │ Model        │         │
│  │ Module       │  │ Module       │  │ Training     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA PROCESSING LAYER                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │         Data Preprocessing Module                │          │
│  │  • Load Data                                     │          │
│  │  • Handle Missing Values                         │          │
│  │  • Encode Labels                                 │          │
│  │  • Feature Engineering                           │          │
│  │  • Train/Test Split                              │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Decision     │  │ Random       │  │ Logistic     │         │
│  │ Tree         │  │ Forest       │  │ Regression   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐                                              │
│  │ K-Nearest    │                                              │
│  │ Neighbors    │                                              │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ CSV Dataset  │  │ Trained      │  │ Output       │         │
│  │ (Input)      │  │ Models (PKL) │  │ Visualizations│        │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Diagram

```
START
  │
  ▼
┌─────────────────┐
│ Load CSV Data   │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Check Missing   │
│ Values          │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Handle Missing  │
│ Values          │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Encode Labels   │
│ (Pass/Fail →    │
│  1/0)           │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Extract         │
│ Features (X)    │
│ & Target (y)    │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Split Data      │
│ 80% Train       │
│ 20% Test        │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Train Models    │
│ • Decision Tree │
│ • Random Forest │
│ • Log Reg       │
│ • KNN           │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Evaluate Models │
│ • Accuracy      │
│ • Precision     │
│ • Recall        │
│ • F1-Score      │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Select Best     │
│ Model           │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Save Model      │
│ (Pickle)        │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Make            │
│ Predictions     │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Generate        │
│ Visualizations  │
└─────────────────┘
  │
  ▼
END
```

## 🏗️ Module Architecture

### 1. Data Preprocessing Module (`data_preprocessing.py`)

**Responsibilities:**
- Load dataset from CSV
- Validate data integrity
- Handle missing values
- Encode categorical variables
- Split data into train/test sets
- Generate statistical summaries

**Key Classes:**
- `DataPreprocessor`

**Key Methods:**
- `load_data()`
- `check_missing_values()`
- `handle_missing_values()`
- `encode_labels()`
- `prepare_features()`
- `split_data()`

### 2. Model Training Module (`model_training.py`)

**Responsibilities:**
- Initialize ML algorithms
- Train multiple models
- Evaluate model performance
- Compare algorithms
- Save best model

**Key Classes:**
- `ModelTrainer`

**Key Methods:**
- `initialize_models()`
- `train_model()`
- `evaluate_model()`
- `train_all_models()`
- `display_comparison()`
- `save_model()`

### 3. Prediction Module (`prediction.py`)

**Responsibilities:**
- Load trained model
- Make predictions for new students
- Batch prediction support
- Identify weak students
- Generate recommendations

**Key Classes:**
- `StudentPerformancePredictor`

**Key Methods:**
- `load_model()`
- `predict_single_student()`
- `predict_batch()`
- `predict_with_details()`
- `identify_weak_students()`

### 4. Visualization Module (`visualization.py`)

**Responsibilities:**
- Generate charts and graphs
- Create confusion matrices
- Plot model comparisons
- Feature importance visualization
- Statistical plots

**Key Classes:**
- `PerformanceVisualizer`

**Key Methods:**
- `plot_class_distribution()`
- `plot_feature_distributions()`
- `plot_correlation_matrix()`
- `plot_confusion_matrix()`
- `plot_model_comparison()`
- `plot_feature_importance()`

## 🎯 Component Interaction

```
┌──────────────┐
│   main.py    │ ◄─── Entry point for CLI execution
└──────────────┘
       │
       ├─────► DataPreprocessor ─────► Load & Clean Data
       │
       ├─────► ModelTrainer ──────────► Train & Evaluate
       │
       ├─────► PerformanceVisualizer ─► Generate Charts
       │
       └─────► StudentPerformancePredictor ─► Make Predictions


┌──────────────┐
│    app.py    │ ◄─── Entry point for Web UI
└──────────────┘
       │
       └─────► Streamlit Interface
                    │
                    ├─► Dataset Overview Tab
                    ├─► Train Model Tab
                    ├─► Make Prediction Tab
                    └─► Analytics Tab
```

## 📊 Database Schema (CSV)

```
student_performance.csv
├── study_hours         (int)    : 0-10
├── attendance          (int)    : 0-100
├── internal_marks      (int)    : 0-100
├── assignment_marks    (int)    : 0-100
├── previous_marks      (int)    : 0-100
├── participation       (int)    : 1-10
├── project_marks       (int)    : 0-100
└── final_result        (string) : Pass/Fail
```

## 🔐 Model Storage Format

```
best_model.pkl (Pickle file)
├── model              : Trained sklearn model object
├── model_name         : String (e.g., "Decision Tree")
└── accuracy           : Float (e.g., 0.92)
```

## 🎨 User Interface Architecture

### Web Application (Streamlit)

```
┌─────────────────────────────────────────┐
│           Navigation Tabs               │
├─────────────────────────────────────────┤
│                                         │
│  Tab 1: Dataset Overview                │
│  ├─ Metrics (Total, Pass, Fail)        │
│  ├─ Sample Data Table                  │
│  ├─ Statistical Summary                │
│  └─ Class Distribution Chart           │
│                                         │
│  Tab 2: Train Model                     │
│  ├─ Algorithm Selection                │
│  ├─ Train Button                       │
│  ├─ Model Comparison Table             │
│  └─ Performance Chart                  │
│                                         │
│  Tab 3: Make Prediction                 │
│  ├─ Input Sliders (7 features)         │
│  ├─ Predict Button                     │
│  ├─ Result Display                     │
│  └─ Recommendations                    │
│                                         │
│  Tab 4: Analytics                       │
│  ├─ Correlation Analysis               │
│  ├─ Pass/Fail Statistics               │
│  └─ Weak Student Identification        │
│                                         │
└─────────────────────────────────────────┘
```

## 🔧 Technology Stack Details

### Core Technologies
- **Python 3.8+**: Programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization

### Web Framework
- **streamlit**: Interactive web application

### Model Persistence
- **pickle**: Model serialization

## 📈 Performance Optimization

### Data Processing
- Efficient pandas operations
- Vectorized numpy computations
- Minimal data copying

### Model Training
- Optimized hyperparameters
- Limited tree depth to prevent overfitting
- Stratified sampling for balanced splits

### Prediction
- Pre-loaded models in memory
- Batch prediction support
- Cached model loading (Streamlit)

## 🔒 Error Handling

```
┌─────────────────┐
│ User Input      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Validation      │
│ • Type check    │
│ • Range check   │
│ • Format check  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Try-Except      │
│ Blocks          │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Error Messages  │
│ & Logging       │
└─────────────────┘
```

## 🚀 Deployment Architecture

### Local Deployment
```
User Machine
├── Python Environment
├── Required Libraries
├── Project Files
└── Data Files
```

### Web Deployment (Future)
```
Cloud Server
├── Streamlit Cloud / Heroku
├── GitHub Repository
├── requirements.txt
└── app.py
```

## 📦 File Organization

```
manan-project-ml/
│
├── data/                    # Data storage
│   └── student_performance.csv
│
├── src/                     # Source code modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│   └── visualization.py
│
├── models/                  # Trained models
│   └── best_model.pkl
│
├── outputs/                 # Generated visualizations
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix_*.png
│   └── model_comparison.png
│
├── main.py                  # CLI entry point
├── app.py                   # Web UI entry point
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── VIVA_QUESTIONS.md       # Q&A for viva
└── ARCHITECTURE.md         # This file
```

## 🔄 Execution Flow

### CLI Mode (main.py)
1. Initialize preprocessor
2. Load and clean data
3. Generate initial visualizations
4. Train all models
5. Compare performance
6. Save best model
7. Generate model visualizations
8. Make sample predictions
9. Identify weak students
10. Display summary

### Web Mode (app.py)
1. Launch Streamlit server
2. Load cached resources
3. Display navigation tabs
4. Handle user interactions
5. Update UI dynamically
6. Process predictions in real-time
7. Generate visualizations on-demand

## 🎯 Design Patterns Used

### 1. Object-Oriented Design
- Encapsulation in classes
- Modular architecture
- Reusable components

### 2. Separation of Concerns
- Data processing separate from ML
- Visualization independent module
- UI decoupled from logic

### 3. Factory Pattern
- Model initialization in ModelTrainer
- Dynamic model creation

### 4. Singleton Pattern
- Streamlit caching for model loading
- Single instance of predictor

## 📊 Scalability Considerations

### Current Scale
- 100 student records
- 7 features
- 4 algorithms
- Local execution

### Future Scale
- Thousands of students
- Additional features
- More algorithms
- Cloud deployment
- Real-time updates
- Multi-user support

---

**Architecture Version:** 1.0  
**Last Updated:** 2024  
**Status:** Production Ready
