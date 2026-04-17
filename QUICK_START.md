# Quick Start Guide - Student Performance Prediction

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

Open terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Interface

#### Option A: Run Complete Pipeline (CLI)

```bash
python main.py
```

This will:
- ✅ Load and preprocess data
- ✅ Train all 4 ML models
- ✅ Compare performance
- ✅ Save best model
- ✅ Generate visualizations
- ✅ Make sample predictions

#### Option B: Launch Web Application

```bash
streamlit run app.py
```

Then open your browser to the displayed URL (usually http://localhost:8501)

---

## 📊 What You'll Get

### After Running `main.py`:

1. **Console Output:**
   - Data statistics
   - Model training progress
   - Accuracy comparison
   - Sample predictions

2. **Saved Files:**
   - `models/best_model.pkl` - Trained model
   - `outputs/*.png` - Visualization charts

3. **Performance Metrics:**
   - Accuracy: ~90-95%
   - Precision, Recall, F1-Score
   - Confusion matrices

### Using the Web App (`app.py`):

**Tab 1: Dataset Overview**
- View student data
- See statistics
- Class distribution

**Tab 2: Train Model**
- Click "Start Training"
- Compare 4 algorithms
- See best model

**Tab 3: Make Prediction**
- Adjust sliders for student features
- Click "Predict Performance"
- Get Pass/Fail result + recommendations

**Tab 4: Analytics**
- Correlation analysis
- Pass/Fail comparisons
- Identify weak students

---

## 🎯 Quick Test

### Test Prediction (Good Student):
```python
python -c "from src.prediction import StudentPerformancePredictor; p = StudentPerformancePredictor(); p.predict_with_details(7, 90, 85, 88, 82, 9, 87)"
```

### Test Prediction (Weak Student):
```python
python -c "from src.prediction import StudentPerformancePredictor; p = StudentPerformancePredictor(); p.predict_with_details(2, 55, 42, 38, 45, 3, 40)"
```

---

## 🔧 Individual Module Testing

### Test Data Preprocessing:
```bash
cd src
python data_preprocessing.py
```

### Test Model Training:
```bash
cd src
python model_training.py
```

### Test Predictions:
```bash
cd src
python prediction.py
```

### Test Visualizations:
```bash
cd src
python visualization.py
```

---

## 📁 Project Structure

```
manan-project-ml/
├── data/
│   └── student_performance.csv    ← Dataset (100 students)
├── src/
│   ├── data_preprocessing.py      ← Data cleaning
│   ├── model_training.py          ← ML training
│   ├── prediction.py              ← Make predictions
│   └── visualization.py           ← Generate charts
├── models/
│   └── best_model.pkl             ← Saved model (created after training)
├── outputs/
│   └── *.png                      ← Charts (created after running)
├── main.py                        ← Run everything (CLI)
├── app.py                         ← Web interface
└── requirements.txt               ← Dependencies
```

---

## ⚡ Common Commands

| Task | Command |
|------|---------|
| Install packages | `pip install -r requirements.txt` |
| Run full pipeline | `python main.py` |
| Launch web app | `streamlit run app.py` |
| Train models only | `cd src && python model_training.py` |
| Make predictions | `cd src && python prediction.py` |
| Generate charts | `cd src && python visualization.py` |

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'src'"
**Solution:** Run from project root directory
```bash
cd manan-project-ml
python main.py
```

### Issue: "Model file not found"
**Solution:** Train the model first
```bash
cd src
python model_training.py
```

### Issue: Streamlit won't start
**Solution:** Check if port 8501 is available or specify different port
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Expected Output

### Console Output (main.py):
```
================================================================================
                    STUDENT PERFORMANCE PREDICTION SYSTEM
================================================================================

[STEP 1] DATA PREPROCESSING
--------------------------------------------------------------------------------
Loading dataset...
Dataset loaded successfully! Shape: (100, 8)
...

[STEP 3] MODEL TRAINING
--------------------------------------------------------------------------------
Training Decision Tree...
Decision Tree training completed!
Accuracy: 92.00%
...

🏆 BEST MODEL: Decision Tree with 92.00% accuracy
```

### Web App:
- Interactive sliders
- Real-time predictions
- Beautiful charts
- Model comparison tables

---

## 🎓 For Your Project Report

After running the project, you'll have:

1. ✅ **Working Code** - All modules functional
2. ✅ **Results** - Accuracy metrics and comparisons
3. ✅ **Visualizations** - Charts for report
4. ✅ **Documentation** - README, Architecture, Viva Q&A
5. ✅ **Demo** - Web application for presentation

---

## 🎯 Next Steps

1. **Run the project** - Execute `python main.py`
2. **Explore web app** - Launch `streamlit run app.py`
3. **Review outputs** - Check `outputs/` folder for charts
4. **Read documentation** - Go through README.md
5. **Prepare for viva** - Study VIVA_QUESTIONS.md
6. **Customize** - Modify dataset or add features

---

## 💡 Pro Tips

- 📸 **Take screenshots** of the web app for your report
- 📊 **Save all charts** from the outputs folder
- 📝 **Note the accuracy** of each model for comparison
- 🎤 **Practice demo** using the web interface
- 📚 **Understand the code** - Read through each module

---

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Review `VIVA_QUESTIONS.md` for Q&A
3. Read `ARCHITECTURE.md` for system design
4. Look at code comments in each module

---

**Ready to impress your evaluators! 🎓✨**
