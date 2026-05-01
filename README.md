# 🧠 Wafer Fault Detection System (End-to-End ML Project)

## 🚀 Overview

This project is an end-to-end Machine Learning system designed to detect wafer faults using sensor data. It includes data ingestion, preprocessing, model training, prediction pipeline, and a Flask-based web interface for real-time predictions.

---

## 🎯 Problem Statement

In semiconductor manufacturing, detecting faulty wafers early is critical. This project uses sensor data to classify wafers as:

* ✅ **Good**
* ❌ **Bad**

---

## 🏗️ Project Architecture

```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   ├── utils/
│   ├── exception.py
│   ├── logger.py
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│
├── prediction_artifacts/
├── templates/
│   ├── upload_file.html
│
├── static/
│
├── config/
│   ├── model.yaml
│
├── app.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Tech Stack

* **Language:** Python
* **ML Libraries:** Scikit-learn, XGBoost
* **Web Framework:** Flask
* **Data Handling:** Pandas, NumPy
* **Visualization (optional):** Matplotlib, Seaborn

---

## 🔄 Pipeline Flow

### 1. Training Pipeline

```
Data Ingestion → Data Transformation → Model Training → Save Model
```

### 2. Prediction Pipeline

```
Upload CSV → Preprocess → Predict → Generate Output CSV → Download
```

---

## 📊 Dataset

* Sensor-based wafer dataset
* Multiple numerical features
* Target column: `Good/Bad`

  * `1` → Good
  * `-1` → Bad

---

## 🧪 How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/wafer-fault-detection.git
cd wafer-fault-detection
```

### Step 2: Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the model

```bash
python app.py
```

Then open:

```
http://localhost:5000/train
```

---

### Step 5: Run prediction

Go to:

```
http://localhost:5000/predict
```

Upload CSV file → Download predictions automatically

---

## 📥 Input Format (Important)

### ✔ Correct Input (for prediction)

* Only feature columns
* ❌ Do NOT include `Good/Bad`

### ❌ Wrong Input

* Including target column will cause:

```
Feature names mismatch error
```

---

## 📤 Output

Generated file:

```
prediction_file.csv
```

Contains:

* Original features
* Predicted label (`good` / `bad`)

---

## 🧠 Key Learnings

* Handling ML pipelines end-to-end
* Debugging real-world data issues
* Managing feature consistency between training and prediction
* Building deployable ML systems using Flask



## 🤝 Contribution

Feel free to fork this repo and improve it!

---

## 📬 Contact

If you found this useful, connect with me on LinkedIn 🚀
www.linkedin.com/in/aditya-narayan-chaubey-857b9929a
