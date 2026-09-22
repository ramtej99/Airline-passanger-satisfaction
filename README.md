# ✈️ Airline Passenger Satisfaction Analytics & ML Dashboard

A complete end-to-end Data Analytics and Machine Learning project built on a real-world airline passenger satisfaction dataset. The project features exploratory data analysis, three trained classification models, and a professional interactive Streamlit dashboard with KPI cards, dynamic filters, visualizations, model evaluation, and live passenger satisfaction prediction.

---

## 📌 Project Overview

Airlines rely heavily on passenger experience metrics to maintain competitive edge. This project analyses **58,313 airline passenger records** to understand the key drivers of satisfaction, build accurate predictive ML models, and present findings through an interactive dashboard.

---

## ❓ Problem Statement

Given passenger demographics, travel details, and service ratings, can we accurately predict whether a passenger will be **satisfied** or **neutral/dissatisfied** with their flight experience? And what are the most influential factors driving that satisfaction?

---

## 🎯 Objectives

1. Perform comprehensive **Exploratory Data Analysis (EDA)** to uncover patterns and insights.
2. Identify key features that drive passenger satisfaction.
3. Train and evaluate multiple **ML classification models** for accurate prediction.
4. Build a **professional interactive Streamlit dashboard** with real-time prediction capability.
5. Provide actionable insights for airline service improvement.

---

## 📂 Dataset Information

| Property | Value |
|---|---|
| **File** | `airline_satisfaction_data.csv` (https://drive.google.com/file/d/1y06clgsbrpjnEO5QVB-1hR3Ea_ZwFBCl/view?usp=drivesdk) |
| **Records** | 58,313 passengers |
| **Features** | 22 columns |
| **Target** | `satisfaction` (Boolean → `satisfied` / `neutral or dissatisfied`) |
| **Satisfied** | 25,332 passengers (43.4%) |
| **Dissatisfied** | 32,981 passengers (56.6%) |

### Feature Categories

**Demographic:** Gender, Customer Type, Age

**Travel Details:** Type of Travel, Class, Flight Distance

**Service Ratings (0–5):** Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Cleanliness

**Delays:** Departure Delay in Minutes, Arrival Delay in Minutes

---

## 🛠️ Technologies Used

| Category | Libraries/Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Web Framework** | Streamlit 1.32+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting |

---

## 🔬 Methodology

```
Raw CSV Data
    │
    ▼
Data Cleaning & Preprocessing
  • Fix 'Busi' → 'Business' typo
  • Standardise Customer Type casing
  • Convert Boolean satisfaction → string labels
  • Fill Arrival Delay NaNs with median
  • Clip rating values to 0–5 range
  • Engineer Age Group & Distance Category features
    │
    ▼
Exploratory Data Analysis
  • Satisfaction distribution
  • Demographic breakdown (Gender, Class, Travel Type, Customer Type)
  • Age and flight distance analysis
  • Delay analysis
  • Service rating heatmap & radar chart
  • Feature correlation matrix
    │
    ▼
ML Preprocessing
  • Label encode categorical columns
  • StandardScaler normalization
  • 80/20 stratified train/test split
    │
    ▼
Model Training
  • Logistic Regression
  • Random Forest (150 trees, max_depth=12)
  • Gradient Boosting (150 estimators, lr=0.1)
    │
    ▼
Evaluation
  • Accuracy, Precision, Recall, F1-Score
  • Confusion Matrix
  • ROC Curve & AUC Score
  • Feature Importance (RF)
    │
    ▼
Interactive Dashboard
  • KPI Cards | Filters | EDA Charts
  • Model Comparison | Confusion Matrices
  • Live Satisfaction Prediction
```

---

## 📁 Project Structure

```
airline_satisfaction/
├── app.py                          # Main Streamlit application (all logic)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Project_Report.docx             # Full college project report
└── airline_satisfaction_data.csv   # Dataset
```

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites

- Python 3.9 or higher
- pip package manager

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

### 4. Navigation

Use the **sidebar** to navigate between the four pages and apply demographic/flight filters.

---

## 🖥️ Dashboard Features

### 📊 Page 1 — Overview & KPIs
- **6 KPI cards**: Total passengers, Satisfied count, Satisfaction rate %, Average age, Average flight distance, Average arrival delay
- Satisfaction distribution pie chart and bar chart
- Demographic breakdown (Gender, Customer Type, Travel Type, Class)
- Summary statistics table

### 🔍 Page 2 — EDA & Visualizations
- Age distribution histograms and age group breakdown
- Flight distance analysis by satisfaction
- Departure and arrival delay distributions
- Service ratings heatmap (average per satisfaction group)
- Service ratings radar chart
- Feature correlation matrix
- Interactive per-rating distribution explorer

### 🤖 Page 3 — ML Model Performance
- Side-by-side model comparison table (Accuracy, Precision, Recall, F1, AUC)
- Grouped bar chart comparison
- ROC curves for all three models
- Confusion matrices for each model
- Feature importance chart (Random Forest)
- Detailed classification report (selectable model)

### 🎯 Page 4 — Predict Satisfaction
- Interactive form: demographics, flight details, 13 service ratings
- Select prediction model
- Instant prediction result with probability gauge
- Confidence score display

---

## 📊 Actual Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8770 | 0.8765 | 0.8346 | 0.8550 | 0.9254 |
| Random Forest | 0.9532 | 0.9573 | 0.9339 | 0.9455 | 0.9912 |
| **Gradient Boosting** | **0.9581** | **0.9683** | **0.9341** | **0.9509** | **0.9934** |

> Results computed on 20% stratified test set (11,663 records), random_state=42. **Gradient Boosting** achieves the highest F1-Score (0.9509) and AUC (0.9934).

---

## 💡 Key Insights

1. **Online Boarding** and **Inflight Entertainment** are the top predictors of satisfaction.
2. **Business class** passengers show significantly higher satisfaction rates than Eco passengers.
3. **Loyal customers** are substantially more satisfied than disloyal ones.
4. **Business travellers** are more satisfied than personal travellers.
5. Passengers aged **31–60** show the highest satisfaction rates.
6. **Arrival delay** has a stronger negative correlation with satisfaction than departure delay.
7. **Long-haul flights (1500–3000 km)** tend to produce more satisfied passengers.
8. Service ratings for **Inflight wifi, Seat comfort, and On-board service** differ most between satisfied and dissatisfied groups.

---

## 📎 Dataset Source

- **Dataset:** Airline Passenger Satisfaction  
- **Source:** (https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
- **License:** Public domain / educational use

---

## 👤 Author

**Ram Kumar G**
**Airline Satisfaction Analytics Project**  
**LinkedIn:** Ram Kumar G(www.linkedin.com/in/ramkumar-g-245685302)
**Email:** ramkumar91847@gmail.com  
  

---

*Built with ❤️ using Python, Streamlit, Scikit-learn, Matplotlib, and Seaborn*
