# Pancreatic_Cancer_Detection_Using_Machine_Learning

A Flask-based web application for **early detection of pancreatic cancer** using advanced Machine Learning techniques.  
The system is trained using **Random Forest, XGBoost, and Logistic Regression models**, with **Random Forest selected for deployment** based on best performance.

---

## 🚀 Features

✔️ Cancer risk prediction using trained ML models  
✔️ Probability score for better interpretability  
✔️ Analysis dashboard with key feature insights  
✔️ PDF report generation  
✔️ SQLite database for storing prediction history  
✔️ Clean UI with multiple pages (Home, Detection, Analysis, About)  
✔️ Separate ML model (.pkl) hosted in GitHub Release  

---

## 📊 Model Performance
| Model               | Accuracy   | Precision  | Recall     | F1-Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 62.77%     | 61.55%     | 68.04%     | 64.63%     |
| Random Forest       | **82.53%** | **81.89%** | **83.52%** | **82.70%** |
| XGBoost             | 80.01%     | 89.16%     | 68.34%     | 77.37%     |

📌 Best Model: Random Forest (Accuracy: 82.53%)

---

## 📁 Project Structure
- `app.py` – Main Flask application  
- `static/` – CSS, JS, and images  
- `templates/` – HTML pages  
- `PancreaticGuard/` – ML model, encoders, scalers, configs  
- `users.db` – SQLite database  

---

## 🔗 Download ML Model (.pkl)

The trained ML model is hosted under **GitHub Releases** because it exceeds the 100 MB repository limit.

👉 **Download the model here:**  
https://github.com/pratiksha123-sys/Pancreatic_Cancer_Detection_Using_Machine_Learning/releases/tag/v1.0

After downloading, **place the file in the following path** inside your project:

