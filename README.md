<<<<<<< HEAD
# Towards Intelligent Dermatology: Deep Learning-Based Detection and Classification of 22 Skin Conditions

---

## 🚀 Quick Setup (PyCharm / Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Install TensorFlow for live model inference
```bash
pip install tensorflow>=2.15.0
```

### 3. (Optional) Add your trained model
Place your exported ensemble model file at:
```
models/final_model.keras
```
> If no model is found, the system runs in **demo mode** automatically.

### 4. Run the application
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🔑 Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Health Worker | `drkofi` | `password123` |

---

## 📁 Project Structure

```
skindx/
├── app.py                  # Main Flask application
├── requirements.txt
├── skindx.db               # SQLite database (auto-created)
├── models/
│   └── final_model.keras   # Place your trained model here
├── utils/
│   ├── db.py               # Database initialisation
│   ├── predictor.py        # Model inference
│   └── reports.py          # PDF report generation
├── templates/
│   ├── base.html           # Layout with sidebar
│   ├── login.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── analytics.html
│   ├── admin.html
│   ├── profile.html
│   ├── model_info.html
│   └── report.html
└── static/
    ├── uploads/            # Uploaded images
    └── reports/            # Generated PDF reports
```

---

## 🎯 Features

- **22-Class Skin Disease Classification** using Triple-Branch CNN Ensemble
- **Patient Record Management** with search and filtering
- **PDF Clinical Report Generation** with recommendations
- **Analytics Dashboard** with charts and statistics
- **Admin Panel** for user management
- **Responsive Design** — works on mobile and desktop
- **Demo Mode** — runs without a trained model for testing

---

## 🧠 Model Architecture
- EfficientNetB3 (1,536-dim features)
- MobileNetV2 (1,280-dim features)  
- ResNet50 (2,048-dim features)
- Concatenated → Dense(512) → BatchNorm → Dropout(0.4) → Softmax(22)

---

*Developed for MSc thesis defence — GCTU Department of Computer Science*
=======
# Towards-Intelligent-Dermatology-DeepL-Based-Detection-and-Classification-of-22-Skin-Conditions
>>>>>>> 86c254420d9394ab0386ecc8b02c2f3d497e9a87
