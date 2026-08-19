# 📡 Sonar Prediction Model

**Classify sonar returns as Rock or Mine using Machine Learning — with an interactive Streamlit app.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 Overview

**Sonar Prediction Model** is a machine learning project that classifies sonar signal returns as either a **Rock (R)** or a **Mine (M)**, based on the classic [Sonar, Mines vs. Rocks dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks). It uses a **Random Forest Classifier** trained on 60 frequency-band energy features extracted from sonar signals, and ships with a clean, interactive **Streamlit** web app so you can test predictions in real time.

> 🎯 Enter 60 sonar signal values manually, load a sample row from the dataset, and instantly see whether the model predicts a **Rock** or a **Mine** — along with the model's confidence and evaluation metrics.

---

## ✨ Features

- 🔍 **Real-time classification** — Predict Rock vs. Mine from 60 sonar signal-strength inputs
- 🎛️ **Interactive UI** — Adjust signal values with number inputs or load a sample from the dataset
- 🌲 **Random Forest model** — 300 estimators, balanced class weights, feature scaling via `StandardScaler`
- 📊 **Model transparency** — Sidebar shows live test accuracy, training size, and feature count
- 🧮 **Confusion matrix visualization** — Inspect model performance directly in the app
- ⚡ **Cached training** — Fast reloads using Streamlit's `@st.cache_data` and `@st.cache_resource`

---

## 🖼️ Demo

```
┌─────────────────────────────────────────────┐
│  📡 Sonar Signal Classifier                  │
│  Predict whether a sonar return is a mine    │
│  or a rock.                                  │
│                                               │
│  [ Sample signal ▼ ]                         │
│  Feature 1  Feature 2  Feature 3  Feature 4  │
│  [0.02]     [0.03]     [0.05]     [0.09]     │
│  ...                                         │
│                                               │
│  [ Classify Signal ]                         │
│  ✅ Prediction: Rock (91.3% confidence)      │
└─────────────────────────────────────────────┘
```

*Run the app locally to see it live — see [Getting Started](#-getting-started) below.*

---

## 🗂️ Project Structure

```
Sonar-Prediction-Model/
├── Sonar_Prediction_Model.ipynb   # Exploratory analysis & model development notebook
├── app.py                         # Streamlit web application
├── sonar.csv                      # Sonar signal dataset (Mines vs. Rocks)
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Tech Stack

| Category            | Tools                                  |
|----------------------|-----------------------------------------|
| Language             | Python                                  |
| Data Handling         | pandas, numpy                          |
| Machine Learning      | scikit-learn (RandomForestClassifier)  |
| Visualization         | matplotlib, seaborn                    |
| Web App               | Streamlit                              |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### 1. Clone the repository

```bash
git clone https://github.com/tafrusaidev/Sonar-Prediction-Model.git
cd Sonar-Prediction-Model
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 How It Works

1. **Data Loading** — The dataset (`sonar.csv`) contains 60 numeric features per sample, each representing energy within a particular frequency band, plus a label (`R` for Rock, `M` for Mine).
2. **Preprocessing** — Features are scaled using `StandardScaler` to normalize the input range.
3. **Train/Test Split** — Data is split 80/20 with stratification to preserve class balance.
4. **Model Training** — A `RandomForestClassifier` (300 trees, balanced class weights) is trained on the scaled features.
5. **Evaluation** — Accuracy and a confusion matrix are computed on the held-out test set and surfaced in the app sidebar and expander.
6. **Inference** — Users can input a custom 60-value signal or select a sample row, and the app returns a prediction with confidence score.

---

## 📓 Notebook

`Sonar_Prediction_Model.ipynb` contains the exploratory data analysis and model experimentation behind the app — including data inspection, feature distributions, and model comparison. Open it with Jupyter or VS Code to explore the reasoning behind the final model choice.

```bash
jupyter notebook Sonar_Prediction_Model.ipynb
```

---

## 📚 Dataset

This project uses the **Connectionist Bench (Sonar, Mines vs. Rocks)** dataset, originally collected to study the classification of sonar signals bounced off a metal cylinder (mine) versus a rock, under a variety of conditions. It contains 208 samples with 60 continuous features each.

- 📎 Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)

---

## 🗺️ Roadmap

- [ ] Add model comparison (SVM, Logistic Regression, XGBoost)
- [ ] Add cross-validation metrics to the UI
- [ ] Deploy live demo (Streamlit Community Cloud)
- [ ] Add unit tests for data loading and model training

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tafrusai Dev**
GitHub: [@tafrusaidev](https://github.com/tafrusaidev)

---

<p align="center">⭐ If you found this project useful, consider giving it a star!</p>
