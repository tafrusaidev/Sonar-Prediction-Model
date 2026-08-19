from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / "sonar.csv"


@st.cache_data
def load_data():
	data = pd.read_csv(DATA_PATH, header=None)
	features = data.iloc[:, :-1].astype(float)
	features.columns = [f"Feature {index}" for index in range(1, features.shape[1] + 1)]
	labels = data.iloc[:, -1].map({"R": "Rock", "M": "Mine"})
	return features, labels


@st.cache_resource
def train_model(features, labels):
	x_train, x_test, y_train, y_test = train_test_split(
		features, labels, test_size=0.2, random_state=42, stratify=labels
	)
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)
	model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
	model.fit(x_train_scaled, y_train)
	predictions = model.predict(x_test_scaled)
	accuracy = accuracy_score(y_test, predictions)
	matrix = confusion_matrix(y_test, predictions, labels=["Mine", "Rock"])
	return model, scaler, accuracy, matrix


st.set_page_config(page_title="Sonar Classifier", page_icon="📡", layout="wide")
st.title("Sonar Signal Classifier")
st.caption("Predict whether a sonar return is more likely to be a mine or a rock.")

try:
	features, labels = load_data()
except FileNotFoundError:
	st.error(f"Dataset not found: {DATA_PATH}")
	st.stop()

model, scaler, accuracy, matrix = train_model(features, labels)

with st.sidebar:
	st.header("Model overview")
	st.metric("Test accuracy", f"{accuracy:.1%}")
	st.write(f"Training rows: {len(features):,}")
	st.write(f"Signal features: {features.shape[1]}")

st.subheader("Enter a sonar signal")
st.write("Use the sliders to enter 60 signal-strength values, or load a row from the dataset.")

if "signal" not in st.session_state:
	st.session_state.signal = features.iloc[0].tolist()

sample_choice = st.selectbox(
	"Sample signal", ["Custom input"] + [f"Dataset row {index + 1}" for index in range(len(features))]
)
if sample_choice != "Custom input":
	st.session_state.signal = features.iloc[int(sample_choice.split()[-1]) - 1].tolist()

columns = st.columns(4)
for index, column in enumerate(features.columns):
	with columns[index % 4]:
		st.session_state.signal[index] = st.number_input(
			column,
			min_value=0.0,
			max_value=1.0,
			value=float(st.session_state.signal[index]),
			step=0.001,
			format="%.4f",
		)

if st.button("Classify signal", type="primary", use_container_width=True):
	signal = scaler.transform([st.session_state.signal])
	prediction = model.predict(signal)[0]
	probabilities = model.predict_proba(signal)[0]
	probability = probabilities[list(model.classes_).index(prediction)]
	if prediction == "Mine":
		st.error(f"Prediction: {prediction} ({probability:.1%} confidence)")
	else:
		st.success(f"Prediction: {prediction} ({probability:.1%} confidence)")

with st.expander("View evaluation details"):
	st.write("Confusion matrix order: Mine, Rock")
	figure, axis = plt.subplots()
	axis.imshow(matrix, cmap="Blues")
	axis.set_xlabel("Predicted label")
	axis.set_ylabel("Actual label")
	axis.set_xticks([0, 1], ["Mine", "Rock"])
	axis.set_yticks([0, 1], ["Mine", "Rock"])
	for row in range(2):
		for column in range(2):
			axis.text(column, row, matrix[row, column], ha="center", va="center")
	st.pyplot(figure)
