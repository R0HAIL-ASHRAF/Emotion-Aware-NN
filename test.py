import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("lstm_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")

X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

emotion_labels = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "fear",
}

emotion_names = ["angry", "happy", "sad", "neutral", "fear"]
num_classes = len(emotion_names)

all_predictions = []
all_actuals = []
correct = 0

print("\nRunning predictions...\n")

for i in range(len(X)):
    sample = X[i]
    prediction = model.predict(sample)
    actual = y[i]

    all_predictions.append(prediction)
    all_actuals.append(actual)

    if prediction == actual:
        correct += 1

    print("=" * 50)
    print(f"Sample {i+1}")
    print("Predicted:", emotion_labels[prediction])
    print("Actual   :", emotion_labels[actual])

accuracy = (correct / len(X)) * 100

conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for actual, predicted in zip(all_actuals, all_predictions):
    conf_matrix[actual][predicted] += 1

# From-scratch calculation of Precision, Recall, and F1-Score per class
precision_list = []
recall_list = []
f1_list = []

for i in range(num_classes):
    tp = conf_matrix[i, i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

print("\n" + "=" * 50)
print(f"FINAL ACCURACY: {accuracy:.2f}%")
print("=" * 50)
print(f"{'Class':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
print("-" * 50)
for i in range(num_classes):
    print(f"{emotion_names[i]:<12}{precision_list[i]:<12.4f}{recall_list[i]:<12.4f}{f1_list[i]:<12.4f}")
print("=" * 50)

os.makedirs("graphs", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')
plt.title("Confusion Matrix Evaluation")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(range(num_classes), emotion_names, rotation=45)
plt.yticks(range(num_classes), emotion_names)

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png")
plt.close()

print("Confusion matrix saved to graphs/confusion_matrix.png")

results = pd.DataFrame({
    "Actual": [emotion_labels[x] for x in all_actuals],
    "Predicted": [emotion_labels[x] for x in all_predictions],
})
results.to_csv("graphs/predictions.csv", index=False)
print("Predictions saved to graphs/predictions.csv")

print("\n" + "=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)