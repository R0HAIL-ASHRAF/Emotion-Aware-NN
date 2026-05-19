import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os


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
    5: "surprise"
}


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

print("\n" + "=" * 50)
print(f"FINAL ACCURACY: {accuracy:.2f}%")
print("=" * 50)


conf_matrix = np.zeros((6, 6), dtype=int)

for actual, predicted in zip(all_actuals, all_predictions):
    conf_matrix[actual][predicted] += 1


os.makedirs("graphs", exist_ok=True)

plt.figure(figsize=(8, 6))

plt.imshow(conf_matrix)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

emotion_names = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "fear",
    "surprise"
]

plt.xticks(range(6), emotion_names, rotation=45)
plt.yticks(range(6), emotion_names)

# Annotate values
for i in range(6):
    for j in range(6):
        plt.text(j, i, conf_matrix[i, j],
                 ha="center", va="center")

plt.savefig("graphs/confusion_matrix.png")
plt.close()

print("Confusion matrix saved to graphs/confusion_matrix.png")


results = pd.DataFrame({
    "Actual": [emotion_labels[x] for x in all_actuals],
    "Predicted": [emotion_labels[x] for x in all_predictions]
})

results.to_csv("graphs/predictions.csv", index=False)

print("Predictions saved to graphs/predictions.csv")


print("\n" + "=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)
print(f"Accuracy: {accuracy:.2f}%")
print("Graphs saved in /graphs directory")