import numpy as np
import matplotlib.pyplot as plt
import pickle
from lstm_model import LSTM
import os

X = np.load("data/X.npy")
y = np.load("data/y.npy")

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

print("X shape:", X.shape)
print("y shape:", y.shape)


def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def cross_entropy(pred, true):
    return -np.sum(true * np.log(pred + 1e-9))


input_size = X.shape[2]
hidden_size = 64
output_size = 6

model = LSTM(input_size, hidden_size, output_size)

epochs = 10
learning_rate = 0.001

loss_history = []


for epoch in range(epochs):

    total_loss = 0

    for i in range(len(X)):

        x_sample = X[i].reshape(1, X.shape[1], X.shape[2])
        y_true = one_hot(y[i], output_size).reshape(-1, 1)

        y_pred = model.forward(x_sample[0])

        loss = float(cross_entropy(y_pred, y_true))
        total_loss += loss

        error = y_pred - y_true

        model.Wy -= learning_rate * (error @ model.h.T)
        model.by -= learning_rate * error

    avg_loss = total_loss / len(X)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

os.makedirs("graphs", exist_ok=True)
save_path = os.path.join("graphs", "loss_curve.png")
plt.savefig(save_path)

plt.close()

print(f"Loss graph saved at: {save_path}")


with open("lstm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
