import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lstm_model import LSTM

X = np.load("data/X.npy")
y = np.load("data/y.npy")

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

val_split = int(0.85 * len(X))
X_train, X_val = X[:val_split], X[val_split:]
y_train, y_val = y[:val_split], y[val_split:]

print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
print("X_train shape:", X_train.shape)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def cross_entropy(pred, true):
    return -np.sum(true * np.log(pred + 1e-9))

input_size = X_train.shape[2]
hidden_size = 64
output_size = 5

model = LSTM(input_size, hidden_size, output_size)

epochs = 15
learning_rate = 0.002

train_loss_history = []
val_loss_history = []

beta1, beta2 = 0.9, 0.999
eps = 1e-8
m = {param: np.zeros_like(getattr(model, param)) for param in ['Wf', 'Wi', 'Wc', 'Wo', 'Uf', 'Ui', 'Uc', 'Uo', 'Wy', 'bf', 'bi', 'bc', 'bo', 'by']}
v = {param: np.zeros_like(getattr(model, param)) for param in ['Wf', 'Wi', 'Wc', 'Wo', 'Uf', 'Ui', 'Uc', 'Uo', 'Wy', 'bf', 'bi', 'bc', 'bo', 'by']}
t_step = 0

for epoch in range(epochs):
    total_train_loss = 0
    
    for i in range(len(X_train)):
        x_sample = X_train[i] 
        y_true = one_hot(y_train[i], output_size).reshape(-1, 1)
        
        T = x_sample.shape[0]
        h_states = { -1: np.zeros((model.hidden_sz, 1)) }
        c_states = { -1: np.zeros((model.hidden_sz, 1)) }
        
        f_gates, i_gates, c_hats, o_gates = {}, {}, {}, {}
        
        h = h_states[-1]
        c = c_states[-1]
        
        for t in range(T):
            x_t = x_sample[t].reshape(-1, 1)
            f_gates[t] = 1 / (1 + np.exp(-np.clip(model.Wf @ x_t + model.Uf @ h + model.bf, -500, 500)))
            i_gates[t] = 1 / (1 + np.exp(-np.clip(model.Wi @ x_t + model.Ui @ h + model.bi, -500, 500)))
            c_hats[t] = np.tanh(model.Wc @ x_t + model.Uc @ h + model.bc)
            c = f_gates[t] * c + i_gates[t] * c_hats[t]
            c_states[t] = c
            o_gates[t] = 1 / (1 + np.exp(-np.clip(model.Wo @ x_t + model.Uo @ h + model.bo, -500, 500)))
            h = o_gates[t] * np.tanh(c)
            h_states[t] = h
            
        exp_x = np.exp(model.Wy @ h + model.by - np.max(model.Wy @ h + model.by, axis=0, keepdims=True))
        y_pred = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
        loss = float(cross_entropy(y_pred, y_true))
        total_train_loss += loss
        
        grads = {param: np.zeros_like(getattr(model, param)) for param in m.keys()}
        
        error = y_pred - y_true
        grads['Wy'] = error @ h_states[T-1].T
        grads['by'] = error
        
        dh_next = np.zeros((model.hidden_sz, 1))
        dc_next = np.zeros((model.hidden_sz, 1))
        
        for t in reversed(range(T)):
            x_t = x_sample[t].reshape(-1, 1)
            c = c_states[t]
            c_prev = c_states[t-1]
            h_prev = h_states[t-1]
            
            if t == T - 1:
                dh = model.Wy.T @ error + dh_next
            else:
                dh = dh_next
                
            dc = dh * o_gates[t] * (1 - np.tanh(c)**2) + dc_next
            
            df = dc * c_prev * f_gates[t] * (1 - f_gates[t])
            di = dc * c_hats[t] * i_gates[t] * (1 - i_gates[t])
            dc_hat = dc * i_gates[t] * (1 - c_hats[t]**2)
            do = dh * np.tanh(c) * o_gates[t] * (1 - o_gates[t])
            
            grads['Wf'] += df @ x_t.T
            grads['Uf'] += df @ h_prev.T
            grads['bf'] += df
            
            grads['Wi'] += di @ x_t.T
            grads['Ui'] += di @ h_prev.T
            grads['bi'] += di
            
            grads['Wc'] += dc_hat @ x_t.T
            grads['Uc'] += dc_hat @ h_prev.T
            grads['bc'] += dc_hat
            
            grads['Wo'] += do @ x_t.T
            grads['Uo'] += do @ h_prev.T
            grads['bo'] += do
            
            dh_next = model.Uf.T @ df + model.Ui.T @ di + model.Uc.T @ dc_hat + model.Uo.T @ do
            dc_next = f_gates[t] * dc

        t_step += 1
        for param in grads.keys():
            np.clip(grads[param], -1.0, 1.0, out=grads[param])
            m[param] = beta1 * m[param] + (1 - beta1) * grads[param]
            v[param] = beta2 * v[param] + (1 - beta2) * (grads[param] ** 2)
            m_hat = m[param] / (1 - beta1 ** t_step)
            v_hat = v[param] / (1 - beta2 ** t_step)
            setattr(model, param, getattr(model, param) - learning_rate * m_hat / (np.sqrt(v_hat) + eps))

    avg_train_loss = total_train_loss / len(X_train)
    train_loss_history.append(avg_train_loss)
    
    total_val_loss = 0
    for i in range(len(X_val)):
        y_v_true = one_hot(y_val[i], output_size).reshape(-1, 1)
        y_v_pred = model.forward(X_val[i])
        total_val_loss += float(cross_entropy(y_v_pred, y_v_true))
        
    avg_val_loss = total_val_loss / len(X_val)
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title("Neural Model Loss Convergence")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

os.makedirs("graphs", exist_ok=True)
save_path = os.path.join("graphs", "loss_curve.png")
plt.savefig(save_path)
plt.close()

print(f"Loss graph saved at: {save_path}")

with open("lstm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model weights exported successfully inside 'lstm_model.pkl'!")