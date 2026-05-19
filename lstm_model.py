import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class LSTM:
    def __init__(self, input_sz, hidden_sz, output_sz):
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz

        limit = np.sqrt(1 / (input_sz + hidden_sz))

        # Weights for Inputs (x_t)
        self.Wf = np.random.randn(hidden_sz, input_sz) * limit
        self.Wi = np.random.randn(hidden_sz, input_sz) * limit
        self.Wc = np.random.randn(hidden_sz, input_sz) * limit
        self.Wo = np.random.randn(hidden_sz, input_sz) * limit

        # Weights for Recurrent Hidden States (h_t-1)
        self.Uf = np.random.randn(hidden_sz, hidden_sz) * limit
        self.Ui = np.random.randn(hidden_sz, hidden_sz) * limit
        self.Uc = np.random.randn(hidden_sz, hidden_sz) * limit
        self.Uo = np.random.randn(hidden_sz, hidden_sz) * limit

        # Biases
        self.bf = np.zeros((hidden_sz, 1))
        self.bi = np.zeros((hidden_sz, 1))
        self.bc = np.zeros((hidden_sz, 1))
        self.bo = np.zeros((hidden_sz, 1))

        # Output Layer Weights
        self.Wy = np.random.randn(output_sz, hidden_sz) * np.sqrt(1 / hidden_sz)
        self.by = np.zeros((output_sz, 1))

    def forward(self, X):
        # Expected X shape: (time_steps, input_sz)
        T = X.shape[0]

        h = np.zeros((self.hidden_sz, 1))
        c = np.zeros((self.hidden_sz, 1))

        for t in range(T):
            x_t = X[t].reshape(-1, 1)

            f_t = sigmoid(self.Wf @ x_t + self.Uf @ h + self.bf)
            i_t = sigmoid(self.Wi @ x_t + self.Ui @ h + self.bi)
            c_hat = tanh(self.Wc @ x_t + self.Uc @ h + self.bc)
            
            c = f_t * c + i_t * c_hat
            
            o_t = sigmoid(self.Wo @ x_t + self.Uo @ h + self.bo)
            h = o_t * tanh(c)

        y_pred = softmax(self.Wy @ h + self.by)
        return y_pred

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred)