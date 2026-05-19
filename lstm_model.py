import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class LSTM:
    def __init__(self, input_sz, hidden_sz, output_sz):

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        limit = np.sqrt(1 / input_sz)

        self.Wf = np.random.randn(hidden_sz, input_sz) * limit
        self.bf = np.zeros((hidden_sz, 1))

        self.Wi = np.random.randn(hidden_sz, input_sz) * limit
        self.bi = np.zeros((hidden_sz, 1))

        self.Wc = np.random.randn(hidden_sz, input_sz) * limit
        self.bc = np.zeros((hidden_sz, 1))

        self.Wo = np.random.randn(hidden_sz, input_sz) * limit
        self.bo = np.zeros((hidden_sz, 1))

        self.Wy = np.random.randn(output_sz, hidden_sz) * limit
        self.by = np.zeros((output_sz, 1))

    def forward(self, X):

        T = X.shape[1]

        self.h = np.zeros((self.hidden_sz, 1))
        self.c = np.zeros((self.hidden_sz, 1))

        for t in range(T):

            x_t = X[t].reshape(-1, 1)

            f_t = sigmoid(self.Wf @ x_t + self.bf)

            i_t = sigmoid(self.Wi @ x_t + self.bi)

            c_hat = tanh(self.Wc @ x_t + self.bc)

            self.c = f_t * self.c + i_t * c_hat

            o_t = sigmoid(self.Wo @ x_t + self.bo)

            self.h = o_t * tanh(self.c)

        y_pred = softmax(self.Wy @ self.h + self.by)

        return y_pred

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred)
