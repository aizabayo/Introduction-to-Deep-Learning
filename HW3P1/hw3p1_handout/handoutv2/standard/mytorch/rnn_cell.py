import numpy as np
from mytorch.nn.activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        """

        # h_t = None  # TODO

        # # return h_t
        # raise NotImplementedError
        self.x = x  # Store inputs and previous hidden state for backward pass
        self.h_prev_t = h_prev_t
        
        # Compute new hidden state
        i_h = np.dot(x, self.W_ih.T)+ self.b_ih
        h_h = np.dot(h_prev_t, self.W_hh.T) + self.b_hh
        self.h_t = self.activation.forward(i_h + h_h)
        
        return self.h_t
        

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        
        # 0) Step backward through the tanh activation function.
        dz = self.activation.backward(delta, h_t)

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size
        self.db_ih += np.sum(dz, axis=0) / batch_size
        self.db_hh += np.sum(dz, axis=0) / batch_size

        # 2) Compute dx, dh_prev_t
        dx = np.dot(dz, self.W_ih)
        dh_prev_t = np.dot(dz, self.W_hh)

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t
        
 