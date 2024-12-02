import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        self.x = None
        self.hidden = None
        self.r = None
        self.z = None
        self.n = None

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        r_t = self.r_act.forward(np.dot(self.Wrx, x) + self.brx + np.dot(self.Wrh, h_prev_t) + self.brh)
        z_t = self.z_act.forward(np.dot(self.Wzx, x) + self.bzx + np.dot(self.Wzh, h_prev_t) + self.bzh)
        n_t = self.h_act.forward(np.dot(self.Wnx, x) + self.bnx + r_t * (np.dot(self.Wnh, h_prev_t) + self.bnh))
        self.r = r_t
        self.z = z_t
        self.n = n_t
        h_t = (1 - z_t) * n_t + z_t * h_prev_t
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t
        # raise NotImplementedError


    def backward(self, delta):
        """GRU cell backward.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.
        """
        
        dh_t = delta
        
        # Update gate gradients
        dz_t = dh_t * (self.hidden - self.n) * self.z * (1 - self.z)  # Correctly apply sigmoid derivative
        dz = dz_t
        
        # New memory gradients
        dn_t = dh_t * (1 - self.z) * (1 - self.n ** 2)  # Correctly apply tanh derivative
        dn = dn_t
        
        # Reset gate gradients
        dr_t = dn * (np.dot(self.Wnh, self.hidden) + self.bnh) * self.r * (1 - self.r)  # Fixed: use self.r instead of self
        dr = dr_t
        
        # Input gradients
        dx = (np.dot(self.Wrx.T, dr) +
            np.dot(self.Wzx.T, dz) +
            np.dot(self.Wnx.T, dn))
        
        # Previous hidden state gradients
        dh_prev_t = (dh_t * self.z +
                    np.dot(self.Wrh.T, dr) +
                    np.dot(self.Wzh.T, dz) +
                    np.dot(self.Wnh.T, dn * self.r))  # Ensure correct usage of r here

        # Weight and bias gradients
        # Reset gate
        self.dWrx += np.outer(dr, self.x)
        self.dWrh += np.outer(dr, self.hidden)
        self.dbrx += dr
        self.dbrh += dr
        
        # Update gate
        self.dWzx += np.outer(dz, self.x)
        self.dWzh += np.outer(dz, self.hidden)
        self.dbzx += dz
        self.dbzh += dz
        
        # New memory content
        self.dWnx += np.outer(dn, self.x)
        self.dWnh += np.outer(dn * self.r, self.hidden)  # Ensure correct usage of r here as well
        self.dbnx += dn
        self.dbnh += dn * self.r
        
        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)
        
        return dx, dh_prev_t
    
    
