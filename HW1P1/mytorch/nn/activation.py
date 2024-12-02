import numpy as np
import scipy
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        A = 1 / (1 + np.exp(-Z))
        self.A = A
        
        return self.A 
    
    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)   # elementwise multiplication
        dLdZ = dLdA * dAdZ
        
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    
    def forward(self, Z):
        self.A = np.tanh(Z)
        
        return self.A
    
    
    def backward(self, dLdA):
        
        dAdZ = 1 - np.square(self.A)
        dLdZ = dLdA * dAdZ
        
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        
        self.A = np.maximum(0, Z)
        
        return self.A
    
    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1, 0)
        dLdZ = dLdA * dAdZ
        
        return dLdZ

        
        

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU
    """
    
    def forward(self, Z):
        
        self.Z = Z
        self.var = 0.5 *(1 + erf(Z / np.sqrt(2)))
        self.A = Z * self.var
        return self.A
    
    def backward(self, dLdA):
        
        
        fi = (self.Z / np.sqrt(2 * np.pi)) * np.exp(-np.square(self.Z) / 2)
        dAdZ = self.var + fi
        dLdZ = dLdA * dAdZ
        
        
        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        Zm = np.exp(Z - np.max(Z, axis = 1, keepdims=True))
        Zk = np.sum(Zm, axis = 1, keepdims= True)

        self.A = Zm / Zk # TODO
        
        

        #return NotImplementedError
        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N, C= dLdA.shape # TODO
        # C =  # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ =  np.zeros_like(dLdA)   # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C)) # TODO
            

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m,n] = self.A[i, m] * (1 - self.A[i, m]) # TODO
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = np.dot(dLdA[i, :], J)
        return dLdZ


