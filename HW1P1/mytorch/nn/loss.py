import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape  # TODO
        #self.C =  # TODO
        se = np.square(A - Y)  # TODO
        sse = np.sum(se)  # TODO
        mse = sse / (self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = 2 * ((self.A - self.Y)/ (self.N * self.C))

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N, C =  self.A.shape # TODO
        # TODO

        #Ones_C =    # TODO
        #Ones_N =   # TODO
        
        ez = np.exp(A - np.max(A, axis = 1, keepdims= True))
        sz = np.sum(ez, axis = 1, keepdims= True)

        self.softmax = ez / sz   # TODO
        crossentropy = -(Y *np.log(self.softmax))  # TODO
        # crossentropy = -(Y *np.log(self.softmax + 1e-6))
        sum_crossentropy = np.sum(crossentropy)  # TODO
        L = sum_crossentropy / N



        return L

    def backward(self):
        N = self.A.shape[0]

        dLdA = (self.softmax - self.Y) / N  # TODO

        return dLdA
