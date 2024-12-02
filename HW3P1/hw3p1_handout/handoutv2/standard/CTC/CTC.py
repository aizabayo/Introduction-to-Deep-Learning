import numpy as np

class CTC(object):
    """CTC class for Connectionist Temporal Classification."""

    def __init__(self, BLANK=0):
        """
        Initialize instance variables.

        Parameters
        ----------
        BLANK : int, optional
            Blank label index. Default is 0.
        """
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """
        Extend target sequence with blank labels.

        Parameters
        ----------
        target : np.array
            Target output sequence.

        Returns
        -------
        extended_symbols : np.array
            Extended target sequence with blanks.
        skip_connect : np.array
            Skip connections.
        """
        extSymbols = [self.BLANK]
        for sy in target:
            extSymbols.append(sy)
            extSymbols.append(self.BLANK)

        N = len(extSymbols)
        skip_connect = np.zeros(N, dtype=int)
        for i in range(3, N):
            skip_connect[i] = 1 if extSymbols[i] != extSymbols[i-2] else skip_connect[i]

        extSymbols = np.array(extSymbols).reshape((N,))
        skipConnect = np.array(skip_connect).reshape((N,))

        return extSymbols, skipConnect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """
        Compute forward probabilities.

        Parameters
        ----------
        logits : np.array
            Predicted log probabilities.
        extended_symbols : np.array
            Extended label sequence with blanks.
        skip_connect : np.array
            Skip connections.

        Returns
        -------
        alpha : np.array
            Forward probabilities.
        """
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros((T, S))

        alpha[0][0] = logits[0, extended_symbols[0]]
        alpha[0][1] = logits[0, extended_symbols[1]]
        alpha[0, 2:S] = 0

        for t in range(1, T):
            alpha[t][0] = alpha[t-1, 0] * logits[t, extended_symbols[0]]
            for sym in range(1, S):
                alpha[t, sym] = alpha[t-1, sym] + alpha[t-1, sym-1]
                if sym > 1 and skip_connect[sym] == 1:
                    alpha[t, sym] += alpha[t-1, sym-2]
                alpha[t, sym] *= logits[t, extended_symbols[sym]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """
        Compute backward probabilities.

        Parameters
        ----------
        logits : np.array
            Predicted log probabilities.
        extended_symbols : np.array
            Extended label sequence with blanks.
        skip_connect : np.array
            Skip connections.

        Returns
        -------
        beta : np.array
            Backward probabilities.
        """
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros((T, S))

        beta[T-1, S-1] = logits[T-1, extended_symbols[S-1]]
        beta[T-1, S-2] = logits[T-1, extended_symbols[S-2]]

        for t in range(T-2, -1, -1):
            beta[t, S-1] = beta[t+1, S-1] * logits[t, extended_symbols[S-1]]
            for i in range(S-2, -1, -1):
                current_logit = logits[t, extended_symbols[i]]
                beta[t, i] = beta[t+1, i] + beta[t+1, i+1]
                if i < S - 3 and skip_connect[i + 2] == 1:
                    beta[t, i] += beta[t+1, i+2]
                beta[t, i] *= current_logit

        for t in range(T - 1, -1, -1):
            for i in range(S - 1, -1, -1):
                beta[t, i] /= logits[t, extended_symbols[i]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """
        Compute posterior probabilities.

        Parameters
        ----------
        alpha : np.array
            Forward probabilities.
        beta : np.array
            Backward probabilities.

        Returns
        -------
        gamma : np.array
            Posterior probabilities.
        """
        T, S = alpha.shape
        gamma = np.zeros((T, S))
        sum_gamma = np.zeros(T)

        for t in range(T):
            sum_gamma[t] = 0
            for i in range(S):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                sum_gamma[t] += gamma[t, i]
            for n in range(S):
                gamma[t, n] /= sum_gamma[t]

        return gamma


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """
        Initialize instance variables.

        Parameters
        ----------
        BLANK : int, optional
            Blank label index. Default is 0.
        """
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    def __call__(self, logits, target, input_lengths, target_lengths):
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """
        Compute the CTC Loss.

        Parameters
        ----------
        logits : np.array
            Log probabilities from the RNN/GRU.
        target : np.array
            Target sequences.
        input_lengths : np.array
            Lengths of the inputs.
        target_lengths : np.array
            Lengths of the target sequences.

        Returns
        -------
        loss : float
            Average divergence between the posterior probability and the target.
        """
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            target = self.target[batch_itr][:self.target_lengths[batch_itr]]
            logit = self.logits[:self.input_lengths[batch_itr], batch_itr]
            extended, skip_connect = self.ctc.extend_target_with_blank(target)
            self.extended_symbols.append(extended)

            alpha = self.ctc.get_forward_probs(logit, extended, skip_connect)
            beta = self.ctc.get_backward_probs(logit, extended, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            self.gammas.append(gamma)

            for i in range(gamma.shape[0]):
                for j in range(gamma.shape[1]):
                   total_loss[batch_itr] += -gamma[i][j] * np.log(logit[i, extended[j]])
        
        total_loss = np.sum(total_loss) / B    
        return total_loss

    def backward(self):
        """
        Compute the gradients with respect to the inputs.

        Returns
        -------
        dY : np.array
            Derivative of divergence with respect to the input symbols at each time.
        """
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            target_trunc = self.target[batch_itr, :self.target_lengths[batch_itr]]
            logits_trunc = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_trunc)

            alpha = self.ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            for t in range(self.input_lengths[batch_itr]):
                for s, symbol in enumerate(extended_symbols):
                    dY[t, batch_itr, symbol] -= gamma[t, s] / (logits_trunc[t, symbol] + 1e-10)  # Add epsilon

        return dY