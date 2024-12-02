import numpy as np

class GreedySearchDecoder(object):
    """Greedy Search Decoder class."""

    def __init__(self, symbol_set):
        """
        Initialize instance variables.

        Parameters
        ----------
        symbol_set : list[str]
            All the symbols (the vocabulary without blank).
        """
        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """
        Perform greedy search decoding.

        Parameters
        ----------
        y_probs : np.array
            Log probabilities of shape (len(symbols) + 1, seq_length, batch_size).

        Returns
        -------
        decoded_path : str
            Compressed symbol sequence without blanks or repeated symbols.
        path_prob : float
            Forward probability of the greedy path.
        """
        decoded_path = []
        blank = 0
        path_prob = 1

        merged_length, seq_length, batch_size = y_probs.shape
        for batch in range(batch_size):
            for seq in range(seq_length):
                path_prob *= np.max(y_probs[:, seq, batch])
                idx = np.argmax(y_probs[:, seq, batch])
                if idx != 0:
                    if blank:
                        n = self.symbol_set[idx - 1]
                        decoded_path.append(n)
                        blank = 0
                    else:
                        if seq == 0 or (decoded_path[-1] != self.symbol_set[idx - 1]):
                            decoded_path.append(self.symbol_set[idx - 1])
                else:
                    blank = 1

        decoded_path = self.clean_path(decoded_path)

        return decoded_path, path_prob

    def clean_path(self, path):

        cleaned_path = []
        prev_symbol = None
        for symbol in path:
            if symbol != prev_symbol:
                cleaned_path.append(symbol)
                prev_symbol = symbol
        return ''.join(cleaned_path)


class BeamSearchDecoder(object):
    """Beam Search Decoder class."""

    def __init__(self, symbol_set, beam_width):
        """
        Initialize instance variables.

        Parameters
        ----------
        symbol_set : list[str]
            All the symbols (the vocabulary without blank).
        beam_width : int
            Beam width for selecting top-k hypotheses for expansion.
        """
        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        Perform beam search decoding.

        Parameters
        ----------
        y_probs : np.array
            Log probabilities of shape (len(symbols) + 1, seq_length, batch_size).

        Returns
        -------
        forward_path : str
            The symbol sequence with the best path score (forward probability).
        merged_path_scores : dict
            All the final merged paths with their scores.
        """
        beam_search = BeamSearchClass(self.symbol_set, y_probs, self.beam_width)
        best_path, merged_path_scores = beam_search()
        return best_path, merged_path_scores


class BeamSearchClass:
    """Beam Search Class for decoding."""

    def __init__(self, symbol_set, y_probs, beam_width):
        """
        Initialize instance variables.

        Parameters
        ----------
        symbol_set : list[str]
            All the symbols (the vocabulary without blank).
        y_probs : np.array
            Log probabilities of shape (len(symbols) + 1, seq_length, batch_size).
        beam_width : int
            Beam width for selecting top-k hypotheses for expansion.
        """
        self.symbols = symbol_set
        self.y_probs = y_probs
        self.k = beam_width

        self.paths_blank = ['']
        self.paths_blank_score = {'': y_probs[0, 0, 0]}

        self.paths_symbol = [c for c in self.symbols]
        self.paths_symbol_score = {c: y_probs[i + 1, 0, 0] for i, c in enumerate(symbol_set)}

    def __call__(self):
        for t in range(1, self.y_probs.shape[1]):
            self.prune()
            updated_paths_symbol, updated_paths_symbol_score = self.extend_with_symbol(t)
            updated_paths_blank, updated_paths_blank_score = self.extend_with_blank(t)
            self.paths_blank = updated_paths_blank
            self.paths_symbol = updated_paths_symbol
            self.paths_blank_score = updated_paths_blank_score
            self.paths_symbol_score = updated_paths_symbol_score

        return self.merge()

    def extend_with_symbol(self, t):
        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        for path in self.paths_blank:
            for i, c in enumerate(self.symbols):
                new_path = path + c
                updated_paths_symbol.append(new_path)
                updated_paths_symbol_score[new_path] = self.paths_blank_score[path] * self.y_probs[i + 1, t, 0]

        for path in self.paths_symbol:
            for i, c in enumerate(self.symbols):
                new_path = path if c == path[-1] else path + c
                if new_path in updated_paths_symbol_score:
                    updated_paths_symbol_score[new_path] += self.paths_symbol_score[path] * self.y_probs[i + 1, t, 0]
                else:
                    updated_paths_symbol_score[new_path] = self.paths_symbol_score[path] * self.y_probs[i + 1, t, 0]
                    updated_paths_symbol.append(new_path)

        return updated_paths_symbol, updated_paths_symbol_score

    def extend_with_blank(self, t):
        updated_paths_blank = []
        updated_paths_blank_score = {}

        for path in self.paths_blank:
            updated_paths_blank.append(path)
            updated_paths_blank_score[path] = self.paths_blank_score[path] * self.y_probs[0, t, 0]

        for path in self.paths_symbol:
            if path in updated_paths_blank:
                updated_paths_blank_score[path] += self.paths_symbol_score[path] * self.y_probs[0, t, 0]
            else:
                updated_paths_blank_score[path] = self.paths_symbol_score[path] * self.y_probs[0, t, 0]
                updated_paths_blank.append(path)

        return updated_paths_blank, updated_paths_blank_score

    def prune(self):
        updated_paths_blank = []
        updated_paths_blank_score = {}

        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        scores = list(self.paths_blank_score.values()) + list(self.paths_symbol_score.values())
        scores.sort()

        cutoff = scores[-1] if len(scores) < self.k else scores[-self.k]

        for path in self.paths_blank:
            if self.paths_blank_score[path] >= cutoff:
                updated_paths_blank.append(path)
                updated_paths_blank_score[path] = self.paths_blank_score[path]

        for path in self.paths_symbol:
            if self.paths_symbol_score[path] >= cutoff:
                updated_paths_symbol.append(path)
                updated_paths_symbol_score[path] = self.paths_symbol_score[path]

        self.paths_symbol_score = updated_paths_symbol_score
        self.paths_symbol = updated_paths_symbol
        self.paths_blank_score = updated_paths_blank_score
        self.paths_blank = updated_paths_blank

    def merge(self):
        paths = self.paths_blank
        scores = self.paths_blank_score

        for path in self.paths_symbol:
            if path in paths:
                scores[path] += self.paths_symbol_score[path]
            else:
                paths.append(path)
                scores[path] = self.paths_symbol_score[path]

        max_path = max(scores, key=scores.get)
        return max_path, scores