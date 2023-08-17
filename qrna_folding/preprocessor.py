import numpy as np


class BasicPreProcessor(object):
    """Basic preprocessor for RNA sequences with some common helper functions."""

    rna_sequence: str = ""
    adj_matrix: np.ndarray = None
    rna_length: int = 0

    def __init__(self, rna: str, **kwargs):
        self.rna_sequence = rna
        self.rna_length = len(rna)
        self.adj_matrix = np.zeros((self.rna_length, self.rna_length))
        self.kwargs = kwargs

    def __repr__(self):
        return f"BasePreProcessor(rna_sequence={self.rna_sequence}) \n adj_matrix={self.adj_matrix}"

    def __str__(self):
        return f"BasePreProcessor(rna_sequence={self.rna_sequence})"

    def compute_adjacency_matrix(self, **kwargs):
        valid = {
            "AU": 2,
            "UA": 2,
            "GC": 3,
            "CG": 3,
            "GU": 2,
            "UG": 2,
        }  # Values represent no. of H-bonds
        for i in range(self.rna_length):
            for j in range(i + 1, self.rna_length):
                if x := valid.get(self.rna_sequence[i] + self.rna_sequence[j]):
                    self.adj_matrix[i, j] = self.adj_matrix[j, i] = x

    def process(self, **kwargs):
        raise NotImplementedError
