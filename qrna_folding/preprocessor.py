import numpy as np


class BasicPreProcessor(object):
    """Basic preprocessor for RNA sequences with some commonly used helper functions."""

    rna_sequence: str = ""
    adj_matrix: np.ndarray = None
    rna_length: int = 0
    potential_stems: list[tuple[int, int, int]] = []

    def __init__(self, rna: str, **kwargs):
        self.rna_sequence = rna
        self.rna_length = len(rna)
        self.adj_matrix = np.zeros((self.rna_length, self.rna_length), dtype=int)

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

    def get_adjacency_matrix(self):
        return self.adj_matrix

    def are_stems_overlapping(self, stem1: tuple, stem2: tuple) -> bool:
        """Check if two stems are overlapping. Refer: https://github.com/JuanGiraldo0212/Qhack-qnyble/blob/main/qrna_folding_qaoa.ipynb
        TODO: Give explicit example.
        """
        stem1_span1 = set(range(stem1[0], stem1[0] + stem1[2]))
        stem2_span1 = set(range(stem2[0], stem2[0] + stem2[2]))
        stem1_span2 = set(range(stem1[1] - stem1[2] + 1, stem1[1] + 1))
        stem2_span2 = set(range(stem2[1] - stem2[2] + 1, stem2[1] + 1))

        if (
            (len(stem1_span1 & stem2_span1) > 0)
            or (len(stem1_span2 & stem2_span2) > 0)
            or (len(stem1_span1 & stem2_span2) > 0)
            or (len(stem1_span2 & stem2_span1) > 0)
        ):
            return True
        return False

    def are_stems_pseudoknotted(self, stem1: tuple, stem2: tuple) -> bool:
        """Check if two stems are pseudoknotted. Refer: Fox et al. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010032"""

        assert not self.are_stems_overlapping(
            stem1, stem2
        ), "Stems are overlapping. Cannot check for pseudoknots."

        i_a: int = stem1[0]
        j_a: int = stem1[1]
        i_b: int = stem2[0]
        j_b: int = stem2[1]

        if (i_a < i_b < j_a < j_b) or (i_b < i_a < j_b < j_a):
            return True
        return False

    def compute_potential_stems(self, **kwargs):
        raise NotImplementedError

    def get_potential_stems(self):
        return self.potential_stems

    def process(self, **kwargs):
        raise NotImplementedError