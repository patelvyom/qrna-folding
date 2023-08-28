import random
from abc import ABC, abstractmethod

import numpy as np


def are_stems_overlapping(stem1: tuple, stem2: tuple) -> bool:
    """Check if two stems are overlapping. Refer:
    https://github.com/JuanGiraldo0212/Qhack-qnyble/blob/main/qrna_folding_qaoa.ipynb
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


def are_stems_pseudoknotted(stem1: tuple, stem2: tuple) -> bool:
    """Check if two stems are pseudoknotted. Refer: Fox et al.
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010032
    """

    assert not are_stems_overlapping(
        stem1, stem2
    ), "Stems are overlapping. Cannot check for pseudoknots."

    i_a: int = stem1[0]
    j_a: int = stem1[1]
    i_b: int = stem2[0]
    j_b: int = stem2[1]

    if (i_a < i_b < j_a < j_b) or (i_b < i_a < j_b < j_a):
        return True
    return False


class BasicPreProcessor(ABC):
    """Basic preprocessor for RNA sequences with some commonly used helper functions."""

    rna_sequence: str = ""
    _adj_matrix: np.ndarray = None
    rna_length: int = 0
    _potential_stems: list[tuple[int, int, int]] = []
    _selected_stems: list[tuple[int, int, int]] = []
    valid_bonds = {
        "AU": 2,
        "UA": 2,
        "GC": 3,
        "CG": 3,
        "GU": 2,
        "UG": 2,
    }  # Values represent no. of H-bonds
    _largest_stem_length: int = 0

    def __init__(self, rna: str, **kwargs):
        self.rna_sequence = rna
        self.rna_length = len(rna)
        self._adj_matrix = np.zeros((self.rna_length, self.rna_length), dtype=int)

    def __repr__(self):
        return f"BasePreProcessor(rna_sequence={self.rna_sequence}) \n adj_matrix={self._adj_matrix}"

    def __str__(self):
        return f"BasePreProcessor(rna_sequence={self.rna_sequence})"

    def _compute_adjacency_matrix(self, **kwargs):
        print("[preprocessors.py] Computing adjacency matrix...")
        for i in range(self.rna_length):
            for j in range(i + 1, self.rna_length):
                if x := self.valid_bonds.get(
                    self.rna_sequence[i] + self.rna_sequence[j]
                ):
                    self._adj_matrix[i, j] = self._adj_matrix[j, i] = x

    @property
    def adjacency_matrix(self):
        return self._adj_matrix

    @abstractmethod
    def compute_potential_stems(self):
        pass

    @property
    def potential_stems(self):
        return self._potential_stems

    @property
    def largest_stem_length(self):
        return self._largest_stem_length

    @property
    def n_potential_stems(self):
        return len(self._potential_stems)

    @property
    def selected_stems(self):
        return self._selected_stems

    def select_n_stems(self, n: int, method: str = "random"):
        """Basic methods to select the top n stems."""
        if not self._potential_stems:
            print(
                "[preprocessors.py] Warning: Potential stems not computed. Computing now. Calling process() will "
                "overwrite this."
            )
            self.compute_potential_stems()
        assert n <= len(
            self._potential_stems
        ), f"n={n} is greater than the number of potential stems={len(self._potential_stems)}"
        if method == "random":
            self._selected_stems = random.choices(self._potential_stems, k=n)
        elif method == "longest":
            self._selected_stems = sorted(
                self._potential_stems, key=lambda x: x[2], reverse=True
            )[:n]
        else:
            raise NotImplementedError

    @abstractmethod
    def process(self, **kwargs):
        pass


class NormalStemLengthPreProcessor(BasicPreProcessor):
    """
    Preprocessor where stem length model is the basic (BP count) model: length of a potential stem is the number of
    base pairs it can form.
    """

    def __init__(self, rna: str, **kwargs):
        super().__init__(rna, **kwargs)

    def compute_potential_stems(self, min_stem_length: int = 3):
        print("[preprocessors.py] Computing potential stems...")
        matrix = np.triu(
            self._adj_matrix
        )  # Upper triangular matrix because of symmetry
        for i in range(self.rna_length):
            for j in range(i + 1, self.rna_length):
                if matrix[i, j] > 0:
                    i_temp, j_temp = i, j
                    stem_length = 0
                    while matrix[i_temp, j_temp] > 0:
                        stem_length += 1
                        i_temp += 1
                        j_temp -= 1
                        if stem_length >= min_stem_length:
                            stem = (i + 1, j + 1, stem_length)
                            self._potential_stems.append(stem)
                    if stem_length > self._largest_stem_length:
                        self._largest_stem_length = stem_length

    def process(self, **kwargs):
        print("[preprocessors.py] Processing RNA sequence...")
        self._compute_adjacency_matrix()
        self.compute_potential_stems(min_stem_length=kwargs["min_stem_length"])
        print(f"[preprocessors.py] Found {self.n_potential_stems} potential stems.")


class HBondCountPreProcessor(BasicPreProcessor):
    """
    Preprocessor where stem length model used is the H-bond count model: length of a potential stem is the number of
    H-bonds it can form.
    """

    def __init__(self, rna: str, **kwargs):
        super().__init__(rna, **kwargs)

    def compute_potential_stems(self, min_stem_length: int = 3):
        matrix = np.triu(
            self._adj_matrix
        )  # Upper triangular matrix because of symmetry
        for i in range(self.rna_length):
            for j in range(i + 1, self.rna_length):
                if matrix[i, j] > 0:
                    i_temp, j_temp = i, j
                    stem_length, h_bonds = 0, 0
                    while matrix[i_temp, j_temp] > 0:
                        stem_length += 1
                        h_bonds += matrix[i_temp, j_temp]
                        i_temp += 1
                        j_temp -= 1
                        if stem_length >= min_stem_length:
                            stem = (i + 1, j + 1, h_bonds)
                            self._potential_stems.append(stem)
                    if h_bonds > self._largest_stem_length:
                        self._largest_stem_length = h_bonds

    def process(self, **kwargs):
        print("[preprocessors.py] Processing RNA sequence...")
        self._compute_adjacency_matrix()
        self.compute_potential_stems(min_stem_length=kwargs["min_stem_length"])
        print(f"[preprocessors.py] Found {self.n_potential_stems} potential stems.")
