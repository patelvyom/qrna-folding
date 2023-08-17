from abc import ABC, abstractmethod

import pennylane as qml
import preprocessors as preprocessors


class QAOAExperiment(ABC):
    """
    Abstract class for all QAOA experiments.
    """

    name: str = "QAOA Base Experiment"
    preprocessor: preprocessors.BasicPreProcessor = None
    n_qubits: int = 0
    device: qml.device = None

    def __init__(self, preprocessor: preprocessors.BasicPreProcessor, n_qubits: int):
        self.preprocessor = preprocessor
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=self.n_qubits)

    def __repr__(self):
        return f"Experiment(name={self.name}, preprocessor={self.preprocessor}, n_qubits={self.n_qubits})"

    def __str__(self):
        return f"Experiment(name={self.name}, preprocessor={self.preprocessor}, n_qubits={self.n_qubits})"

    @abstractmethod
    def cost_hamiltonian(self):
        pass

    @abstractmethod
    def mixer_hamiltonian(self):
        pass

    @abstractmethod
    def qaoa_layer(self, gamma, alpha):
        pass

    @abstractmethod
    def qaoa_circuit(self, gamma, alpha):
        pass

    @abstractmethod
    def run(self):
        pass


class HamiltonianV1(QAOAExperiment):
    """
    QAOA experiment for Hamiltonian as shown in Jiang et al. (2023) https://arxiv.org/abs/2305.09561.
    """

    name: str = "Hamiltonian V1"

    def __init__(self, preprocessor: preprocessors.BasicPreProcessor, n_qubits: int):
        super().__init__(preprocessor, n_qubits)

    def _penalty(self, stem_1, stem_2, c_p: float = 0.0) -> float:
        """
        Penalty function for two overlapping or pseudoknotted stems as described in Jiang et al. (2023).
        """
        if self.preprocessor.are_stems_overlapping(stem_1, stem_2):
            return -(len(stem_1) + len(stem_2))
        elif self.preprocessor.are_stems_pseudoknotted(stem_1, stem_2):
            return c_p * (len(stem_1) + len(stem_2))
        else:
            return 0

    def cost_hamiltonian(self, eps: float = 6, c_p: float = 0.0):
        stems = self.preprocessor.get_selected_stems
        h_c: qml.Hamiltonian = qml.Hamiltonian([], [])
        n_stems = len(stems)
        for i in range(n_stems):
            stem_i = stems[i]
            k_i = len(stem_i)
            qubit_ops = (qml.Identity(wires=i) - qml.PauliZ(wires=i)) / 2
            coeffs = -2 * k_i + n_stems / (2 * k_i + eps)
            h_c += coeffs * qubit_ops
            for j in range(0, i):
                stem_j = stems[j]
                h = (
                    (qml.Identity(wires=i) - qml.PauliZ(wires=i))
                    @ (qml.Identity(wires=j) - qml.PauliZ(wires=j))
                    / 4
                )
                h_c += h * self._penalty(stem_i, stem_j, c_p)
        return h_c

    def mixer_hamiltonian(self):
        pass

    def qaoa_layer(self, gamma, alpha):
        pass

    def qaoa_circuit(self, gamma, alpha):
        pass

    def run(self):
        pass
