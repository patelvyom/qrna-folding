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
