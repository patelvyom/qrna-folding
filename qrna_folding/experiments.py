from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pennylane as qml
import preprocessors as preprocessors
import utils
from pennylane import numpy as np
from pennylane import qaoa

N_SHOTS = 100


class QAOAExperiment(ABC):
    """
    Abstract class for all QAOA experiments.
    """

    name: str = "QAOA Base Experiment"
    preprocessor: preprocessors.BasicPreProcessor = None
    n_qubits: int = 0
    dev: qml.device = None
    optimizer = None
    optimizer_steps = 100

    def __init__(self, preprocessor: preprocessors.BasicPreProcessor, **kwargs):
        self.preprocessor = preprocessor
        self.n_qubits = len(preprocessor.selected_stems)
        self.dev = qml.device(
            kwargs.get("device", "default.qubit"),
            wires=self.n_qubits,
            shots=kwargs.get("shots", N_SHOTS),
        )
        self.optimizer = qml.AdamOptimizer(stepsize=0.1)

    def __repr__(self):
        return f"Experiment(name={self.name}, preprocessor={self.preprocessor}, n_qubits={self.n_qubits})"

    def __str__(self):
        return f"Experiment(name={self.name}, preprocessor={self.preprocessor}, n_qubits={self.n_qubits})"

    @abstractmethod
    def _compute_cost_h(self):
        pass

    @abstractmethod
    def _compute_mixer_h(self):
        pass

    @abstractmethod
    def initial_layer(self):
        pass

    @abstractmethod
    def qaoa_layer(self, params):
        pass

    @abstractmethod
    def circuit(self, params):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def cost_function(self, params):
        pass


class HamiltonianV1(QAOAExperiment):
    """
    QAOA experiment for Hamiltonian as shown in Jiang et al. (2023) https://arxiv.org/abs/2305.09561.
    """

    name: str = "Hamiltonian V1"
    circuit_depth: int = 2
    cost_h: qml.Hamiltonian = None
    mixer_h: qml.Hamiltonian = None

    def __init__(
        self,
        preprocessor: preprocessors.BasicPreProcessor,
        circuit_depth: int = 2,
        **kwargs,
    ):
        super().__init__(preprocessor, **kwargs)
        self.circuit_depth = circuit_depth
        self.cost_h = self._compute_cost_h(kwargs.get("eps", 6), kwargs.get("c_p", 0.0))
        self.mixer_h = self._compute_mixer_h()

    def _penalty(self, stem_1, stem_2, c_p: float = 0.0) -> float:
        """
        Penalty function for two overlapping or pseudoknotted stems as described in Jiang et al. (2023).
        """
        if preprocessors.are_stems_overlapping(stem_1, stem_2):
            return -(len(stem_1) + len(stem_2))
        elif preprocessors.are_stems_pseudoknotted(stem_1, stem_2):
            return c_p * (len(stem_1) + len(stem_2))
        else:
            return 0

    def _compute_cost_h(self, eps: float = 6, c_p: float = 0.0) -> qml.Hamiltonian:
        print("[experiments.py] Generating cost Hamiltonian...")
        stems = self.preprocessor.selected_stems
        n_bases = self.preprocessor.rna_length
        n_stems = len(stems)

        coeffs = []
        observables = []
        for i in range(n_stems):
            stem_i = stems[i]
            k_i = len(stem_i)
            coeff = -2 * k_i + n_bases / (2 * k_i + eps)
            coeffs.append(coeff)
            observables.append((qml.Identity(wires=i) - qml.PauliZ(wires=i)) / 2)
            for j in range(0, i):
                stem_j = stems[j]
                penalty = self._penalty(stem_i, stem_j, c_p)
                if penalty != 0:
                    obs = (
                        (qml.Identity(wires=i) - qml.PauliZ(wires=i))
                        @ (qml.Identity(wires=j) - qml.PauliZ(wires=j))
                        / 4
                    )
                    coeffs.append(penalty)
                    observables.append(obs)

        h_c = utils.hamiltonian_sum(coeffs, observables)
        return h_c

    def _compute_mixer_h(self) -> qml.Hamiltonian:
        """
        Simple X-mixer as described in Jiang et al. (2023).
        """
        print("[experiments.py] Generating mixer Hamiltonian...")
        return qaoa.x_mixer(wires=range(self.n_qubits))

    def initial_layer(self):
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)

    def qaoa_layer(self, params):
        qaoa.cost_layer(params[0], self.cost_h)
        qaoa.mixer_layer(params[1], self.mixer_h)

    def circuit(self, params):
        self.initial_layer()
        qml.layer(self.qaoa_layer, self.circuit_depth, params)

    def cost_function(self, params):
        @qml.qnode(self.dev)
        def _circuit(params):
            self.circuit(params)
            return qml.expval(self.cost_h)

        return _circuit(params)

    def run(self):
        params = np.random.rand(self.circuit_depth, 2)
        print(f"Using device: {self.dev}")
        print(f"Initial params: {params}")
        for i in range(self.optimizer_steps):
            params, prev_cost = self.optimizer.step_and_cost(self.cost_function, params)
            print(f"Cost after step {i}: {prev_cost}")
            print(f"Params after step {i}: {params}")
