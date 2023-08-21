import numpy as np
import pennylane as qml


def hamiltonian_sum(coeffs_for_sum, obs_to_sum):
    """
    Sum hamiltonians with different coefficients and observables. Why not use pythonic way (e.g. 0.5 * (I - Z))?
    Because it is at least 100 times slow. See https://discuss.pennylane.ai/t/efficient-hamiltonian-sum-for-qaoa/2591/2

    coeffs_for_sum: list of coefficients for each hamiltonian
    obs_to_sum: list of observables for each hamiltonian
    """
    assert len(obs_to_sum) == len(coeffs_for_sum)
    obs = []
    coeffs = np.array([])

    for i in range(len(obs_to_sum)):
        obs += obs_to_sum[i].terms()[1]
        coeff = np.array(obs_to_sum[i].terms()[0]) * coeffs_for_sum[i]
        coeffs = np.append(coeffs, coeff)

    return qml.Hamiltonian(coeffs=coeffs, observables=obs).simplify()
