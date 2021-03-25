import numpy as np
from openfermioncirq import SwapNetworkTrotterAnsatz
from openfermion.transforms import get_diagonal_coulomb_hamiltonian
from openfermioncirq.variational.letter_with_subscripts import LetterWithSubscripts
from openfermioncirq import swap_network
import cirq
import itertools

class CustomSwapNetworkTrotterAnsatz(SwapNetworkTrotterAnsatz):
    """With SwapNetworkTrotterHubbardAnsatz I kept getting stuck at about 0.94
    overlap with the true ground state. When I used SwapNetworkTrotterAnsatz, 
    I could get arbitrarily close, but the number of parameters increased too
    quickly to be practical on my computer. After experimenting, I noticed the 
    tunneling parameter for vertical tunneling was the same, so I assigned it 
    only one parameter instead of one per vertical tunneling term. That's the 
    main difference between this class and SwapNetworkTrotterAnsatz."""
    def __init__(self, hubbard, iterations=1):
        self.hubbard = hubbard
        super().__init__(get_diagonal_coulomb_hamiltonian(hubbard.hamiltonian()), iterations=iterations)
        
    def params(self):
        for i in range(self.iterations):
            yield LetterWithSubscripts('Tv', i)
            for p, q in itertools.combinations(range(len(self.qubits)), 2):
                if (self.include_all_xxyy or not np.isclose(
                        self.hamiltonian.one_body[p, q].real, 0)) and not custom_is_vertical_edge(
                    p, q, self.hubbard.lattice.x_dimension, 
                                                self.hubbard.lattice.y_dimension, self.hubbard.lattice.periodic):
                    yield LetterWithSubscripts('T', p, q, i)
                if (self.include_all_yxxy or not
                        np.isclose(self.hamiltonian.one_body[p, q].imag, 0)):
                    yield LetterWithSubscripts('W', p, q, i)
                if (self.include_all_cz or not
                        np.isclose(self.hamiltonian.two_body[p, q], 0)):
                    yield LetterWithSubscripts('V', p, q, i)
    
    def param_bounds(self):
        """Bounds on the parameters."""
        bounds = []
        for param in self.params():
            if param.letter == 'U' or param.letter == 'V':
                bounds.append((-1.0, 1.0))
            elif param.letter == 'T' or param.letter == 'Tv' or param.letter == 'W':
                bounds.append((-2.0, 2.0))
            else: 
                print(param.letter)
        return bounds

    def operations(self, qubits): 
        param_set = set(self.params())
        for i in range(self.iterations):
            # Apply one- and two-body interactions with a swap network that
            # reverses the order of the modes
            def one_and_two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
                tv_symbol = LetterWithSubscripts('Tv', i)
                t_symbol = LetterWithSubscripts('T', p, q, i)
                w_symbol = LetterWithSubscripts('W', p, q, i)
                v_symbol = LetterWithSubscripts('V', p, q, i)
                if custom_is_vertical_edge(p, q, self.hubbard.lattice.x_dimension, 
                                                self.hubbard.lattice.y_dimension, self.hubbard.lattice.periodic):
                    yield cirq.ISwapPowGate(exponent=-tv_symbol).on(a,b)
                if t_symbol in param_set:
                    yield cirq.ISwapPowGate(exponent=-t_symbol).on(a, b)
                if w_symbol in param_set:
                    yield cirq.PhasedISwapPowGate(exponent=w_symbol).on(a, b)
                if v_symbol in param_set:
                    yield cirq.CZPowGate(exponent=v_symbol).on(a, b)
            yield swap_network(
                    qubits, one_and_two_body_interaction, fermionic=True)
            qubits = qubits[::-1]

            # Apply one- and two-body interactions again. This time, reorder
            # them so that the entire iteration is symmetric
            def one_and_two_body_interaction_reversed_order(p, q, a, b
                    ) -> cirq.OP_TREE:
                tv_symbol = LetterWithSubscripts('Tv', i)
                t_symbol = LetterWithSubscripts('T', p, q, i)
                w_symbol = LetterWithSubscripts('W', p, q, i)
                v_symbol = LetterWithSubscripts('V', p, q, i)
                if v_symbol in param_set:
                    yield cirq.CZPowGate(exponent=v_symbol).on(a, b)
                if w_symbol in param_set:
                    yield cirq.PhasedISwapPowGate(exponent=w_symbol).on(a, b)
                if t_symbol in param_set:
                    yield cirq.ISwapPowGate(exponent=-t_symbol).on(a, b)
                if custom_is_vertical_edge(p, q, self.hubbard.lattice.x_dimension, 
                                                self.hubbard.lattice.y_dimension, self.hubbard.lattice.periodic):
                    yield cirq.ISwapPowGate(exponent=-tv_symbol).on(a,b)

            yield swap_network(
                    qubits, one_and_two_body_interaction_reversed_order,
                    fermionic=True, offset=True)
            qubits = qubits[::-1]

    def default_initial_params(self):
        total_time = self.adiabatic_evolution_time
        step_time = total_time / self.iterations
        hamiltonian = self.hamiltonian

        params = []
        for param in self.params():
            if param.letter == 'U':
                params.append(_canonicalize_exponent(
                    # Maybe this isn't the best way to do this...
                    -hamiltonian.one_body[0,0].real * step_time / np.pi, 2))
            elif param.letter == 'Tv':
                params.append(_canonicalize_exponent(
                    # Maybe this isn't the best way to do this...
                    hamiltonian.one_body[0,2].real * step_time / np.pi, 4))
            elif param.letter == 'W' or param.letter == 'T' or param.letter == 'V':
                p, q, i = param.subscripts
                # Use the midpoint of the time segment
                interpolation_progress = 0.5 * (2 * i + 1) / self.iterations
                if param.letter == 'T':
                    params.append(_canonicalize_exponent(
                        hamiltonian.one_body[p, q].real *
                        step_time / np.pi, 4))
                elif param.letter == 'W':
                    params.append(_canonicalize_exponent(
                        hamiltonian.one_body[p, q].imag *
                        step_time / np.pi, 4))
                elif param.letter == 'V':
                    params.append(_canonicalize_exponent(
                        -hamiltonian.two_body[p, q] * interpolation_progress *
                        step_time / np.pi, 2))
        return np.array(params)

def _canonicalize_exponent(exponent: float, period: int) -> float:
    # Shift into [-p/2, +p/2).
    exponent += period / 2
    exponent %= period
    exponent -= period / 2
    # Prefer (-p/2, +p/2] over [-p/2, +p/2).
    if exponent <= -period / 2:
        exponent += period  # coverage: ignore
    return exponent

def custom_is_vertical_edge(p, q, x_dim, y_dim, periodic):
    if periodic: raise ValueError("Periodic not implemented")
    else:
        return p == q + 2 * x_dim or q == p + 2 * x_dim
