import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from fractions import Fraction
import random
from functools import lru_cache

# Function to select a random number a that is relatively prime to N
def select_a(N):
    while True:
        a = random.randint(2, round(N**0.5))  # ensures a <= sqrt(N) as recommended in lectures
        gcd = np.gcd(a, N)
        if gcd == 1:  # a is relatively prime to N
            return a, True
        elif gcd > 1:
            return gcd, False  # Factor found by chance!

# Function to apply controlled modular exponentiation (a^(2^j) mod N)
def controlled_mod_exp(qc, a, N, control_qubit, target_qubits):
    num_target_qubits = len(target_qubits)
    for i in range(num_target_qubits):
        exp = 2**i
        # Controlled unitary to represent a^(2^i) mod N
        qc.cx(control_qubit, target_qubits[i])  # Example, replace with correct modular operation

# Quantum Phase Estimation to find the order r of a mod N
@lru_cache(maxsize= None)
def QPEN(a, N):
    num_qubits = int(np.ceil(np.log2(N))) + 2
    qc = QuantumCircuit(num_qubits + 4, num_qubits)
    for qubit in range(num_qubits):
        qc.h(qubit)

    # First qubit in the target register is set to state |1>
    qc.x(num_qubits + 1)
    
    for qubit in range(num_qubits):
        controlled_mod_exp(qc, a, N, qubit, list(range(num_qubits, num_qubits + 4)))

    qft_circuit = QFT(num_qubits).inverse()
    qc.append(qft_circuit, range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    aer_sim = AerSimulator()
    transpiled_qc = transpile(qc, aer_sim)
    result = aer_sim.run(transpiled_qc).result()

    counts = result.get_counts()
    measured_phases = []
    
    for output in counts:
        decimal = int(output, 2)
        phase = decimal / (2**num_qubits)
        measured_phases.append(phase)
    return measured_phases

# Function to find the order r of a (mod N) with the measured phases from QPEN
def r_of_a_mod_N(a, N):
    phases = QPEN(a, N)
    fractions = [Fraction(phase).limit_denominator(N) for phase in phases]
    orders = []
    for fraction in fractions:
        if fraction.denominator != 1 and fraction.denominator not in orders:
            orders.append(fraction.denominator)
    return orders

# Main body for Shor's algorithm
def shors_algorithm(N):
    # If N is even, return 2 as a factor
    if N % 2 == 0:
        return 2, N // 2, True

    # Try multiple times to find valid a values or factors by luck
    for i in range(100):
        a, relatively_prime = select_a(N)

        if not relatively_prime:
            return a, N // a, True  # Found factor (GCD)

        orders = r_of_a_mod_N(a, N)
        if len(orders) > 0:
            r = orders[0]

            # If the order is even, calculate factors
            if r % 2 == 0:
                factor1 = np.gcd(a**(r // 2) - 1, N)
                factor2 = np.gcd(a**(r // 2) + 1, N)

                if factor1 > 1 and factor2 > 1:
                    return factor1, factor2, False
    return None

# Recursively calls Shor's Algorithm on its results until they can no longer be prime factorized.
@lru_cache(maxsize= None)
def all_the_way_down(N):
    
    # Base case
    if N < 2:
        return []

    result = shors_algorithm(N)

    if result:
        factor1, factor2, composite = result
        if composite:  # Composite primes can only occur in the second argument
            # Recursion all the way down.
            factors = [factor1] + all_the_way_down(factor2)
        else:
            factors = [factor1] + [factor2]
    else:
        # If Shor’s algorithm fails to find factors, N is now the base prime.
        factors = [N]

    return sorted(factors)

# Main execution
try:
    N = int(input("Choose an integer N: "))
    result = all_the_way_down(N)
    print(f"{'-'*21} Results! {'-'*21}")
    print(f"N = {N}")
    if result:
        print(f"Prime factors: {result}")
    else:
        print("No factors found.")
except ValueError:
    print(f"Input is not a valid integer.")