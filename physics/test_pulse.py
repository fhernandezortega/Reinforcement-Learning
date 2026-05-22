import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np
from physics.pulses import RamanPulse

pulse = RamanPulse()

print("=== Transferencia por pulso ===\n", flush=True)

for omega in [np.pi/2, np.pi, 1.0, 0.8]:
    A = pulse.transition_matrix(i=0, f=1, omega=omega, t=1.0)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    result = A @ state
    print(f"omega={omega:.4f} -> estado resultante: {np.round(result, 4)}", flush=True)

print("\n=== Secuencia 0->1->2->3 con omega=pi/2 ===\n", flush=True)

state = np.array([1.0, 0.0, 0.0, 0.0])
print(f"Inicial: {state}", flush=True)

for i, f in [(0,1), (1,2), (2,3)]:
    A = pulse.transition_matrix(i=i, f=f, omega=np.pi/2, t=1.0)
    state = A @ state
    state = state / np.sum(state)
    print(f"Tras pulso {i}->{f}: {np.round(state, 4)}", flush=True)