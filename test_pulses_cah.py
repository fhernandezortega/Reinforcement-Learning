import sys
sys.path.append('.')
from physics.pulses_cah import CaHPulse
import numpy as np

pulse = CaHPulse()

print('Test de transferencia por pulso:')

for idx, p in enumerate(pulse.pulse_library):

    state = np.zeros(16)
    state[p['i']] = 1.0

    A0, A1 = pulse.transition_matrices(
        p['i'], p['f'], p['omega'], p['t']
    )

    p0 = np.sum(A0 @ state)
    p1 = np.sum(A1 @ state)
    pop_f_k1 = (A1 @ state)[p['f']]

    print(
        f"  Pulso {idx+1:2d}: "
        f"p(k=0)={p0:.3f} "
        f"p(k=1)={p1:.3f} "
        f"pop_f={pop_f_k1:.3f} "
        f"{p['label']}"
    )