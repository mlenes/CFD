import numpy as np
import matplotlib.pyplot as plt

def htau_u(Peh):
	return 0.5*(1/(np.tanh(Peh)) - 1/Peh)
	
Peh = np.linspace(0.0001, 100, 1000)

plt.plot(Peh, htau_u(Peh))
plt.xlabel('Peh')
plt.ylabel('τ [h/u]')
plt.show()
