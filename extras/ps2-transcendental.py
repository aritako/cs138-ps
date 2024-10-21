import numpy as np
from scipy.optimize import fsolve

# Define the transcendental equation f(n)
def f(n):
    return n*np.pi*np.cos(0.5*n*np.pi) + 2*np.sin(0.5*n*np.pi) - 138*np.exp(1)

# Initial guess for n (you may need to try different guesses)
n_guess = 128.24088
# Solve the equation using fsolve
solution = fsolve(f, n_guess)

print(f"Solution for n: {solution}")
