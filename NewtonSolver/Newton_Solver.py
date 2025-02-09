'''
This file contains a few functions required to implement the Newton's Method
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

'''
Hello world function to check the proper importing and functionality of the Newton solver.
'''
def hello_world():
    return "hello world!"

'''
This function checks if the number of equations and the number of guesses are same.
It returns a ValueError if rhe sizes don't match
'''
def initial_guesses_check(F, x0):
    if isinstance(F, list) and all(callable(f) for f in F):
        if len(F) == len(x0):
            return True
        else:
            raise ValueError("The inputs are either missing expressions or initial guesses.")

'''
This function ensures if the Jacobian is non singular.
It returns ValueError if the Jacobian matrix is singular.
'''
def check_Jacobian(Jx):
    if np.linalg.det(Jx) == 0:
        raise ValueError("Jacobian matrix is singular.")
    return True

'''
This function calculates delta X = J^(-1)*F for the iterative values of x
'''
def calculate_delta_x(J_xi, F_xi):
    return np.linalg.solve(J_xi, -F_xi)

'''
This is the main function which implements the Newton Sovler
'''
def newton_solver(F: Callable[[np.ndarray], np.ndarray], 
                  J: Callable[[np.ndarray], np.ndarray], 
                  x0_list: list, 
                  tol: float, 
                  max_iter: int) -> list:
    """
    Finds multiple roots using the Newton-Raphson method.

    Parameters:
    - F: Function handle for the system of equations (returning a vector).
    - J: Function handle for the Jacobian matrix (returning a matrix).
    - x0_list: List of initial guesses.
    - tol: Convergence tolerance.
    - max_iter: Maximum number of iterations.

    Returns:
    - roots: List of unique converged roots.
    """
    roots = []                          # For storing the roots
    initial_guesses_check(F, x0_list)   # Checking the initial guesses

    all_iterations = []                 # Storing all iterations for plotting purposes

    for x0 in x0_list:
        x0 = np.atleast_1d(np.asarray(x0, dtype=float)) # Ensures that x0 has at least one entry
        guesses = []                    # Storing iterative values of the roots
        F_values = []                   # Storing iterative values of function

        for iter_count in range(1, max_iter + 1):
            # Fx represent value of F at x, i.e. F(x), similarly for Jx
            Fx = np.atleast_1d(np.asarray(F(x0), dtype=float))  # Ensures F(x0) is an array
            Jx = np.atleast_2d(np.asarray(J(x0), dtype=float))  # Ensures J is a dimension higher than F

            check_Jacobian(Jx)          # For checking if Jacobian is singular or not.
            delta_x = calculate_delta_x(Jx, Fx.reshape(-1, 1))  # Calculates delta x while ensuring that F is a column vector for matrix multiplication

            x0 = x0 + delta_x.flatten() # Iterates the value of the guesses and ensures that delta x is a row vector for addition

            guesses.append(x0)
            F_values.append(Fx)

            Fx_norm = np.linalg.norm(Fx, ord=np.inf) # Calculating the maximum norm of F for checking the convergence

            if Fx_norm < tol:
                # Avoid duplicates: Check if root is already found
                if not any(np.allclose(x0, root, atol=tol) for root in roots):
                    roots.append(x0)
                all_iterations.append({'guesses': guesses, 'F_values': F_values})
                break
        else:
            # Raises RuntimeError if the solution is not converging within the given maximum number of iterations.
            raise RuntimeError("Newton solver did not converge within given tolerance and iterations. Try changing parameters.")
    
    # Plottig only if the system is 1D.
    if np.shape(F(x0))[0] == 1:  # Check if the system is 1D (i.e., F(x0) has 1 row)
        plot_iterations(all_iterations)
    return roots

'''
Plotting function.
'''
def plot_iterations(all_iterations):
    """
    Plots the guesses and function values for each iteration, combining the results of all guesses in a single plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot guesses vs. iterations
    plt.subplot(1, 2, 1)
    for idx, iteration_data in enumerate(all_iterations):
        guesses = np.array(iteration_data['guesses'])
        # Plot guesses (each component of the guess)
        plt.plot(range(len(guesses)), guesses[:, 0], label=f'Guess {idx + 1}')

    plt.title("Iteration for Calculating the Root (All Guesses)")
    plt.xlabel("Number of steps")
    plt.ylabel("Root")
    plt.grid(True)
    plt.legend()

    # Plot F(x0) vs. iterations
    plt.subplot(1, 2, 2)
    for idx, iteration_data in enumerate(all_iterations):
        F_values = np.array(iteration_data['F_values'])
        # Plot F(x) for each guess
        plt.plot(range(len(F_values)), F_values[:, 0], label=f'F(x) for Guess {idx + 1}')

    plt.title("Iterative Values of F(x) for All Guesses")
    plt.xlabel("Iteration")
    plt.ylabel("F(x) Value")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()