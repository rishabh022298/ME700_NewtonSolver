from Newton_Solver import Newton_Solver as ns
import pytest
import numpy as np
from pathlib import Path
import re

'''
Test 1: For checking the proper import of Newton Solver
'''
def test_hello_world():
    known_statement = "hello world!"
    found_statement = ns.hello_world()
    assert known_statement == found_statement

'''
Test 2: Tests if the initial inputs are sensible
'''
def test_initial_guesses_check_correct():
    F_1 = lambda x,y: x**2 + y**2
    F_2 = lambda x,y: x**3 + y**3
    F = [F_1,F_2]
    x0 = np.array([1,2])
    assert ns.initial_guesses_check(F, x0) == True

'''
Test 3: Tests if the function is returning the ValueError if the inputs aren't sensible.
'''
def test_initial_guesses_check_incorrect():
    F1 = lambda x: x[0]*x[0] + x[1]*x[1]
    F2 = lambda x: (x[0])**3 + (x[1])**3
    F = [F1,F2]
    x0 = [1]
    with pytest.raises(ValueError, match = "The inputs are either missing expressions or initial guesses."):
        ns.initial_guesses_check(F, x0)

'''
Test 4: Tests if the function is calculating delta x properly.
'''
def test_calculate_delta_x():
    F_1 = lambda x: x[0]*x[0] + 2*x[0]*x[1]
    F_2 = lambda x: (x[0])**16 + 4*x[1]
    F = lambda x: np.array([F_1(x), F_2(x)])
    J_11 = lambda x: 2*x[0] + 2*x[1]
    J_12 = lambda x: 2*x[0]
    J_21 = lambda x: 16*((x[0])**15)
    J_22 = lambda x: 4
    xi = [1, 1]
    J = lambda x: np.array([[J_11(x), J_12(x)], [J_21(x), J_22(x)]])
    F_xi = F(xi)
    J_xi = J(xi)
    calculated_delta_x = ns.calculate_delta_x(J_xi,F_xi)
    actual_delta_x = [0.125, -1.75]
    assert np.allclose(actual_delta_x, calculated_delta_x)

'''
Test 5: Test for if the function is returning a ValueError if the Jacobian is singular.
'''
def test_Singular_J():
    J = [[4, 2],[8, 4]]
    with pytest.raises(ValueError, match = "Jacobian matrix is singular."):
        ns.check_Jacobian(J)

'''
Test 6: Test for non singular Jacobian
'''
def test_Valid_J():
    Jx = [[4, 2], [16, 4]]
    assert ns.check_Jacobian(Jx) == True

'''
Test 7: Test for Newton Solver for 1D case for a system with known roots.
'''
def test_NewtonSolver_1D():
    F_1 = lambda x: x**2 - 4
    J_11 = lambda x: 2*x
    F = lambda x: np.array([F_1(x)])
    J = lambda x: np.array([[J_11(x)]])
    x0_list = [5]  # Provide multiple initial guesses
    known_roots = [2]
    
    calculated_roots = ns.run_newton_solver(F, J, x0_list, tol=1e-6, max_iter=100)
    
    # Check if all expected roots are found
    assert all(any(np.isclose(root, calculated, atol=1e-6) for calculated in calculated_roots) for root in known_roots)

'''
Test 8: Test for Newton Solver for 2D case for a system with known roots.
'''
def test_NewtonSolver_2D():
    # Define functions F and J
    F_1 = lambda x: (x[0])**2 + 2*x[0]*x[1]
    F_2 = lambda x: (x[1])**2 + 2*x[0]*x[1]
    F = lambda x: np.array([F_1(x), F_2(x)])

    J_11 = lambda x: 2*x[0] + 2*x[1]
    J_12 = lambda x: 2*x[0]
    J_21 = lambda x: 2*x[1]
    J_22 = lambda x: 2*x[1] + 2*x[0]

    x0_list = np.array([[0.25, 0.25]])  # Multiple initial guesses
    J = lambda x: np.array([[J_11(x), J_12(x)],
                            [J_21(x), J_22(x)]])
    
    known_roots = np.array([[0, 0]])  # Only one root expected

    # Pass the initial guess list directly
    calculated_roots = ns.run_newton_solver(F, J, x0_list, tol=1e-6, max_iter=1000)

    # Filter roots that are close to each other
    unique_roots = []
    for root in calculated_roots:
        if not any(np.allclose(root, r, atol=1e-6) for r in unique_roots):
            unique_roots.append(root)

    # Convert the list of calculated roots into a numpy array for comparison
    calculated_roots_array = np.array([root for root in unique_roots])
    assert np.allclose(np.array(known_roots), calculated_roots_array, atol=1e-3)


'''
Test 9: Test for convergence. Checks if the function is returning a RuntimeError.
'''
def test_convergence():
    F_1 = lambda x: x[0]**2 + 2*x[0]*x[1]
    F_2 = lambda x: x[1]**2 + 2*x[0]*x[1]
    F = lambda x: np.array([F_1(x), F_2(x)])
    J_11 = lambda x: 2*x[0] + 2*x[1]
    J_12 = lambda x: 2*x[0]
    J_21 = lambda x: 2*x[1]
    J_22 = lambda x: 2*x[1] + 2*x[0]
    x0 = np.array([100, 100])  # Ensure x0 is a 2D array with 2 elements
    J = lambda x: np.array([[J_11(x), J_12(x)],
                            [J_21(x), J_22(x)]])
    with pytest.raises(RuntimeError, match="Newton solver did not converge within given tolerance and iterations. Try changing parameters."):
        ns.run_newton_solver(F, J, [x0], tol=1e-6, max_iter=10)

'''
Test 10: Test for checking if the function is returning multiple unique roots if the reasonable multiple initial guesses are provided.
'''
def test_NewtonSolver_MultipleRoots():
    # Define the function and its Jacobian
    F = lambda x: np.array([x**2 - 4])  # Function: x^2 - 4
    J = lambda x: np.array([[2*x]])  # Jacobian: dF/dx = 2x

    # Provide multiple initial guesses
    x0_list = [5, -5, 1, -1]

    # Known roots
    known_roots = [2, -2]

    # Compute roots using Newton's method
    calculated_roots = ns.run_newton_solver(F, J, x0_list, tol=1e-6, max_iter=100)

    # Ensure both roots are found
    assert all(any(np.isclose(root, calculated, atol=1e-6) for calculated in calculated_roots) for root in known_roots)

    # Ensure no extra roots are found
    assert len(calculated_roots) == len(known_roots)
