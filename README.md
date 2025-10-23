# Numerical Solutions for PDEs - Project 2

## Overview

This project implements numerical solutions for Partial Differential Equations (PDEs) using the Finite Element Method (FEM). The implementation focuses on solving one-dimensional boundary value problems and analyzing convergence properties of the numerical solutions.

## Project Structure

- `FEM.py` - Core implementation of the Finite Element Method solver
- `2d.py` - Example script demonstrating the solver with a specific test case

## Features

### Core Classes

- **Grid**: Manages spatial discretization with flexible grid generation
- **Function**: Represents discrete functions on the grid with basic arithmetic operations
- **Solution**: Extends Function to handle numerical solutions with exact solution comparison
- **Solver**: Implements the FEM solver for linear second-order ODEs

### Capabilities

- Solves differential equations of the form: `a*u'' + b*u' + c*u = f(x)`
- Supports Dirichlet boundary conditions (u = 0 at boundaries)
- Provides error analysis with L2 and H1 norm calculations
- Convergence analysis with automatic mesh refinement
- Visualization of numerical vs exact solutions and error plots

## Mathematical Background

The solver uses piecewise linear finite elements to discretize the weak formulation of the differential equation. For each element, it constructs local stiffness matrices that account for:

- Diffusion term (second derivative): coefficient `a`
- Advection term (first derivative): coefficient `b`
- Reaction term (zeroth order): coefficient `c`

## Usage

### Basic Example

```python
from FEM import solve_system
import numpy as np

# Define the right-hand side function
def f(x):
    return np.where(x < 1/2, 2, -2)

# Define the exact solution for comparison
def exact(x):
    return np.where(x < 1/2, 2*x, 2*(1 - x))

# Solve the system with 100 grid points
u = solve_system(f, exact=exact, N=100)

# Plot comparison between numerical and exact solutions
u.plot_comparison()
plt.show()

# Plot the error
u.plot_error()
plt.show()
```

### Convergence Analysis

```python
from FEM import plot_convergence

# Analyze convergence rates
plot_convergence(f, exact, title="Convergence Analysis")
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install numpy matplotlib scipy
   ```
3. Run the example:
   ```
   python 2d.py
   ```

## Examples

The `2d.py` script demonstrates solving a piecewise linear forcing function that results in a tent-shaped exact solution. This example showcases:

- Solution accuracy on a non-smooth problem
- Error visualization
- Convergence rate analysis

## Theory

This implementation solves boundary value problems using the Galerkin finite element method with piecewise linear basis functions. The method provides:

- Second-order convergence in the L2 norm
- First-order convergence in the H1 norm
- Stable numerical solutions for well-posed problems

## Contributing

This project is part of a numerical analysis course. Contributions should focus on:

- Additional test cases
- Performance improvements
- Extended boundary condition support
- Higher-order element implementations
