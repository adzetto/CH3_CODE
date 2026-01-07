# Non-Linear Truss Analysis

This repository contains a Python implementation for the non-linear analysis of truss structures, capable of handling both geometric non-linearities (large deformations) and material non-linearities (elastoplasticity). It is based on the algorithmic procedures described in **Chapter 3 of Bonet & Wood**.

## Problem Statement

The main objective is to determine the equilibrium path of truss structures under external loading. The code handles:
*   **Geometric Non-linearity**: Using a logarithmic strain measure to account for finite strains and large rotations.
*   **Material Non-linearity**: implementing an elastoplastic material model with isotropic hardening.
*   **Instability**: Capturing snap-through and snap-back phenomena using path-following techniques.

## Algorithms Used

The solution strategy involves several advanced numerical methods:

### 1. Global Solution Scheme: Newton-Raphson with Arc-Length Control
To trace the non-linear equilibrium path, including limit points, the code employs a **Cylindrical Arc-Length Method** combined with the Newton-Raphson iterative scheme.
*   **Predictor Step**: Estimates the next equilibrium point along the tangent of the path.
*   **Corrector Step**: Iteratively corrects the solution to satisfy equilibrium within a specified tolerance while constraining the step length.

### 2. Kinematics: Logarithmic Strain
The deformation is described using the logarithmic strain measure:
$$
\varepsilon = \ln\left(\frac{l}{L}\right)
$$
where $l$ is the current length and $L$ is the initial length of the truss element. This measure is conjugate to the Kirchhoff stress.

### 3. Constitutive Update: Return Mapping Algorithm
For elastoplasticity, the stress is updated using an **Elastic Predictor - Plastic Corrector** scheme (Return Mapping):
1.  **Elastic Predictor**: Assume the step is purely elastic and calculate a trial stress.
2.  **Yield Check**: Check if the trial stress exceeds the yield surface defined by $\Phi = |\tau| - (\sigma_{y0} + H\alpha) \le 0$.
3.  **Plastic Corrector**: If yielding occurs, return the stress to the yield surface and update the internal hardening variables (plastic strain $\varepsilon^p$ and accumulated plastic strain $\alpha$).

## Examples

The repository includes several benchmark examples demonstrating the capabilities of the solver:

### 1. Shallow Dome (`dome_example.py`)
A 3D shallow truss dome subjected to a central point load. This problem exhibits complex snapping behavior (snap-through).
*   **Output**: `figure_3_11_shallow_dome.pdf`

### 2. 2D Arch (`arch_example.py`)
A deep circular arch modeled with truss elements. It typically demonstrates limit points and unstable branches in the load-displacement curve.
*   **Output**: `figure_3_10_arch.pdf`

### 3. Lee's Frame (`lee_frame.py`)
A classic benchmark problem involving a frame structure. This implementation models the frame using a "trussed" equivalent (two chords with cross-bracing) to simulate bending behavior using only truss elements.
*   **Outputs**:
    *   `figure_3_9_trussed_frame.pdf`
    *   `figure_3_9_trussed_frame_clamped.pdf`

## File Structure

*   `truss_analysis.py`: Core library containing `TrussElement` and `TrussAnalysis` classes.
*   `truss_algorithm.tex`: LaTeX documentation describing the mathematical formulation.
*   `truss_examples.ipynb`: Jupyter Notebook for interactive exploration of the examples.
*   `*.py` (examples): Python scripts for running specific simulations.
*   `*.pdf`: Generated plots of load-displacement curves and deformed shapes.

## Usage

To run an example, simply execute the corresponding Python script:

```bash
python arch_example.py
```

Ensure you have the required dependencies installed (e.g., `numpy`, `matplotlib`).
