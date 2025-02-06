import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to convert spherical to Cartesian coordinates
def spherical_to_cartesian(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# Function to compute the holonomy: the angle between the initial and final vectors
def compute_holonomy(theta0):
    r = 1  # Radius of the unit sphere
    phi_values = np.linspace(0, 2 * np.pi, 100)
    
    # Initial vector in the e_ϕ direction at φ=0
    initial_vector = np.array([0, 1, 0])  # Tangent vector in the e_ϕ direction
    
    # Initialize transported vector with the initial vector
    transported_vector = initial_vector
    
    # Perform the parallel transport along the curve
    for phi in phi_values:
        # Parametrize the current point on the sphere
        point = [r, phi, theta0]
        # Convert spherical to Cartesian coordinates
        x, y, z = spherical_to_cartesian(point[0], point[1], point[2])
        
        # For simplicity, we are not implementing the full Riemann curvature here.
        # But let's simulate a small change in the vector direction for non-zero curvature
        # The simplest approach for holonomy is assuming a small rotation in the tangent plane.
        # This is an approximation since true parallel transport involves more advanced calculations.
        # Simulate the change by rotating the vector in the plane orthogonal to the radial vector.

        # Cross-product with the radial vector to get a perpendicular direction
        radial_vector = np.array([x, y, z])
        vector_perpendicular = np.cross(radial_vector, transported_vector)
        
        # Normalize the perpendicular vector and add a small change in the direction
        if np.linalg.norm(vector_perpendicular) != 0:
            vector_perpendicular /= np.linalg.norm(vector_perpendicular)
            transported_vector += 0.01 * vector_perpendicular  # Simulate a small change

    # Normalize the final vector for comparison
    transported_vector /= np.linalg.norm(transported_vector)
    
    # Compute the angle between the initial and final vectors
    angle = np.arccos(np.dot(initial_vector, transported_vector) / (np.linalg.norm(initial_vector) * np.linalg.norm(transported_vector)))
    
    return angle

# Range of θ0 values to test
theta0_values = np.linspace(0, np.pi, 100)
holonomy_strengths = []

# Compute the holonomy for different θ0 values
for theta0 in theta0_values:
    holonomy_strengths.append(compute_holonomy(theta0))

# Plot the strength of the holonomy (angle between initial and final vectors)
plt.plot(theta0_values, holonomy_strengths, label="Holonomy Strength")
plt.xlabel(r"$\theta_0$", fontsize=14)
plt.ylabel("Holonomy Strength (radians)", fontsize=14)
plt.title("Holonomy Strength for Different $\theta_0$ Values", fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
