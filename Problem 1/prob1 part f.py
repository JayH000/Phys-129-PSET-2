import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the initial parameters
theta0 = np.pi / 4  # Initial angle θ0
r = 1  # Radial coordinate (unit sphere)
phi_initial = 0  # Initial φ
phi_final = 2 * np.pi  # Final φ

# Parametrization of the curve γ(ϕ)
def gamma(phi):
    return np.array([r, phi, theta0])  # Returning (r, phi, theta)

# Function to convert spherical to Cartesian coordinates
def spherical_to_cartesian(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# Function for the parallel transport of the vector (simplified as tangent transport)
def parallel_transport(phi, vector):
   
    return vector

# Set up the initial vector in the ϕ direction at φ = 0
# We assume the initial vector is tangent to the surface of the sphere at φ = 0, pointing along e_ϕ
initial_vector = np.array([0, 1, 0])  # Tangent vector in the e_ϕ direction

# Array for φ values
phi_values = np.linspace(phi_initial, phi_final, 100)

# Arrays to store the Cartesian coordinates of the transported vectors
vector_points = []
vector_directions = []

# Perform parallel transport along the curve
for phi in phi_values:
    # Parametrize the current point on the sphere
    point = gamma(phi)
    
    # Convert the spherical coordinates to Cartesian coordinates
    x, y, z = spherical_to_cartesian(point[0], point[1], point[2])
    vector_points.append([x, y, z])
    
    # Assume the vector remains in the tangent plane at each point
    transported_vector = parallel_transport(phi, initial_vector)
    
    # Add the vector direction (same for simplicity in this example)
    vector_directions.append(transported_vector)

# Convert lists to numpy arrays
vector_points = np.array(vector_points)
vector_directions = np.array(vector_directions)

# Plot the sphere with vectors on the surface
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid 
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere
ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5)

# Plot the parallel transported vectors
for i, point in enumerate(vector_points):
    ax.quiver(point[0], point[1], point[2], vector_directions[i][0], vector_directions[i][1], vector_directions[i][2], length=0.1, color='r')

# Set labels and view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Parallel Transport of Vectors on a Sphere (partf)")


ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.show()
