import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

# Define the stereographic projection from the unit sphere to the plane (z = 0)
def stereographic_projection(P):
    x, y, z = P  # Unpack the array into x, y, z
    if np.isclose(z, 1, atol=1e-6):  # Avoid division by zero for z ≈ 1
        return None  # Skip projection for this point
    return np.array([x / (1 - z), y / (1 - z)])

# Generate a mesh for the unit sphere
theta = np.linspace(0, np.pi, 50)  
phi = np.linspace(0, 2 * np.pi, 100)  
theta, phi = np.meshgrid(theta, phi)

# Parametric equations for the sphere
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Define two sets of curves on the sphere
def curve1(t):
    return np.array([np.sin(t) * np.cos(t), np.sin(t) * np.sin(t), np.cos(t)])

def curve2(t):
    return np.array([np.sin(t) * np.cos(t + np.pi/2), np.sin(t) * np.sin(t + np.pi/2), np.cos(t)])

t = np.linspace(0, np.pi, 100)
curve1_points = np.array([curve1(t_i) for t_i in t])
curve2_points = np.array([curve2(t_i) for t_i in t])

# Compute tangent vectors
def tangent_vector(curve, t):
    h = 1e-6
    return (curve(t + h) - curve(t - h)) / (2 * h)

t1 = np.pi / 4
t2 = np.pi / 4 + np.pi / 2

tangent1_curve1 = tangent_vector(curve1, t1)
tangent1_curve2 = tangent_vector(curve2, t1)
tangent2_curve1 = tangent_vector(curve1, t2)
tangent2_curve2 = tangent_vector(curve2, t2)

# Compute angles on the sphere
def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

angle_sphere = angle_between_vectors(tangent1_curve1, tangent1_curve2)

# Project curves while filtering out None values
curve1_projected = np.array([stereographic_projection(P) for P in curve1_points if stereographic_projection(P) is not None])
curve2_projected = np.array([stereographic_projection(P) for P in curve2_points if stereographic_projection(P) is not None])

# Compute tangent vectors of projected curves
def tangent_vector_projected(curve, index):
    h = 1  # Use integer step for index-based array access
    if index - h < 0 or index + h >= len(curve):  # Avoid out-of-bounds errors
        return np.array([0, 0])  
    return (curve[index + h] - curve[index - h]) / (2 * h)

index_t1 = int(t1 / np.pi * len(curve1_projected))  # Convert t1 to index
index_t2 = int(t2 / np.pi * len(curve1_projected))  # Convert t2 to index

tangent1_curve1_projected = tangent_vector_projected(curve1_projected, index_t1)
tangent1_curve2_projected = tangent_vector_projected(curve2_projected, index_t1)
tangent2_curve1_projected = tangent_vector_projected(curve1_projected, index_t2)
tangent2_curve2_projected = tangent_vector_projected(curve2_projected, index_t2)

# Compute angles on the plane
angle_plane = angle_between_vectors(tangent1_curve1_projected, tangent1_curve2_projected)

# Plotting
fig = plt.figure(figsize=(12, 6))

# Plot on the unit sphere
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, color='w', alpha=0.1)
ax1.plot(curve1_points[:, 0], curve1_points[:, 1], curve1_points[:, 2], label='Curve 1')
ax1.plot(curve2_points[:, 0], curve2_points[:, 1], curve2_points[:, 2], label='Curve 2')
ax1.quiver(curve1(t1)[0], curve1(t1)[1], curve1(t1)[2], tangent1_curve1[0], tangent1_curve1[1], tangent1_curve1[2], color='r', label='Tangent 1')
ax1.quiver(curve2(t1)[0], curve2(t1)[1], curve2(t1)[2], tangent1_curve2[0], tangent1_curve2[1], tangent1_curve2[2], color='b', label='Tangent 2')
ax1.set_title(f'Unit Sphere\nAngle: {np.degrees(angle_sphere):.2f}°')
ax1.legend()

# Plot after stereographic projection
ax2 = fig.add_subplot(122)
ax2.plot(curve1_projected[:, 0], curve1_projected[:, 1], label='Curve 1 Projected')
ax2.plot(curve2_projected[:, 0], curve2_projected[:, 1], label='Curve 2 Projected')
ax2.quiver(curve1_projected[index_t1][0], curve1_projected[index_t1][1], tangent1_curve1_projected[0], tangent1_curve1_projected[1], angles='xy', scale_units='xy', scale=1, color='r', label='Tangent 1 Projected')
ax2.quiver(curve2_projected[index_t1][0], curve2_projected[index_t1][1], tangent1_curve2_projected[0], tangent1_curve2_projected[1], angles='xy', scale_units='xy', scale=1, color='b', label='Tangent 2 Projected')
ax2.set_title(f'Stereographic Projection\nAngle: {np.degrees(angle_plane):.2f}°')
ax2.legend()

plt.show()
