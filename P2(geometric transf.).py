import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

# Define the stereographic projection from the unit sphere to the plane (z = 0)
def stereographic_projection(x, y, z):
    return x / (1 - z), y / (1 - z)

# Generate a mesh for the unit sphere
theta = np.linspace(0, np.pi, 50)  # Polar angle
phi = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
theta, phi = np.meshgrid(theta, phi)

# Parametric equations for the sphere
X = np.sin(theta) * np.cos(phi)
Y = np.sin(theta) * np.sin(phi)
Z = np.cos(theta)

# Define two sets of curves on the sphere
# Curve 1: Meridian (constant phi, varying theta)
phi_curve = np.pi / 4  # Choosing a specific meridian
theta_vals = np.linspace(0, np.pi, 50)
x_curve1 = np.sin(theta_vals) * np.cos(phi_curve)
y_curve1 = np.sin(theta_vals) * np.sin(phi_curve)
z_curve1 = np.cos(theta_vals)

# Curve 2: Parallel (constant theta, varying phi)
theta_curve = np.pi / 3  # Choosing a specific parallel
phi_vals = np.linspace(0, 2 * np.pi, 100)
x_curve2 = np.sin(theta_curve) * np.cos(phi_vals)
y_curve2 = np.sin(theta_curve) * np.sin(phi_vals)
z_curve2 = np.full_like(phi_vals, np.cos(theta_curve))  # Constant z

# Project the curves onto the plane
x_proj1, y_proj1 = stereographic_projection(x_curve1, y_curve1, z_curve1)
x_proj2, y_proj2 = stereographic_projection(x_curve2, y_curve2, z_curve2)

# Find approximate intersection point using closest z-values
z_diff = np.abs(z_curve1[:, None] - z_curve2)  # Compute pairwise z differences
intersection_idx1, intersection_idx2 = np.unravel_index(np.argmin(z_diff), z_diff.shape)

# Compute tangent vectors at the intersection on the sphere
t1_sphere = np.array([
    x_curve1[intersection_idx1 + 1] - x_curve1[intersection_idx1],
    y_curve1[intersection_idx1 + 1] - y_curve1[intersection_idx1],
    z_curve1[intersection_idx1 + 1] - z_curve1[intersection_idx1]
])

t2_sphere = np.array([
    x_curve2[intersection_idx2 + 1] - x_curve2[intersection_idx2],
    y_curve2[intersection_idx2 + 1] - y_curve2[intersection_idx2],
    z_curve2[intersection_idx2 + 1] - z_curve2[intersection_idx2]
])

# Compute the angle on the sphere
cos_angle_sphere = np.dot(t1_sphere, t2_sphere) / (norm(t1_sphere) * norm(t2_sphere))
angle_sphere = np.arccos
