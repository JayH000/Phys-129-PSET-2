#Problem 1 part a conversion code 

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r,theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])
def cylindrical_to_cartesian(rho, psi, z):
    x=rho * np.cos(psi)
    y=rho * np.sin(psi)
    return np.array([x, y, z])
def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi, z])
def spherical_to_cylindrical(r,theta, phi):
    rho = r * np.sin(theta)
    psi = phi
    z = r * np.cos(theta)
    return np.array([rho, psi, z])
def cylindrical_to_spherical(rho, psi, z):
    r = np.sqrt(rho**2 + z**2)
    theta = np.arccos(z / r)
    phi = psi
    return np.array([r, theta, phi])
print(spherical_to_cartesian(1, np.pi/2, np.pi/2))
print(cartesian_to_spherical(0, 1, 0))
print(cylindrical_to_cartesian(1, np.pi/2, 1))
print(cartesian_to_cylindrical(0, 1, 1))
print(spherical_to_cylindrical(1, np.pi/2, np.pi/2))
print(cylindrical_to_spherical(1, np.pi/2, 1))

# problem 1 part b
def normalize(v):
    return v / np.linalg.norm(v)

def generate_orthonormal_basis(point):
    z = normalize(point) 
    if abs(z[0]) < 0.9:
         x = np.array([1, 0, 0])
    else:
        x = np.array([0, 1, 0])
        y = np.cross(z, x)
        x = np.cross(y, z)
        return x, y, z
def plot_sphere_with_frames(num_points=5):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    #create unit sphere 
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.3)
    #select and plot points on sphere
    theta = np.linspace(0, np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    for t in theta:
        for p in phi:
            point = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            x_vec, y_vec, z_vec = generate_orthonormal_basis(point)
            ax.scatter(*point, color='r')
#plot the coordinate frame 
            ax.quiver(*point, *x_vec * 0.2, color='b')  # x
            ax.quiver(*point, *y_vec * 0.2, color='g') # y
            ax.quiver(*point, *z_vec * 0.2, color='y') # z
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X_axis")
            ax.set_ylabel("Y_axis")
            ax.set_zlabel("Z_axis")
            ax.set_title("Local Orthonormal Coordinate Systems on the Unit Sphere")
            plt.show()
            


plot_sphere_with_frames(5)
         




 
  