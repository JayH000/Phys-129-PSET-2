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

#problem 1 part c
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize(v):
    return v / np.linalg.norm(v)

def spherical_basis_vectors(theta, phi):
    """Compute the spherical basis vectors at a given (theta, phi)."""
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi

def plot_sphere_with_spherical_basis(num_points=5):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.3)
    
    # Select points on the sphere
    theta_vals = np.linspace(0, np.pi, num_points)
    phi_vals = np.linspace(0, 2 * np.pi, num_points)
    for theta in theta_vals:
        for phi in phi_vals:
            point = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            e_r, e_theta, e_phi = spherical_basis_vectors(theta, phi)
            
            # Plot the point
            ax.scatter(*point, color='r')
            
            # Plot the spherical basis vectors
            ax.quiver(*point, *e_r * 0.2, color='b', label='$e_r$' if theta == 0 and phi == 0 else "")
            ax.quiver(*point, *e_theta * 0.2, color='g', label='$e_\theta$' if theta == 0 and phi == 0 else "")
            ax.quiver(*point, *e_phi * 0.2, color='r', label='$e_\phi$' if theta == 0 and phi == 0 else "")
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Spherical Basis Vectors on the Unit Sphere")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_sphere_with_spherical_basis()

    #problem 1 part d

    import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def surface_function(x, y):
    """Define the surface function z = f(x, y)."""
    return np.sin(x) * np.cos(y)  # Example surface

def compute_local_basis(x, y, z):
    """Compute local coordinate system (tangent vectors and normal) for the surface."""
    dz_dx, dz_dy = np.gradient(z, x[0, :], y[:, 0])
    
    # Tangent vectors
    t1 = np.array([np.ones_like(dz_dx), np.zeros_like(dz_dx), dz_dx])
    t2 = np.array([np.zeros_like(dz_dy), np.ones_like(dz_dy), dz_dy])
    
    # Normal vector (cross product of tangents)
    normal = np.cross(t1.T, t2.T).T
    
    # Normalize vectors
    t1 = t1 / np.linalg.norm(t1, axis=0)
    t2 = t2 / np.linalg.norm(t2, axis=0)
    normal = normal / np.linalg.norm(normal, axis=0)
    
    return t1, t2, normal

def plot_surface_with_frames():
    """Plot the surface and local coordinate frames."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    Z = surface_function(X, Y)
    
    t1, t2, normal = compute_local_basis(X, Y, Z)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
    
    # Plot frames at selected points
    for i in range(0, X.shape[0], 3):
        for j in range(0, X.shape[1], 3):
            point = np.array([X[i, j], Y[i, j], Z[i, j]])
            
            ax.quiver(*point, *t1[:, i, j] * 0.2, color='r')  # Tangent 1
            ax.quiver(*point, *t2[:, i, j] * 0.2, color='g')  # Tangent 2
            ax.quiver(*point, *normal[:, i, j] * 0.2, color='b')  # Normal
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Surface with Local Coordinate Frames')
    plt.show()

# Run the function to plot the surface with frames
plot_surface_with_frames()

#part e



