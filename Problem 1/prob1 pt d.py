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
            
            ax.quiver(*point, *t1[:, i, j] * 0.5, color='r')  # Tangent 1
            ax.quiver(*point, *t2[:, i, j] * 0.5, color='g')  # Tangent 2
            ax.quiver(*point, *normal[:, i, j] * 0.5, color='b')  # Normal
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Surface with Local Coordinate Frames')
    plt.show()

# Run the function to plot the surface with frames
plot_surface_with_frames()
