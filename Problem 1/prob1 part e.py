import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_basis_vectors(theta, phi):
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_theta, e_phi

def parallel_transport(theta_0, phi_0, alpha, beta, num_steps=100):
    thetas = np.linspace(theta_0, np.pi / 2, num_steps)
    transported_vectors = []
    for theta in thetas:
        e_theta, e_phi = spherical_basis_vectors(theta, phi_0)
        transported_vector = alpha * e_theta + beta * np.sin(theta_0) / np.sin(theta) * e_phi
        transported_vectors.append(transported_vector)
    return thetas, transported_vectors

def plot_parallel_transport():
    theta_0 = np.pi / 5
    phi_0 = 0
    alpha, beta = 1, 1  # Initial vector components
    
    thetas, transported_vectors = parallel_transport(theta_0, phi_0, alpha, beta)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.3)
    
    # Scale arrows and adjust density
    arrow_scale = 0.6  # Increase scaling factor to make arrows larger
    step = 5  # Reduce step size to plot fewer arrows (adjust for clarity)
    
    for i, theta in enumerate(thetas[::step]):  # Adjust density by changing step
        point = np.array([np.sin(theta) * np.cos(phi_0), np.sin(theta) * np.sin(phi_0), np.cos(theta)])
        ax.quiver(*point, *transported_vectors[i] * arrow_scale, color='r')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Parallel Transport of Vector on Sphere")
    plt.show()

if __name__ == "__main__":
    plot_parallel_transport()
