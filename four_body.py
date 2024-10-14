import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def fourbody_ES(t, x, mu):
    """ The ODE function for the orbit in the BCR4BP in the Earth-Sun centered frame
    
    Inputs:
    - t: the time
    - x: A numpy array of 4 elements, the orbit states x, y, x_dot, y_dot
    - mu: The CR3BP mass-ratio parameter.
    """
    LU = 1.49598E+8 # Earth-Sun distance, or 1 AU, [km]
    TU = 5022635 # seconds
    
    xdot = threebody_orbit(t, x, mu) 
    x_dim = x * LU

    
    # tau = # Time non-dim scaling factor

    mu_m = 4.9028695E+3 # Gravitational parameter of the Moon [km^3 s^-2]
    R_m = 384400# Orbital radius of the Moon about the Earth [-]
    moon_earth_orbit_seconds = 2360591.78
    w_m = 2*np.pi/moon_earth_orbit_seconds # Angular velocity of the Moon about the Earth

    r_m = [LU* (1-mu) + R_m * np.cos(w_m * t),
            R_m * np.sin(w_m * t)] 

    r = x_dim[0:2] - r_m # Vector from S/C to the Moon
    
    a2b = -mu_m * r / np.linalg.norm(r) ** 3


    a2b = a2b * TU**2/LU # Non-dimensionalize acceleration


    xdot[2:4] += a2b

    return xdot

def threebody_orbit(t, x, mu):
    """
    The ODE function for the orbit in the CR3BP

    Inputs:
    - t: The time
    - x: A numpy array of 4 elements, the orbit states x, y, x_dot, y_dot
    - mu: The only parameter of importance in the CR3BP
    """
    r = np.linalg.norm(x[0:2])

    xdot = np.empty((4,))
    xdot[0:2] = x[2:5]

    r1 = np.sqrt((x[0] + mu)**2. + x[1]**2.)
    r2 = np.sqrt((x[0] - 1. + mu)**2. + x[1]**2.)

    xdot[2] = 2.*x[3] + x[0] - ((1. - mu)*(x[0] + mu)/(r1**3.)) - (mu*(x[0] - 1. + mu)/(r2**3.))
    xdot[3] = -2.*x[2] + x[1] - ((1. - mu)*x[1]/(r1**3.)) - (mu*x[1]/(r2**3.))
    #xdot[5] = -((1. - mu)*x[2]/(r1**3.)) - (mu*x[2]/(r2**3.))

    return xdot

def plot_traj(trajectory_position, mu, LU):
    # Function to format the axis labels
    def thousands(x, pos):
        'The two args are the value and tick position'
        return '%1.0fk' % (x * 1e-3)
    
    # # Plot the trajectory
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(trajectory_position[0], trajectory_position[1], trajectory_position[2], label="Trajectory")

    # ax.scatter([(1-mu)*LU], [0], [0], marker='o', color='pink', label="Primary 2")
    # ax.scatter([-mu*LU], [0], [0], marker='o', color='blue', label="Primary 1")

    # # libration_pts = calculate_libration_pts(mu)
    # # libration_x = [coord[0] for coord in libration_pts]
    # # libration_y = [coord[1] for coord in libration_pts]
    # # ax.scatter(libration_x, libration_y, [0], marker='<', color='black', label='Libration Points')            


    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    # ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    # ax.zaxis.set_major_formatter(FuncFormatter(thousands))

    # plt.ylabel("y")
    plt.title("CR3BP Trajectory")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    ax2D = plt.figure().add_subplot()
    ax2D.plot(trajectory_position[0], trajectory_position[1], label="Trajectory")
    ax2D.scatter([(1-mu)*LU], [0], marker='o', color='pink', label="Primary 2")
    ax2D.scatter([-mu*LU], [0], marker='o', color='blue', label="Primary 1")
    # ax2D.scatter(libration_x, libration_y, marker='<', color='black', label='Libration Points')  
    ax2D.set_xlabel("x")
    ax2D.set_ylabel("y")
    plt.title("CR3BP Trajectory, X-Y Plane")

    ax2D_nd = plt.figure().add_subplot()
    ax2D_nd.plot(trajectory_position[0]/LU, trajectory_position[1]/LU, label="Trajectory")
    ax2D_nd.scatter([1-mu], [0], marker='o', color='blue', label="Primary 2")
    # ax2D_nd.scatter([-mu], [0], marker='o', color='blue', label="Primary 1")
    ax2D_nd.set_xlabel("x")
    ax2D_nd.set_ylabel("y")
    plt.title("CR3BP Trajectory, X-Y Plane, Non-Dimensioinal Units")

    
    
    return ax2D, ax2D_nd


initial_x = np.array([ 9.99996946e-01,  3.29255398e-05, -3.18556330e-01, 2.88493051e-01])

print('initial_x', initial_x)
t_span = [0,3]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

mu = 3.054200000000000E-6
LU = 149597871
TU = 5022635 #seconds

# Solve the ODE
integrator = solve_ivp(fourbody_ES, t_span, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14)    
trajectory = integrator.y

trajectory_position = trajectory[:2]

# dimensionalize trajectory
trajectory_position = trajectory_position * LU

ax2D,ax2D_nd = plot_traj(trajectory_position, mu, LU)

plt.show()
