import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import fsolve
import math
from matplotlib.animation import FuncAnimation


def U_hessian(mu, X):
    """
    Compute U_xx, U_xy, U_yy, Uyz, Uzz
    Inputs:
    - mu: ...
    - X: State
    """
    r1 = np.sqrt((X[0] + mu) ** 2. + X[1] ** 2. + X[2] ** 2.)
    r2 = np.sqrt((X[0] - 1. + mu) ** 2. + X[1] ** 2. + X[2] ** 2.)
    x = X[0]
    y = X[1]
    z = X[2]
    Uxx = 1. - ((1. - mu)/(r1**3.)) - (mu/(r2**3.)) + (3.*(1. - mu)*((x + mu)**2.)/(r1**5.)) + \
        (3.*mu*((x - 1. + mu)**2.)/(r2**5.))
    Uxy = (3.*(1. - mu)*(x + mu)*y/(r1**5.)) + (3.*mu*(x - 1. + mu)*y/(r2**5.))
    Uyy = 1. - ((1. - mu)/(r1**3.)) - (mu/(r2**3.)) + (3.*(1. - mu)*(y**2.)/(r1**5.)) + (3.*mu*(y**2.)/(r2**5.))
    Uxz = 3.*((1. - mu)*(x + mu)*z/(r1**5.)) + 3.*(mu*(x - 1. + mu)*z/(r2**5.))
    Uyz = 3.*((1. - mu)*y*z/(r1**5.)) + 3*(mu*y*z/(r2**5.))
    Uzz = -((1. - mu)/(r1**3.)) - (mu/(r2**3.)) + 3.*((1. - mu)*(z**2.)/(r1**5.)) + 3.*mu*((z**2.)/(r2**5.))
    Uxx_mat = np.array([[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]])

    return Uxx_mat

def A_mat_CR3BP(mu, X):
    """
    Compute the general plant matrix for CR3BP
    Inputs:
    - mu: ...
    - X: State
    """
    U_mat = U_hessian(mu, X)
    A_mat = np.zeros((6, 6))
    A_mat[0:3, 3:6] = np.identity(3)
    A_mat[3:6, 0:3] = U_mat
    A_mat[3, 4] = 2.
    A_mat[4, 3] = -2.

    return A_mat

def cr3bp_equations(t, state, mu):
    x, y, z, vx, vy, vz = state
    r13 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r23 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    Omega_x = x - (1-mu)*(x+mu)/r13**3 - mu * (x-1+mu) / r23**3
    Omega_y = y-(1-mu)*y/r13**3 - mu*y/r23**3
    Omega_z = -(1 - mu) * z / r13**3 - mu * z / r23**3

    ax = 2*vy + Omega_x
    ay = -2*vx + Omega_y
    az = Omega_z

    return [vx, vy, vz, ax, ay, az]

def ode_STM_cr3bp(t, x, mu):
    '''
    The ODE function for the orbit and STM components, combined

    Inputs:
    - t: The time
    - x: A numpy array of 42 elements. The first six are the orbit states, the remainder are STM elements
    - mu: The only parameter of importance in the CR3BP
    '''

    # Build the A matrix for phi_dot = A*phi:
    A_local = A_mat_CR3BP(mu, x[0:6])

    x_dot = np.zeros((42,))

    # Orbital Dynamics:
    x_dot[0:6] = cr3bp_equations(t, x[0:6], mu)

    # State Transition Matrix Dynamics:
    phi = np.reshape(x[6:42], (6, 6), order='F')
    phi_dot = np.dot(A_local, phi)
    phi_dot_reshape = np.reshape(phi_dot, 36, order='F')
    x_dot[6:42] = phi_dot_reshape

    return x_dot

def fourbody_ES(t, x, mu, phase):
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
    moon_earth_orbit_seconds = 2551442.8032/TU
    w_m = 2*np.pi/moon_earth_orbit_seconds # Angular velocity of the Moon about the Earth

    r_m = [LU* (1-mu) + R_m * np.cos(w_m * t + phase),
            R_m * np.sin(w_m * t + phase)] 

    r = x_dim[0:2] - r_m # Vector from S/C to the Moon
    
    a2b = -mu_m * r / np.linalg.norm(r) ** 3


    a2b = a2b * TU**2/LU # Non-dimensionalize acceleration


    xdot[2:4] += a2b

    # print(t)

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

def calculate_libration_pts(mu):
    # returns non-dimensional libration pts

    def equations_L1_L2_L3(x, mu, position):
        if position == 'L1':
            return x - (1 - mu)/(x + mu)**2 + mu / (x - 1 + mu)**3
        elif position == 'L2':
            return -x + (1 - mu) / (x + mu)**2 + mu/(x - 1 + mu)**2
        elif position == 'L3':
            return -x - (1 - mu)/(x + mu)**2 - mu/(x - 1 + mu)**2

    pts = []

    # triangular libration points
    pts.append([.5-mu, np.sqrt(3)/2])
    pts.append([.5-mu, -np.sqrt(3)/2])

    # Solve for L1, L2, L3 using fsolve
    L1_x = fsolve(equations_L1_L2_L3, 0.5 - mu, args=(mu, 'L1'))[0]
    L2_x = fsolve(equations_L1_L2_L3, 1.5 - mu, args=(mu, 'L2'))[0]
    L3_x = fsolve(equations_L1_L2_L3, -1.0 - mu, args=(mu, 'L3'))[0]

    pts.append([L1_x, 0])
    pts.append([L2_x, 0])
    pts.append([L3_x, 0])

    pts = np.array(pts) * LU

    return pts

def calculate_jacobi_once(state, mu):
    x, y, z, vx, vy, vz = state
    r13 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r23 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)


    U = 0.5 * (x**2 + y**2) + (1 - mu) / r13 + mu / r23
    v2 = vx**2 + vy**2 + vz**2
    
    C = 2*U - v2

    return C

def calculate_jacobi(traj, mu):
    jacobi_arr = []
    for i in range(len(traj[0])):
        x = traj[0][i]
        y = traj[1][i]
        z = traj[2][i]
        vx = traj[3][i]
        vy = traj[4][i]
        vz = traj[5][i]

        r13 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r23 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        
        H = .5*(vx**2 + vy**2 + vz**2) - .5*(x**2 + y**2) - (1-mu)/r13**3 - mu/r23**3
        C = -2*H
        # jacobi_arr.append(C)

        U = 0.5 * (x**2 + y**2) + (1 - mu) / r13 + mu / r23
        v2 = vx**2 + vy**2 + vz**2

        jacobi_arr.append(2*U - v2)
    
    
    x = np.linspace(0,len(traj[0]), len(traj[0]))
    plt.figure(figsize=(10,6))
    plt.plot(x, jacobi_arr)
    plt.title('Jacobi Constant')

    jacobi = np.mean(jacobi_arr)
    print('Jacobi Constant', jacobi)
    return jacobi

def plot_traj(trajectory_position, mu, LU):
    # Function to format the axis labels
    def thousands(x, pos):
        'The two args are the value and tick position'
        return '%1.0fk' % (x * 1e-3)
    
    # Plot the trajectory
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(trajectory_position[0], trajectory_position[1], trajectory_position[2], label="Trajectory")

    ax.scatter([(1-mu)*LU], [0], [0], marker='o', color='pink', label="Primary 2")
    ax.scatter([-mu*LU], [0], [0], marker='o', color='blue', label="Primary 1")

    libration_pts = calculate_libration_pts(mu)
    libration_x = [coord[0] for coord in libration_pts]
    libration_y = [coord[1] for coord in libration_pts]
    ax.scatter(libration_x, libration_y, [0], marker='<', color='black', label='Libration Points')            


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.zaxis.set_major_formatter(FuncFormatter(thousands))

    # plt.ylabel("y")
    plt.title("CR3BP Trajectory")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    ax2D = plt.figure().add_subplot()
    ax2D.plot(trajectory_position[0], trajectory_position[1])
    ax2D.scatter([(1-mu)*LU], [0], marker='o', color='pink', label="Primary 2")
    ax2D.scatter([-mu*LU], [0], marker='o', color='blue', label="Primary 1")
    ax2D.scatter(libration_x, libration_y, marker='<', color='black', label='Libration Points')  
    ax2D.set_xlabel("x")
    ax2D.set_ylabel("y")
    plt.title("Trajectory, X-Y Plane")

    ax2D_nd = plt.figure().add_subplot()
    ax2D_nd.plot(trajectory_position[0]/LU, trajectory_position[1]/LU)
    ax2D_nd.scatter([1-mu], [0], marker='o', color='blue', label="Earth")
    # ax2D_nd.scatter([-mu], [0], marker='o', color='blue', label="Primary 1")
    ax2D_nd.set_xlabel("x")
    ax2D_nd.set_ylabel("y")
    plt.title("3BP Trajectory, X-Y Plane, Sun-Earth, Non-Dimensioinal Units")

    
    
    return ax, ax2D, ax2D_nd

def calculate_stability(eigenvalues, eigenvectors):


    stable_eigs = []
    unstable_eigs = []
    for i in range(len(eigenvalues)):
        val = eigenvalues[i]
        vec = eigenvectors[i]

        if abs(val.imag) < 1e-8 and val.real > 1+1e-5:
            stable_eigs.append((val, vec))
        elif abs(val.imag) < 1e-8 and val.real < 1-1e-5:
            unstable_eigs.append((val, vec))


    sorted_eigenvalues = [[],[]]
    for i in range(len(eigenvalues)):
        ev1 = eigenvalues[i]
        vec = eigenvectors[:,i]
        if abs(ev1 - 1) < 1e-2:
            continue
        if len(sorted_eigenvalues[0]) == 0:
            sorted_eigenvalues[0].append((ev1,vec))
        elif len(sorted_eigenvalues[0]) == 1 and (1/sorted_eigenvalues[0][0][0]).real - ev1.real < 1e-6 and (1/sorted_eigenvalues[0][0][0]).imag - ev1.imag < 1e-6:
            sorted_eigenvalues[0].append((ev1,vec))
        elif len(sorted_eigenvalues[1]) == 0:
            sorted_eigenvalues[1].append((ev1,vec))
        elif len(sorted_eigenvalues[1]) == 1 and (1/sorted_eigenvalues[1][0][0]).real - ev1.real < 1e-6 and (1/sorted_eigenvalues[1][0][0]).imag - ev1.imag < 1e-6:
            sorted_eigenvalues[1].append((ev1,vec))
    
    assert len(sorted_eigenvalues[0]) == 2 and len(sorted_eigenvalues[1]) == 2

    assert (sorted_eigenvalues[0][0][0] + sorted_eigenvalues[0][1][0]).imag < 1e-8 and (sorted_eigenvalues[1][0][0] + sorted_eigenvalues[1][1][0]).imag < 1e-8
    
    stability_indices = [(sorted_eigenvalues[0][0][0] + sorted_eigenvalues[0][1][0]).real, (sorted_eigenvalues[1][0][0] + sorted_eigenvalues[1][1][0]).real]
    return stability_indices, sorted_eigenvalues, stable_eigs, unstable_eigs

def compute_manifold(eigenvalue_pair, trajectory, t_eval, ax2D, ax2D_nd, ax3D, LU, mu):

    def event_unstable(t,y, mu):
        # return y[0]
        return y[0]-1+mu
        val = y[0] * LU - 405000
        return val
    
    event_unstable.terminal = True
    event_unstable.direction = 1

    def event_stable(t,y, mu):
        return y[0]-1+mu
        val = y[0] * LU - 405000
        return val
    
    event_stable.terminal = True
    event_stable.direction = 1


    if eigenvalue_pair[0][0] > 1:
        stable_eig = eigenvalue_pair[1]
        unstable_eig = eigenvalue_pair[0]
    else:
        stable_eig = eigenvalue_pair[0]
        unstable_eig = eigenvalue_pair[1]
    
    # print('stable_eig', stable_eig)
    # print('unstable_eig', unstable_eig)

    manifold_initial = trajectory[:6,::100]
    # print('manifold_initial', manifold_initial)
    stm_arr = trajectory[6:42,::100]
    t_arr = t_eval[::100]

    epsilon = 1e-4
    stable_eig_val = stable_eig[0]
    stable_eiv_vec = stable_eig[1].reshape(6,1) * 1
    t_span_stable = (0, -10)
    t_eval_stable = np.linspace(t_span_stable[0], t_span_stable[1], 10000)

    unstable_eig_val = unstable_eig[0]
    print('Using unstable eigenvalue for manifold:', unstable_eig_val)
    unstable_eiv_vec_f = unstable_eig[1].reshape(6,1) * 1
    unstable_eiv_vec_b = unstable_eig[1].reshape(6,1) * -1
    t_span_unstable = (0, 10)
    t_eval_unstable = np.linspace(t_span_unstable[0], t_span_unstable[1], 10000)

    stable_end_pt = np.empty((0, 6))
    unstable_end_pt = np.empty((0, 6))
    for i in range(len(t_arr)):
        t = t_arr[i]
        m = manifold_initial[:,i]
        stm = stm_arr[:,i]
        stm = np.reshape(stm, (6, 6), order='F')

        # stable
        perturbation = np.matmul(stm, stable_eiv_vec)
        perturbation = perturbation/LA.norm(perturbation) * epsilon
        initial_x = (m + np.reshape(perturbation, (1,6))).flatten()
        integrator = solve_ivp(cr3bp_equations, t_span_stable, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval_stable, rtol=1e-13, atol=1e-14, events=event_stable)   
        trajectory = np.c_[ integrator.y, integrator.y_events[0][0]] 
        
        stable_end_pt = np.vstack((stable_end_pt, integrator.y_events[0]))
        ax2D_nd.plot(trajectory[0], trajectory[1], color='green',  linewidth=.4)
        ax2D.plot(trajectory[0]*LU, trajectory[1]*LU, color='green', linewidth=.4)
        ax3D.plot(trajectory[0]*LU, trajectory[1]*LU, trajectory[2]*LU, color='green', linewidth=.4)

        # unstable going forward
        perturbation = np.matmul(stm, unstable_eiv_vec_f)
        perturbation = perturbation/LA.norm(perturbation) * epsilon
        initial_x = (m + np.reshape(perturbation, (1,6))).flatten()
        integrator = solve_ivp(cr3bp_equations, t_span_unstable, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval_unstable, rtol=1e-13, atol=1e-14, events=event_unstable)   
        trajectory = np.c_[ integrator.y, integrator.y_events[0][0]]  

        unstable_end_pt = np.vstack((unstable_end_pt, integrator.y_events[0]))
        ax2D_nd.plot(trajectory[0], trajectory[1], color='red', linewidth=.4)
        ax2D.plot(trajectory[0]*LU, trajectory[1]*LU, color='red',  linewidth=.4)
        ax3D.plot(trajectory[0]*LU, trajectory[1]*LU, trajectory[2]*LU, color='red', linewidth=.4)

        # # unstable going backward
        # perturbation = np.matmul(stm, unstable_eiv_vec_b)
        # perturbation = perturbation/LA.norm(perturbation) * epsilon
        # initial_x = (m + np.reshape(perturbation, (1,6))).flatten()

        # integrator = solve_ivp(cr3bp_equations, t_span_unstable, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval_unstable, rtol=1e-13, atol=1e-14, events=event_unstable)   
        # trajectory = integrator.y
        # ax2D_nd.plot(trajectory[0], trajectory[1], color='green', linewidth=.4)
        # ax2D.plot(trajectory[0]*LU, trajectory[1]*LU, color='green', linewidth=.4)
        # ax3D.plot(trajectory[0]*LU, trajectory[1]*LU, trajectory[2]*LU, color='green', linewidth=.4)


    ax2D_nd.scatter(manifold_initial[0], manifold_initial[1], marker='.', color='black')
    ax2D.scatter(manifold_initial[0]*LU, manifold_initial[1]*LU, marker='.', color='black')
    ax3D.scatter(manifold_initial[0]*LU, manifold_initial[1]*LU, marker='.', color='black')  

    return stable_end_pt, unstable_end_pt

def compute_poincare_1(mu):
    def y_to_vx(x, y_array, vy, jacobi, mu):
        vx_array = []
        
        i = 0
        while i < len(y_array):
            y = y_array[i]
            r13 = np.sqrt((x + mu)**2 + y**2)
            r23 = np.sqrt((x - 1 + mu)**2 + y**2)
            vx_squared = x**2 + y**2 + 2*((1-mu)/r13 + mu/r23) - vy**2 - jacobi
            if vx_squared < 0:
                y_array= np.delete(y_array, i)
            else:
                vx = np.sqrt(vx_squared)
                vx_array.append(vx)
                i += 1
        
        
        return np.array(vx_array), y_array

    
    def poincare_event(t,y, mu):
        return y[0] - 1 + mu
    
    poincare_event.terminal = False
    poincare_event.direction = 1
    
    jacobi = 3.175  # Jacobi constant – to be fixed
    y_min = 0.0005
    y_max = 0.115
    n_pts = 100

    
    y_array = np.linspace(y_min, y_max, n_pts)
    vx_array, y_array = y_to_vx(1-mu, y_array, 0, jacobi, mu)
    t_span = [0,25]
    t_eval = np.linspace(t_span[0], t_span[1], 120)
    print('vx_array', vx_array)
    print('y_array', y_array)

    plt.style.use('ggplot')

    ax_p = plt.figure().add_subplot()  

    for i in range(len(vx_array)):
        vx = vx_array[i]
        y = y_array[i]
        initial = [1-mu, y, 0, vx, 0, 0]
        
        integrator = solve_ivp(cr3bp_equations, t_span, initial, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14, events=poincare_event)    
        stop_states = integrator.y_events
        if stop_states[0].shape[0] < 1:
            print('stop_states[0].shape[0] < 1')
            continue

        # print('stop states', stop_states)

        # stop_states = stop_states[0][:10,:]
        stop_states = stop_states[0]

        y_final = stop_states[:,1]
        ydot_final = stop_states[:,4]
        # print('x', x_final)
        # print('ydot', ydot_final)
        ax_p.scatter(y_final, ydot_final, s=1.6)

    ax_p.set_xlabel("y")
    ax_p.set_ylabel("ydot")

    # plt.ylabel("y")
    plt.title("Poincare Map")
    plt.legend()
    plt.grid()

    ################# VALIDATION ###################

    initial = (0.98785,0.042836, 0, 0.58641,0.0, 0)
    t_span = [0,25]
    t_eval = np.linspace(t_span[0], t_span[1], 120)
    ax_v = plt.figure().add_subplot()  
    
    integrator = solve_ivp(cr3bp_equations, t_span, initial, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14, events=poincare_event)    
    stop_states = integrator.y_events
    if stop_states[0].shape[0] < 12:
        print('stop_states[0].shape[0] < 12')

        # print('stop states', stop_states)

        # stop_states = stop_states[0][:10,:]
    stop_states = stop_states[0]

    y_final = stop_states[:,1]
    ydot_final = stop_states[:,4]
    # print('x', x_final)
    # print('ydot', ydot_final)
    ax_v.scatter(y_final, ydot_final, s=1.6)

    ax_v.set_xlabel("y")
    ax_v.set_ylabel("ydot")

    # plt.ylabel("y")
    plt.title("Poincare Validation")
    plt.legend()
    plt.grid()
    return

def compute_poincare(initial_state, jacobi, mu):
    def vy_to_vx(initial_state, vy_array, jacobi, mu):
        x, y, z, vx, vy, vz = initial_state
        vy = vy_array
        r13 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r23 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        vx_array = np.sqrt(x**2 + 2*((1-mu)/r13 + mu/r23) - vy**2 - jacobi)
        return vx_array
    
    def poincare_event(t,y, mu):
        return y[1]
    
    poincare_event.terminal = False
    poincare_event.direction = 1


    x = initial_state[0]
    vy = initial_state[4]
    vy_array = np.linspace(vy, vy + .0000001, 30)
    vx_array = vy_to_vx(initial_state, vy_array, jacobi, mu)
    t_span = [0,25]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    print('vx_array', vx_array)

    ax_p = plt.figure().add_subplot()  

    for i in range(len(vx_array)):
        vx = vx_array[i]
        vy = vy_array[i]
        initial = [x, 0, 0, vx, vy, 0]
        print(calculate_jacobi_once(initial, mu))
        
        integrator = solve_ivp(cr3bp_equations, t_span, initial, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14, events=poincare_event)    
        stop_states = integrator.y_events
        assert stop_states[0].shape[0] >= 4

        # print('stop states', stop_states)

        stop_states = stop_states[0][:4,:]

        x_final = stop_states[:,0]
        ydot_final = stop_states[:,4]
        # print('x', x_final)
        # print('ydot', ydot_final)
        ax_p.scatter(x_final, ydot_final, s=1)

    ax_p.set_xlabel("x")
    ax_p.set_ylabel("ydot")

    # plt.ylabel("y")
    plt.title("Poincare Map")
    plt.legend()
    plt.grid()

    return

def compute_poincare_y_ydot(unstable_states, stable_states, jacobi, mu):


    ax_p = plt.figure().add_subplot() 
    
    y = unstable_states[:,1]
    ydot = unstable_states[:,4]

    coord = []
    for i in range(len(y)):
        coord.append((y[i],ydot[i]))
    print('poincare coord:', coord)

    ax_p.scatter(y, ydot, s=2, color='red', label='Unstable Manifold')

    y = stable_states[:,1]
    ydot = stable_states[:,4]
     
    ax_p.scatter(y, ydot, s=2, color='green', label='Stable Manifold')
    ax_p.vlines(0, -.1, .4, colors='blue', label='Earth')

    ax_p.set_xlabel("y")
    ax_p.set_ylabel("ydot")

    # plt.ylabel("y")
    plt.title("Poincare Map")
    plt.legend()
    plt.grid()

    return ax_p

def twisting(jacobi, mu, ax2D_nd, ax_p, LU):

    def calc_vx(x, y, vy, jacobi, mu):
        
        r13 = np.sqrt((x + mu)**2 + y**2)
        r23 = np.sqrt((x - 1 + mu)**2 + y**2)
        vx_squared = x**2 + y**2 + 2*((1-mu)/r13 + mu/r23) - vy**2 - jacobi
        if vx_squared < 0:
            return None
        else:
            vx = np.sqrt(vx_squared)
    
        return vx
    
    def event_stable(t,y, mu):
        return y[0]-1+mu
    
    event_stable.terminal = True
    event_stable.direction = 1

    x = 1-mu
    strip_0 =  (-0.0025759305550807137, 0.029830142937422078)
    y = strip_0[0]
    strip_ydot_unpruned = np.linspace(strip_0[1] + .000070, strip_0[1] + .000076, 3)

    strip_vy = []
    strip_vx = []
    for vy in strip_ydot_unpruned:
        vx = calc_vx(x, y, vy, jacobi, mu)
        if vx:
            strip_vy.append(vy)
            strip_vx.append(vx)

    initial_guesses = []
    
    for i in range(len(strip_vy)):
        vy = strip_vy[i]
        vx = strip_vx[i]
        initial_x = np.array([x,y,0,vx,vy,0])
        print('i', i)
        print('initial_x', initial_x)
        t_span = [0,-5]
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        integrator = solve_ivp(cr3bp_equations, t_span, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14, events=event_stable)   
        if len(integrator.y_events[0]) > 0:
            trajectory = np.c_[integrator.y, integrator.y_events[0][0]] 
            initial_guesses.append(integrator.y_events[0][0])
        else:
            trajectory = integrator.y
        
        ax2D_nd.plot(trajectory[0], trajectory[1], color='black', linewidth=1.3, label='CR3BP Trajectory')

    # plot strip
    ax_p.scatter(np.full((len(strip_vy),1), y), strip_vy, s = 4, color='black', label='ydot variation')
    
    # plot moon's orbit
    moon_r = 384400/LU
    print('moon_r', moon_r)
    print('earth parking orbit', (6371+200)/LU)
    circle = plt.Circle((1-mu, 0), moon_r, color='pink',label='Moon Orbit', fill=False)
    ax2D_nd.add_artist(circle)
    
    ax2D_nd.axis('equal')

    print('initial_guesses', initial_guesses)

    return initial_guesses

    

def find_initial_guess(initial_state, mu, LU, TU, t_span):


    initial_stm = np.identity(6)
    initial_stm = np.reshape(initial_stm, 36, order='F')

    initial_x = np.concatenate([initial_state, initial_stm])

    print('initial_x', initial_x)
    t_eval = np.linspace(t_span[0], t_span[1], 12000)

    # Solve the ODE
    integrator = solve_ivp(ode_STM_cr3bp, t_span, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14)    
    trajectory = integrator.y

    jacobi = calculate_jacobi(trajectory, mu)
    trajectory_position = trajectory[:3]

    # dimensionalize trajectory
    trajectory_position = trajectory_position * LU

    ax3D,ax2D,ax2D_nd = plot_traj(trajectory_position, mu, LU)

    print('last x', trajectory[:,-1])
    print('initial to end diff', initial_x[:6] - trajectory[:6,-1])

    monodromy = np.reshape(trajectory[6:,-1], (6, 6), order='F')
    print('monodromy', monodromy)
    eigenvalues, eigenvectors = LA.eig(monodromy)
    print('eigenvalues', eigenvalues)

    stability_indices, sorted_eigenvalues, stable_eigs, unstable_eigs = calculate_stability(eigenvalues, eigenvectors)

    print('stability_indices', stability_indices)
    print('sorted_eigenvalues', sorted_eigenvalues)
    print('stable_eigs', stable_eigs)
    print('unstable_eigs', unstable_eigs)

    stable_end_pt, unstable_end_pt = compute_manifold(sorted_eigenvalues[0], trajectory, t_eval, ax2D, ax2D_nd, ax3D, LU, mu)

    ax_p = compute_poincare_y_ydot(unstable_end_pt, stable_end_pt, jacobi, mu)

    guesses = twisting(jacobi, mu, ax2D_nd, ax_p, LU)

    return guesses, ax3D,ax2D,ax2D_nd, ax_p





if __name__ == "__main__":

    


    # Define parameters for Sun-Earth System
    mu = 3.054200000000000E-6
    LU = 149597871
    TU = 5022635 #seconds

    # Initial conditions: [x0, y0, z0, vx0, vy0, vz0]

    initial_state = np.array([9.9195722538726860E-1, 6.2636208563647735E-22, -1.1416420063466291E-27, -1.9437416224690786E-16, -1.1810020232594805E-2, 3.6867737976780971E-27])

    

    # Time span for the integration
    period = 3.0848099797153212E+0
    t_span = (0, period)

    guesses, ax3D,ax2D,ax2D_nd, ax_p = find_initial_guess(initial_state, mu, LU, TU, t_span)


    best_guess_3d = guesses[0]

    guess = np.array([best_guess_3d[0], best_guess_3d[1], best_guess_3d[3], best_guess_3d[4]])

    t_span = [0,3]
    t_eval = np.linspace(t_span[0], t_span[1], 50000)

    def event_moon(t,y, mu):
        return y[0]-1+mu
    
    event_moon.terminal = True
    event_moon.direction = 1

    #3body
    tbp_integrator = solve_ivp(cr3bp_equations, t_span, best_guess_3d, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14, events=event_moon)    
    three_body_trajectory = tbp_integrator.y
    if len(tbp_integrator.y_events[0]) > 0:
        three_body_trajectory = np.c_[three_body_trajectory, tbp_integrator.y_events[0][0]] 
    tf = tbp_integrator.t_events[0]
    print('t_events 3',tbp_integrator.t_events[0] )

    ax_compare = plt.figure().add_subplot()
    ax_compare.plot(three_body_trajectory[0], three_body_trajectory[1], color='black', label="Three Body Trajectory")
    ax_compare.scatter([1-mu], [0], marker='o', color='blue', label="Earth")

    # 4body
    def event_time(t,y, mu,phase):
        return t - tf
    
    event_time.terminal = True
    event_time.direction = 1

    phase_arr = np.linspace(0, 2*np.pi, 1000)
    phase_arr = np.linspace(5, 5.1, 100)
    # phase_arr = np.linspace(5.5, 5.8, 50)

    min_dist_arr = []
    t_min_dist_arr = []
    min_vel_norm_arr = []
    min_vel_arr = []
    t_min_v_arr = []

    # plot analysis
    ax_dist = plt.figure().add_subplot()
    ax_vel = plt.figure().add_subplot()
    ax_dis_vel = plt.figure().add_subplot()

    moon_to_L1 = 61350 / LU
    moon_to_L2= 61347 / LU

    best_phase = 5.06162

    phase_arr = [best_phase]  # OVERRIDE

    for phase in phase_arr:

        print('phase', phase)

        bad_phases = [1.89, 2.35, 2.40, 2.45]
        skip = False
        for bad_phase in bad_phases:
            if abs(phase - bad_phase) < 5e-2:
                skip = True

        if skip:
            continue

        fbp_integrator = solve_ivp(fourbody_ES, t_span, guess, method='DOP853', args=(mu,phase,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14, events=event_time)    
        four_body_trajectory = fbp_integrator.y
        if len(fbp_integrator.y_events[0]) > 0:
            four_body_trajectory = np.c_[four_body_trajectory, fbp_integrator.y_events[0][0]] 
        
        moon_earth_orbit = 2551442.8032/TU # [s]
        w_m = 2*math.pi/moon_earth_orbit # Angular velocity of the Moon about the Earth [-]

        # Moon Positions as an array
        R_m = 384400/LU
        r_moon = np.array([[(1-mu) + R_m * np.cos(w_m * t + phase), 
                            R_m * np.sin(w_m * t + phase)] for t in t_eval])
        
        r_L2 = np.array([[(1-mu) + (moon_to_L2+R_m) * np.cos(w_m * t + phase), 
                            (moon_to_L2+R_m) * np.sin(w_m * t + phase)] for t in t_eval])
        
        moon_speed = 2 * np.pi * R_m / moon_earth_orbit
        v_moon = np.array([[moon_speed * -1 * np.sin(w_m*t + phase), moon_speed * np.cos(w_m*t + phase)] for t in t_eval])

        L2_speed = 2 * np.pi * (R_m+moon_to_L2) / moon_earth_orbit
        v_L2 = np.array([[L2_speed * -1 * np.sin(w_m*t + phase), L2_speed * np.cos(w_m*t + phase)] for t in t_eval])

        sc_loc = np.transpose(four_body_trajectory[0:2,:])
        sc_vel = np.transpose(four_body_trajectory[2:4,:])
    
        
        min_dist = 1
        min_dist_loc = None
        t_min_dis = None
        min_vel_norm = 1
        min_vel = ()
        best_i = None
        
        for i in range(np.shape(four_body_trajectory)[1]):
            t = t_eval[i]
            if tf - t < 1e-2 :
                break
            sc_state = four_body_trajectory[:,i]
            moon_loc = r_moon[i]
            L2_loc = r_L2[i]

            
            dist = np.sqrt((sc_state[0]-L2_loc[0])**2 + (sc_state[1]-L2_loc[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_dist_loc = L2_loc
                t_min_dis = t
                vel = np.array([sc_state[2]-v_L2[i,0], sc_state[3]-v_L2[i,1]])
                vel_norm = np.linalg.norm(vel)
                min_vel_norm = vel_norm
                min_vel = vel
                best_i = i



        if t_min_dis:
            min_dist_arr.append(min_dist)
            t_min_dist_arr.append(t_min_dis)
            

            min_vel_norm_arr.append(min_vel_norm)
            min_vel_arr.append(min_vel)
            

            if min_dist < .00001 and min_vel_norm < .023:
                # ax_dist.scatter(phase, min_dist, label=f"T:  {t_min_dis:.2f}")
                # ax_vel.scatter(phase, min_vel_norm, label="V-Dif: (%.3f, %.3f) T: %.2f" % (min_vel[0], min_vel[1], t_min_dis))
                ax_dis_vel.scatter(min_dist, min_vel_norm, label=f"{phase:.5f}")
            

        ax_compare.plot(four_body_trajectory[0], four_body_trajectory[1], label=f"4BP, Phase:  {phase:.2f}")

        
        
    print('L2 Velocity', v_L2[best_i])
    print('S/C Velocity', four_body_trajectory[:,best_i])
    print('Initial State:', four_body_trajectory[:,0])
    print('Initial Phase:', phase)

    ax_dist.set_xlabel("Phase")
    ax_dist.set_ylabel("Non-Dimensional Distance")
    ax_dist.legend()
    ax_dist.grid()
    ax_dist.title.set_text("Minimum Distance to Moon")

    ax_vel.set_xlabel("Phase")
    ax_vel.set_ylabel("Non-Dimensional Velocity")
    ax_vel.legend()
    ax_vel.grid()
    ax_vel.title.set_text("Velocity to Moon at Minimum Distance")

    
    
    ax_dis_vel.set_xlabel("Minimum Distance")
    ax_dis_vel.set_ylabel("Velocity at Minimum Distance")
    ax_dis_vel.legend()
    ax_dis_vel.grid()
    ax_dis_vel.title.set_text("Minimum Distance to L2 vs. Velocity at Minimum Distance. Label: Phase")
    
    
    # plot moon orbit
    R_m = 384400/LU# Orbital radius of the Moon about the Earth [-]
    
    circle = plt.Circle((1-mu, 0), R_m, color='gray', label='Moon Orbit', fill=False)
    ax_compare.add_artist(circle)

    circle = plt.Circle((1-mu, 0), R_m+moon_to_L2, color='pink', label='L2 Orbit', fill=False)
    ax_compare.add_artist(circle)

    ax_compare.scatter(min_dist_loc[0],min_dist_loc[1], s=4, color='black', label="Location of Minimum Distance from L2")

    ax_compare.axis('equal')
    ax_compare.set_xlabel("x")
    ax_compare.set_ylabel("y")
    ax_compare.legend()
    ax_compare.grid()
    ax_compare.title.set_text("Trajectory, X-Y Plane, Non-Dimensioinal Units")


    # animation
                                           
    t_vec_temp = t_eval
    LU = 1.49598E+8 # Earth-Sun distance, or 1 AU, [km]
    TU = 5022635 # seconds
    R_m = 384400 # Orbital radius of the Moon about the Earth [km]
    moon_earth_orbit = 2551442.8032
    w_m = 2*math.pi/moon_earth_orbit # Angular velocity of the Moon about the Earth [-]

    # Moon Positions as an array
    # moonmoonmoon = np.transpose(np.array([[(1-mu) + R_m / LU * np.cos(w_m *TU * t + best_phase ), 
    #                         R_m / LU * np.sin(w_m * TU* t + best_phase)] for t in t_vec_temp]))
    moonmoonmoon = np.transpose(np.array([[(1-mu) + R_m/LU * np.cos(w_m * t * TU + best_phase), 
                            R_m/LU * np.sin(w_m * t * TU + best_phase)] for t in t_eval]))
        

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlim([1 - 2E+6 / LU, 1 + 2E+6 / LU])
    ax.set_ylim([-2E+6 / LU, 2E+6 / LU])
    ax.set_xlabel('X (LU)')
    ax.set_ylabel('Y (LU)')
    ax.set_title('Trajectory Animation')

    # Create the plot elements that will be updated
    line_cr3bp, = ax.plot([], [], 'r--', label='CR3BP', zorder=1)
    line_bcr4bp, = ax.plot([], [], 'b', label='BCR4BP', zorder=0)
    point_cr3bp, = ax.plot([], [], 'ro', markersize=4)  # Red dot for CR3BP
    point_bcr4bp, = ax.plot([], [], 'bo', markersize=4) # Blue dot for BCR4BP
    point_moon, = ax.plot([], [], 'k.', markersize = 5) # Moon
    y_f = 384400 / 149597871 # Nd Moon radius
    circle2 = plt.Circle((1 - mu, 0), y_f, color='gray', linestyle='dashed', fill=False)
    ax.add_artist(circle2)
    circle3 = plt.Circle((1 - mu, 0), moon_to_L2 + 384400/ 149597871, color='black', linestyle='dashed', fill=False)
    ax.add_artist(circle3)
    ax.scatter(min_dist_loc[0],min_dist_loc[1], s=4, color='black', label="Location of Minimum Distance from L2")

    plt.legend()

    # Function to initialize the animation
    def init():
        line_cr3bp.set_data([], [])
        line_bcr4bp.set_data([], [])
        point_cr3bp.set_data([], [])
        point_bcr4bp.set_data([], [])
        point_moon.set_data([], [])
        return line_cr3bp, line_bcr4bp, point_cr3bp, point_bcr4bp, point_moon

    # Function to update the animation for each frame
    def animate(i):
        line_cr3bp.set_data(three_body_trajectory[0, :i], three_body_trajectory[1, :i])
        point_cr3bp.set_data(three_body_trajectory[0, i-1:i], three_body_trajectory[1, i-1:i])

        # Update BCR4BP trajectory (only up to its calculated length)
        j = min(i, len(fbp_integrator.t) - 1) 
        line_bcr4bp.set_data(four_body_trajectory[0, :j], four_body_trajectory[1, :j])
        point_bcr4bp.set_data(four_body_trajectory[0, j-1:j], four_body_trajectory[1, j-1:j])

        point_moon.set_data(moonmoonmoon[0, j-1:j], moonmoonmoon[1, j-1:j])
        return line_cr3bp, line_bcr4bp, point_cr3bp, point_bcr4bp,

    # Create the animation
    n_frames = 120
    frame_steps = [int(i * len(tbp_integrator.t)/n_frames) for i in range(n_frames)]
    ani = FuncAnimation(fig, animate, frames=frame_steps, init_func=init,
                        blit=True, interval=1, repeat=False)

    ani.save(f'animation.gif', writer='pillow', fps=60)

    


    ax2D_nd.legend()
    plt.show()


    # moon_earth_orbit = 2360591.78/TU  # non-dimensional
    # w_m = 2*np.pi/moon_earth_orbit # Angular velocity of the Moon about the Earth
    
    # t = np.linspace(0,tf,20)
    # moon_arc_x = (1-mu) + R_m * np.cos(w_m * t)
    # moon_arc_y = R_m * np.sin(w_m * t)
    # print('t', t)
    # print('moon_arc_x', moon_arc_x)
    # print('moon_arc_y', moon_arc_y)
    # # print('t', t)
    # # r_m = [(1-mu) + R_m * np.cos(w_m * t), R_m * np.sin(w_m * t)] 
    # # print('r_m', r_m)
    # ax_compare.plot(moon_arc_x, moon_arc_y, label="Moon Path")
    
    
    
    

