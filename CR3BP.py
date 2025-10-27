# =============================================================================
# File: CR3BP.py
# Author: Mia Tian
# Created: 7/2023
#
# Description: 
#     Implements the Circular Restricted Three-Body Problem (CR3BP) dynamics and
#     associated analysis tools for the Earth–Moon system or similar two-body
#     configurations. This script integrates the nonlinear equations of motion,
#     computes state transition matrices, evaluates stability via monodromy and
#     Jacobi constants, and visualizes orbital trajectories, manifolds, and
#     Poincaré maps.
#
#     Key capabilities include:
#         - Integration of CR3BP equations of motion using solve_ivp
#         - Linearized dynamics via A-matrix and STM propagation
#         - Computation of Jacobi constants and libration points
#         - Stability analysis using eigenvalue decomposition
#         - Generation of 2D/3D trajectory plots and Poincaré sections
#
#     This module can be used for analyzing periodic orbits, studying invariant
#     manifolds, and exploring stability characteristics in the CR3BP framework.
# =============================================================================

import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import fsolve
import math

from import_data import import_data

INITIAL_X = 0
INITIAL_Y = 1
INITIAL_Z = 2
INITIAL_VX = 3
INITIAL_VY = 4
INITIAL_VZ = 5
JACOBI = 6
PERIOD = 7
STABILITY = 8

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

def calculate_libration_pts():
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

def calculate_jacobi(traj):
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

    # ax.scatter([(1-mu)*LU], [0], [0], marker='o', color='pink', label="Moon")
    # ax.scatter([-mu*LU], [0], [0], marker='o', color='blue', label="Earth")

    # libration_pts = calculate_libration_pts()
    # libration_x = [coord[0] for coord in libration_pts]
    # libration_y = [coord[1] for coord in libration_pts]
    # ax.scatter(libration_x, libration_y, [0], marker='<', color='black', label='Libration Points')            


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
    ax2D.plot(trajectory_position[0], trajectory_position[1], label="Trajectory")
    ax2D.set_xlabel("x")
    ax2D.set_ylabel("y")
    plt.title("CR3BP Trajectory, X-Y Plane")

    ax2D_nd = plt.figure().add_subplot()
    ax2D_nd.plot(trajectory_position[0]/LU, trajectory_position[1]/LU, label="Trajectory")
    ax2D_nd.set_xlabel("x")
    ax2D_nd.set_ylabel("y")
    plt.title("CR3BP Trajectory, X-Y Plane, Non-Dimensioinal Units")

    
    
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
        if abs(ev1 - 1) < 1e-4:
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

def compute_manifold(eigenvalue_pair, trajectory, t_eval, ax2D, ax2D_nd, ax3D, LU):

    def event_unstable(t,y, mu):
        return y[0]
        return y[0]-1+mu
        val = y[0] * LU - 405000
        return val
    
    event_unstable.terminal = True
    event_unstable.direction = 1

    def event_stable(t,y, mu):
        return y[0]
        val = y[0] * LU - 405000
        return val
    
    event_stable.terminal = True
    event_stable.direction = -1


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

     

    stable_eig_val = stable_eig[0]
    stable_eiv_vec = stable_eig[1].reshape(6,1) * -1
    t_span_stable = (0, -10)
    t_eval_stable = np.linspace(t_span_stable[0], t_span_stable[1], 1000)

    unstable_eig_val = unstable_eig[0]
    unstable_eiv_vec = unstable_eig[1].reshape(6,1) * -1
    t_span_unstable = (0, 10)
    t_eval_unstable = np.linspace(t_span_unstable[0], t_span_unstable[1], 1000)
    for i in range(len(t_arr)):
        t = t_arr[i]
        m = manifold_initial[:,i]
        stm = stm_arr[:,i]
        stm = np.reshape(stm, (6, 6), order='F')
        perturbation = np.matmul(stm, stable_eiv_vec)
        perturbation = perturbation/LA.norm(perturbation) * 1e-5
        # perturbation = stable_eiv_vec/LA.norm(stable_eiv_vec) * 1e-5
        initial_x = (m + np.reshape(perturbation, (1,6))).flatten()

        integrator = solve_ivp(cr3bp_equations, t_span_stable, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval_stable, rtol=1e-13, atol=1e-14, events=event_stable)   
        trajectory = integrator.y
        ax2D_nd.plot(trajectory[0], trajectory[1], color='orange', linewidth=.4)
        ax2D.plot(trajectory[0]*LU, trajectory[1]*LU, color='orange', linewidth=.4)
        ax3D.plot(trajectory[0]*LU, trajectory[1]*LU, trajectory[2]*LU, color='orange', linewidth=.4)



        perturbation = np.matmul(stm, unstable_eiv_vec)
        perturbation = perturbation/LA.norm(perturbation) * 1e-5
        initial_x = (m + np.reshape(perturbation, (1,6))).flatten()

        integrator = solve_ivp(cr3bp_equations, t_span_unstable, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval_unstable, rtol=1e-13, atol=1e-14, events=event_unstable)   
        trajectory = integrator.y
        ax2D_nd.plot(trajectory[0], trajectory[1], color='green', linewidth=.4)
        ax2D.plot(trajectory[0]*LU, trajectory[1]*LU, color='green', linewidth=.4)
        ax3D.plot(trajectory[0]*LU, trajectory[1]*LU, trajectory[2]*LU, color='green', linewidth=.4)


    ax2D_nd.scatter(manifold_initial[0], manifold_initial[1], marker='.', color='black')
    ax2D.scatter(manifold_initial[0]*LU, manifold_initial[1]*LU, marker='.', color='black')
    ax3D.scatter(manifold_initial[0]*LU, manifold_initial[1]*LU, marker='.', color='black')  


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


def cr3bp(initial_state, mu, LU, TU, t_span):


    initial_stm = np.identity(6)
    initial_stm = np.reshape(initial_stm, 36, order='F')

    initial_x = np.concatenate([initial_state, initial_stm])

    print('initial_x', initial_x)
    t_eval = np.linspace(t_span[0], t_span[1], 12000)

    # Solve the ODE
    integrator = solve_ivp(ode_STM_cr3bp, t_span, initial_x, method='DOP853', args=(mu,), dense_output=True, t_eval=t_eval, rtol=1e-13, atol=1e-14)    
    trajectory = integrator.y

    jacobi = calculate_jacobi(trajectory)
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
    # print('sorted_eigenvalues', sorted_eigenvalues)
    # print('stable_eigs', stable_eigs)
    # print('unstable_eigs', unstable_eigs)

    compute_manifold(sorted_eigenvalues[0], trajectory, t_eval, ax2D, ax2D_nd, ax3D, LU)

    compute_poincare_1(mu)
    
    # compute_poincare(initial_state, jacobi, mu)

    plt.show()


if __name__ == "__main__":

    # params = {
    #     'sys': 'earth-moon',  
    #     'family': 'halo',
    #     'libr': '2',
    #     'branch': 'N'
    # }

    # data = import_data(params)

    # signature = data['signature']
    # system = data['system']
    # family = data['family']
    # libration_point = data['libration_point']
    # # branch = data['branch']
    # limits = data['limits']
    # count = data['count']
    # fields = data['fields']
    # data_list = data['data']

    # mu = float(system['mass_ratio'])
    # LU = float(system['lunit'])
    # TU = float(system['tunit'])
    
    # data_matrix = np.matrix(data_list)
    # data_matrix = data_matrix.astype(float)

    # initial_x = data_matrix[:,INITIAL_X]
    # initial_y = data_matrix[:,INITIAL_Y]
    # initial_z = data_matrix[:,INITIAL_Z]
    # initial_vx = data_matrix[:,INITIAL_VX]
    # initial_vy = data_matrix[:,INITIAL_VY]
    # initial_vz = data_matrix[:,INITIAL_VZ]
    # jacobi = data_matrix[:,JACOBI]
    # period = data_matrix[:,PERIOD]
    # stabiility = data_matrix[:,STABILITY]

    # index = 35

    # t_span = (0,period[index,0])
    # print('period',period[index,0])
    # initial_state = np.array([initial_x[index,0], initial_y[index,0], initial_z[index,0], initial_vx[index,0], initial_vy[index,0], initial_vz[index,0]])
    # cr3bp(initial_state, mu, LU, TU, t_span)


    # # plot parameters for the entire family
    # plt.figure(figsize=(10,6))
    # plt.scatter(list(jacobi), list(stabiility))
    # plt.xlabel("Jacobi Constant")
    # plt.ylabel("Stability Index")
    # plt.title('Jacobi vs Stability')
    # plt.show()

    # plt.figure(figsize=(10,6))
    # plt.scatter(list(jacobi), list(period))
    # plt.xlabel("Jacobi Constant")
    # plt.ylabel("Period")
    # plt.title('Jacobi vs Period')
    # plt.show()



    # Define parameters for Earth-Moon System
    mu = 0.01215058426954224
    LU = 389703
    TU = 382981 #seconds

    # Initial conditions: [x0, y0, z0, vx0, vy0, vz0]

    initial_state = np.array([0.857840719015296, 0, 0, 0, -0.1540301356477451, 0])

    # Time span for the integration
    t_span = (0, 2.759864712710722)

    cr3bp(initial_state, mu, LU, TU, t_span)
