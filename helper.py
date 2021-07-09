import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.integrate import complex_ode
import scipy as sp
import matplotlib
# from legacy import *
import copy
import math


def ssd(x, L):
    return 2 * np.sin(np.pi * (x + 1) / L) * np.sin(np.pi * (x + 1) / L)


def prepareState(vals, vecs, N, T):
    Wbar = np.matrix(vecs)

    if T != 0:
        #### State Preparation ####
        beta = 1 / T
        mu = 0

        distribution = 1 / (np.exp(beta * (vals - mu)) + 1)
        init_state = np.matrix(np.einsum("ik, kj, k -> ij", Wbar, Wbar.H, distribution))
    else:
        init_state = Wbar[:, :N] @ ((Wbar.H)[:N, :])

    return init_state


def fermions1D(L, envelope, hopping=[0, 1], pbc=False):
    ham = np.zeros((L, L), dtype=complex)
    for j, h in enumerate(hopping):
        for i in range(L - j):
            ham[i, i + j] = -envelope((i + i + j - 1) / 2) * h / 2
            ham[i + j, i] = np.conjugate(ham[i, i + j])
        if pbc:
            for i in range(L - j, L):
                ham[i, (i + j) % L] = -envelope((i + i + j - 1) / 2) * h / 2
                ham[(i + j) % L, i] = np.conjugate(ham[i, (i + j) % L])

    return ham


def ind_to_coord(L, ind):
    x = ind % L
    y = ind // L
    return x, y


def coord_to_ind(L, x, y):
    return y * L + (x) % L


def fermions2D(L, envelope):
    N = L * L
    ham = np.zeros((N, N), dtype=complex)
    for i in range(N):
        x = i % L
        y = i // L

        right = y * L + (x + 1) % L
        down = ((y + 1) % L) * L + x

        ham[i, right] = -envelope(x + 0.5) * envelope(y)
        ham[i, down] = -envelope(x) * envelope(y + 0.5)
        # if x % 2 == 0:
        #     ham[i, down] = -envelope(x) * envelope(y + 0.5)
        # else:
        #     ham[i, down] = -envelope(x) * envelope(y + 0.5)

        ham[right, i] = np.conjugate(ham[i, right])
        ham[down, i] = np.conjugate(ham[i, down])

    return ham


def to_majorana(fermionic_ham):
    shape = fermionic_ham.shape
    L = shape[0]

    maj_H = np.zeros((2 * L, 2 * L), dtype=complex)
    maj_H[0::2, 0::2] = fermionic_ham[:, :] / 4
    maj_H[0::2, 1::2] = -1j * fermionic_ham[:, :] / 4
    maj_H[1::2, 0::2] = 1j * fermionic_ham[:, :] / 4
    maj_H[1::2, 1::2] = fermionic_ham[:, :] / 4
    maj_H = np.matrix(maj_H)
    maj_H = 0.5 * (maj_H - maj_H.T)
    return maj_H


def to_majorana_state(fermion_state):
    L = fermion_state.shape[0]
    maj_state = np.zeros((2 * L, 2 * L), dtype=complex)
    maj_state[0::2, 0::2] = np.identity(L) - fermion_state + fermion_state.T
    maj_state[1::2, 1::2] = np.identity(L) - fermion_state + fermion_state.T
    maj_state[0::2, 1::2] = (
        -1j * np.identity(L) + 1j * fermion_state + 1j * fermion_state.T
    )
    maj_state[1::2, 0::2] = (
        1j * np.identity(L) - 1j * fermion_state - 1j * fermion_state.T
    )
    maj_state -= np.identity(2 * L)
    maj_state = np.matrix(maj_state)
    return maj_state


def to_fermion_state(majorana_state):
    L = int(majorana_state.shape[0] / 2)
    c_state = (
        (
            majorana_state[0::2, 0::2]
            + np.identity(L)
            + 1j * majorana_state[1::2, 0::2]
        ).H
    ) / 2
    return c_state


def get_dissipator(L, gl_1, gl_2, gr_1, gr_2):
    M = np.zeros((2 * L, 2 * L), dtype=complex)
    M[0, 0] = gl_2 + gl_1
    M[1, 1] = gl_2 + gl_1
    M[0, 1] = 1j * (gl_2 - gl_1)
    M[1, 0] = -1j * (gl_2 - gl_1)
    M[2 * L - 2, 2 * L - 2] = gr_2 + gr_1
    M[2 * L - 1, 2 * L - 1] = gr_2 + gr_1
    M[2 * L - 2, 2 * L - 1] = 1j * (gr_2 - gr_1)
    M[2 * L - 1, 2 * L - 2] = -1j * (gr_2 - gr_1)
    M = np.matrix(M)
    return M


def lindblad_stepper(maj_Hssd, maj_H, M, maj_state, T0, T1, dt):
    for _ in range(int(T1 / dt)):
        temp = (4j * maj_Hssd + 4 * M.real) @ maj_state
        delta = dt * (-temp + temp.T - 8j * M.imag)
        maj_state = maj_state + delta
    for _ in range(int(abs(T0) / dt)):
        temp = (4j * maj_H + 4 * M.real) @ maj_state
        delta = dt * (-temp + temp.T - 8j * M.imag)
        maj_state = maj_state + delta
    return maj_state


def lindblad_stepper_sp(maj_Hssd, maj_H, M, maj_state, T0, T1):
    L = int(maj_Hssd.shape[0])

    def odeFunc_uniform(t, y):
        ymat = np.matrix(y.reshape(L, L))
        temp = (4j * maj_H + 4 * M.real) @ ymat
        deriv = -temp + temp.T - 8j * M.imag
        return np.array(deriv).reshape(-1)

    def odeFunc_ssd(t, y):
        ymat = np.matrix(y.reshape(L, L))
        temp = (4j * maj_Hssd + 4 * M.real) @ ymat
        deriv = -temp + temp.T - 8j * M.imag
        return np.array(deriv).reshape(-1)

    y0 = np.array(maj_state).reshape(-1)
    ode = complex_ode(odeFunc_ssd)
    ode.set_initial_value(y0, 0)
    ode.integrate(ode.t + T1)

    y0 = ode.y
    ode = complex_ode(odeFunc_uniform)
    ode.set_initial_value(y0, 0)
    ode.integrate(ode.t + abs(T0))

    maj_state = np.matrix(ode.y.reshape(L, L))

    return maj_state


def lindblad_stepper(maj_Hssd, maj_H, M, maj_state, T0, T1):
    L = int(0.5 * maj_Hssd.shape[0])
    #### Evolve by 1 full period
    ## SSD
    A_ssd = 4j * maj_Hssd + 4 * M.real
    vals_ssd, vecs_ssd = np.linalg.eig(-A_ssd)
    expD_ssd = np.exp(abs(T1) * np.add.outer(vals_ssd, vals_ssd))
    intD_ssd = np.zeros((2 * L, 2 * L), dtype=complex)
    for i in range(2 * L):
        for j in range(2 * L):
            if np.abs(vals_ssd[i] + vals_ssd[j]) < 1e-10:
                intD_ssd[i, j] = abs(T1)
            else:
                intD_ssd[i, j] = (np.exp(abs(T1) * (vals_ssd[i] + vals_ssd[j])) - 1) / (
                    vals_ssd[i] + vals_ssd[j]
                )

    maj_state = (
        vecs_ssd
        @ (
            np.multiply(
                expD_ssd,
                np.linalg.inv(vecs_ssd)
                @ maj_state
                @ np.linalg.inv(vecs_ssd).transpose(),
            )
            + np.multiply(
                intD_ssd,
                np.linalg.inv(vecs_ssd)
                @ (-8j * M.imag)
                @ np.linalg.inv(vecs_ssd).transpose(),
            )
        )
        @ vecs_ssd.transpose()
    )

    ## Uniform
    A_uni = 4j * maj_H + 4 * M.real
    vals, vecs = np.linalg.eig(-A_uni)
    D = np.add.outer(vals, vals)
    expD = np.exp(abs(T0) * D)
    intD = np.zeros((2 * L, 2 * L), dtype=complex)
    for i in range(2 * L):
        for j in range(2 * L):
            if np.abs(vals[i] + vals[j]) < 1e-10:
                intD[i, j] = abs(T0)
            else:
                intD[i, j] = (np.exp(abs(T0) * (vals[i] + vals[j])) - 1) / (
                    vals[i] + vals[j]
                )
    maj_state = (
        vecs
        @ (
            np.multiply(
                expD,
                np.linalg.inv(vecs) @ maj_state @ np.linalg.inv(vecs).transpose(),
            )
            + np.multiply(
                intD,
                np.linalg.inv(vecs) @ (-8j * M.imag) @ np.linalg.inv(vecs).transpose(),
            )
        )
        @ vecs.transpose()
    )
    return maj_state


####### Observables #######


def energy(state, x):
    return -0.5 * (state[x, x + 1] + state[x + 1, x])


def particle_number(state, x):
    return state[x, x]


def entropy(state, subset, eps=1e-8):
    if len(subset) == 0:
        return 0
    w, v = LA.eigh(state[subset][:, subset])
    e_sum = 0
    for i in range(len(subset)):
        if w[i] > eps and w[i] < (1 - eps):
            e_sum = e_sum - (w[i] * np.log(w[i]) + (1 - w[i]) * np.log(1 - w[i]))
    return e_sum


def entropy_density(c_state, L):
    e_d = [entropy(c_state, list(range(x))) for x in range(0, L + 1)]
    return e_d


def mutual_information(state, A, B):
    return entropy(state, A) + entropy(state, B) - entropy(state, A + B)


def mutual_information_density(state, L, eps=1e-8):
    total_entrop = entropy(state, list(range(0, L)))
    e_d = [
        entropy(state, list(range(0, x)))
        + entropy(state, list(range(x, L)))
        - total_entrop
        for x in range(0, L + 1)
    ]
    return e_d


def energy_density(c_state, L):
    e_d = [energy(c_state, i) for i in range(L - 1)]
    return e_d


def particle_density(c_state, L):
    e_d = [particle_number(c_state, i) for i in range(L)]
    return e_d


def particle_current(c_state, L):
    e_d = [c_state[i, i + 1] - c_state[i, i - 1] for i in range(1, L - 1)]
    return e_d

##### Wrapper #####
def openSystem(
    L,
    T0,
    T1,
    cycles,
    dissipator,
    ham0,
    ham1,
    init_state=None,
    observable=None,
    use_scipy=False,
    pbc=False,
):
    measurements = []

    ########################################
    ##### Construct Majoranna Matrices #####
    ########################################
    maj_H0 = to_majorana(np.sign(T0) * ham0)
    maj_H1 = to_majorana(np.sign(T1) * ham1)

    #################################################
    ##### Obtain the initial correlation matrix #####
    #################################################
    if init_state is None:
        vals, U = np.linalg.eigh(ham0)
        Wbar = np.matrix(U)
        init_state = Wbar[:, : int(L / 2)] @ ((Wbar.H)[: int(L / 2), :])
    maj_state = to_majorana_state(init_state)

    if observable is not None:
        measurements.append(observable(to_fermion_state(maj_state), L))

    #####################
    ##### Main Loop #####
    #####################
    peak_found = False
    if use_scipy:
        for i in range(cycles):
            #### Evolve by 1 full period
            maj_state = lindblad_stepper_sp(
                maj_H1, maj_H0, dissipator, maj_state, T0, T1
            )
            if observable is not None:
                measurements.append(observable(to_fermion_state(maj_state), L))
    else:
        #print("Using Xueda-like approach")

        A_ssd = 4j * maj_H1 + 4 * dissipator.real
        vals_ssd, vecs_ssd = np.linalg.eig(-A_ssd)
        expD_ssd = np.exp(abs(T1) * np.add.outer(vals_ssd, vals_ssd))
        intD_ssd = np.zeros((2 * L, 2 * L), dtype=complex)
        for i in range(2 * L):
            for j in range(2 * L):
                if np.abs(vals_ssd[i] + vals_ssd[j]) < 1e-10:
                    intD_ssd[i, j] = abs(T1)
                else:
                    intD_ssd[i, j] = (
                        np.exp(abs(T1) * (vals_ssd[i] + vals_ssd[j])) - 1
                    ) / (vals_ssd[i] + vals_ssd[j])
        A_uni = 4j * maj_H0 + 4 * dissipator.real
        vals, vecs = np.linalg.eig(-A_uni)
        D = np.add.outer(vals, vals)
        expD = np.exp(abs(T0) * D)
        intD = np.zeros((2 * L, 2 * L), dtype=complex)
        for i in range(2 * L):
            for j in range(2 * L):
                if np.abs(vals[i] + vals[j]) < 1e-10:
                    intD[i, j] = abs(T0)
                else:
                    intD[i, j] = (np.exp(abs(T0) * (vals[i] + vals[j])) - 1) / (
                        vals[i] + vals[j]
                    )
        for i in range(cycles):
            #### Evolve by 1 full period
            # maj_state = lindblad_stepper(maj_H1, maj_H0, dissipator, maj_state, T0, T1)
            maj_state = (
                vecs_ssd
                @ (
                    np.multiply(
                        expD_ssd,
                        np.linalg.inv(vecs_ssd)
                        @ maj_state
                        @ np.linalg.inv(vecs_ssd).transpose(),
                    )
                    + np.multiply(
                        intD_ssd,
                        np.linalg.inv(vecs_ssd)
                        @ (-8j * dissipator.imag)
                        @ np.linalg.inv(vecs_ssd).transpose(),
                    )
                )
                @ vecs_ssd.transpose()
            )

            maj_state = (
                vecs
                @ (
                    np.multiply(
                        expD,
                        np.linalg.inv(vecs)
                        @ maj_state
                        @ np.linalg.inv(vecs).transpose(),
                    )
                    + np.multiply(
                        intD,
                        np.linalg.inv(vecs)
                        @ (-8j * dissipator.imag)
                        @ np.linalg.inv(vecs).transpose(),
                    )
                )
                @ vecs.transpose()
            )

            #### Measurements
            if observable is not None:
                measurements.append(observable(to_fermion_state(maj_state), L))

            # if i > 0 and measurements[-1] < measurements[-2]:
            #     peak_found = True
            # if peak_found and measurements[-1] > measurements[-2]:
            #     measurements.pop()
            #     break

    return measurements


def closeSystem(
    L,
    T0,
    T1,
    cycles,
    ham0,
    ham1,
    temp=0,
    mu=None,
    observable=None,
    use_scipy=False,
    pbc=False,
):
    measurements = []

    ########################################
    ##### Diagonalise the Hamiltonians #####
    ########################################
    vals, U = np.linalg.eigh(ham0)
    vals_ssd, V = np.linalg.eigh(ham1)

    Egs = np.sum(vals[: int(L / 2)])
    E0 = np.exp(-1j * T0 * vals)
    E1 = np.exp(-1j * T1 * vals_ssd)
    E0 = np.matrix(np.diag(E0))
    E1 = np.matrix(np.diag(E1))

    U = np.matrix(U)
    V = np.matrix(V)
    W = V.H @ U
    W1 = E1 @ W
    W0 = E0 @ (W.H)
    Wtotal = W0 @ W1
    Wbar = U

    #################################################
    ##### Obtain the initial correlation matrix #####
    #################################################
    if temp == 0:
        distribution = np.zeros(shape=(int(L)))
        if mu == None:
            distribution[: int(L / 2)] = 1
        else:
            npar = np.where(vals < mu)[0].shape[0]
            distribution[:npar] = 1
    else:
        distribution = 1 / (np.exp(beta * (vals - mu)) + 1)

    state = np.matrix(np.einsum("ik, kj, k -> ij", Wbar, Wbar.H, distribution))

    if observable is not None:
        measurements.append(observable(state, L))

    #####################
    ##### Main Loop #####
    #####################
    peak_found = False
    for i in range(cycles):
        Wbar = Wbar @ Wtotal
        # state = Wbar[:, : int(L / 2)] @ ((Wbar.H)[: int(L / 2), :])
        state = np.matrix(np.einsum("ik, kj, k -> ij", Wbar, Wbar.H, distribution))
        if observable is not None:
            measurements.append(observable(state, L))

    return measurements


#### CFT

phi = (1 + np.sqrt(5)) / 2


def gamma(n):
    return n % 2
    # return math.floor((n + 1) * phi) - math.floor(n * phi) - 1


def cft_M(T0, T1, L, n):
    if n == 0:
        return np.matrix(
            [
                [1 + np.pi * T1 / L, -np.pi * T1 / L],
                [np.pi * T1 / L, 1 - np.pi * T1 / L],
            ]
        )
    elif n == 1:
        return np.matrix([[np.exp(np.pi * T0 / L), 0], [0, np.exp(-np.pi * T0 / L)]])
    else:
        return cft_M(T0, T1, L, int(gamma(n))) @ cft_M(T0, T1, L, n - 1)


def cft_energy(T0, T1, L, n):
    mat = cft_M(1j * T0, 1j * T1, L, n)
    return (2 * np.pi / L) * (1 / 4) * (mat[0, 0] * mat[1, 1] + mat[0, 1] * mat[1, 0])
