import numpy as np
# from qutip import *
from scipy.sparse import diags
import scipy.linalg as la
import math as math
import matplotlib.pyplot as plt
import timeit
import seaborn as sns
from matplotlib import ticker
import matplotlib.patches as mpatches

st = sns.axes_style("ticks")
sns.set(style=st, palette=sns.color_palette("muted"), rc={'figure.figsize': (12, 9)})

# sns.set(color_codes = True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

from numpy import ndarray


class myarray(ndarray):
    @property
    def H(self):
        return self.conj().T


def EnergyDensityCorrect(L, cycles, T_0, T_1, pbc):
    T_0 = T_0 * L
    T_1 = T_1 * L

    def F(x):
        return 2 * np.sin(np.pi * x / L) ** 2

    def energy(state):
        energy = 0
        for i in range(0, L - 1):
            energy = state[i, i + 1] + state[i + 1, i]
        return energy

    H_0 = np.zeros((L, L))
    H_1 = np.zeros((L, L))
    for i in range(L - 1):
        H_0[i, i + 1] = 1 / 2
        H_0[i + 1, i] = 1 / 2
        H_1[i, i + 1] = F(i) / 2
        H_1[i + 1, i] = F(i) / 2
    if pbc == True:
        H_0[0, L - 1] = 1 / 2
        H_0[L - 1, 0] = 1 / 2
        H_1[0, L - 1] = F(0) / 2
        H_1[L - 1, 0] = F(L) / 2
    eigenvalues0, U = np.linalg.eigh(H_0)
    eigenvalues1, V = np.linalg.eigh(H_1)
    U = np.matrix(U)
    V = np.matrix(V)

    exponential_E0 = np.exp(-1j * T_0 * eigenvalues0)
    exponential_E1 = np.exp(-1j * T_1 * eigenvalues1)
    exponential_E0 = np.matrix(np.diag(exponential_E0))
    exponential_E1 = np.matrix(np.diag(exponential_E1))

    W = V.H @ U
    mathcalW = U
    W1_T1 = exponential_E1 @ W
    W0_T0 = exponential_E0 @ (W.H)
    W_total = W0_T0 @ W1_T1
    E_density = []
    state = U[:, :int(L / 2)] @ ((U.H)[:int(L / 2), :])
    print(state.shape)
    E_density.append([1 / 2 * (state[i, i + 1] + state[i + 1, i])
                      for i in range(L - 1)])
    for i in range(cycles - 1):
        mathcalW = mathcalW @ W_total
        state = mathcalW[:, :int(L / 2)] @ ((mathcalW.H)[:int(L / 2), :])
        E_density.append([1 / 2 * (state[i, i + 1] + state[i + 1, i])
                          for i in range(L - 1)])
    E_density = np.real(np.array(E_density))
    i = 1
    while j in range(cycles - 1):
        plt.plot(E_density[cycles - j] - E_density[0], label="{}-cycles".format(cycles - j))
        j += 3

    #     plt.plot(E_density[cycles-1] - E_density[0], label="A")
    #     plt.plot(E_density[cycles-5] - E_density[0], label="B")

    #     plt.plot(E_density[cycles-10] - E_density[0], label="C")

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    text = plt.gca().yaxis.get_offset_text()
    text.set_size(18)

    plt.xlabel(r'Lattice Site $x$', fontsize=18)
    plt.ylabel(r'$E(x,n)$', fontsize=18)
    plt.tight_layout()
    plt.ticklabel_format(axis='y', scilimits=[0, 0])
    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.legend(fontsize=17)

    plt.savefig('Energy_Density1.pdf', bbox_inches='tight')

EnergyDensityCorrect(1000,20,0.5,0.95,False)
