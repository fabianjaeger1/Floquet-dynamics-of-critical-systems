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
sns.set(style = st,palette = sns.color_palette("muted"), rc={'figure.figsize': (12,9)})

# np.set_printoptions(threshold=np.inf)
   
#sns.set(color_codes = True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})


from numpy import ndarray

class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T


L = 20

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

cycles=11

basis = [(x, y) for x in range(0, L) for y in range(0, L)]

def ind_to_coord(L, ind):
    x = ind % L
    y = ind // L
    return x, y
def coord_to_ind(L, x, y):
    return y * L + (x) % L

T_0 = -0.05
T_1 = 0.05

# T_0 = 0.95
# T_1 = 0.05

T_0 = T_0 * L
T_1 = T_1 * L

def fermions2D(L, envelope):
    N = L * L
    ham = np.zeros((N, N), dtype=float)
    for i in range(N):
        x = i % L
        y = i // L
        right = y * L + (x + 1) % L
        down = ((y + 1) % L) * L + x
        ham[i, right] = -envelope(x + 0.5) * envelope(y)
        ham[i, down] = -envelope(x) * envelope(y + 0.5)
        ham[right, i] = np.conjugate(ham[i, right])
        ham[down, i] = np.conjugate(ham[i, down])
    return ham


def F(x):
    return 2*np.sin(np.pi*x/L)**2

def ssd(x, L):
    return np.sin(np.pi * (x + 0.5) / L) *  np.sin(np.pi * (x + 0.5) / L)

hamSSD = fermions2D(L, lambda i: ssd(i, L))
ham0 = fermions2D(L, lambda i: 1)

def F(x,y):
    return 2*np.sin(np.pi*x/L)**2*np.sin(np.pi*y/L)**2

# Create Basis states

t_1 = -1
t_2 = -1


eigvals0, U = np.linalg.eigh(ham0)
eigvals1, V = np.linalg.eigh(hamSSD)

# eigvals0, eigvecs0 = np.linalg.eigh(ham0)
# idx0 = eigvals0.argsort()[::1]
# eigenValues0 = eigvals0[idx0]
# U = eigvecs0[:,idx0]

# eigvals1, eigvecs1 = la.eig(hamSSD)
# idx1 = eigvals1.argsort()[::1]
# eigenValues1 = eigvals1[idx1]
# V = eigvecs1[:,idx1]

U = np.matrix(U)
V = np.matrix(V)

exponential_E0 = np.exp(-1j*T_0*eigvals0)
exponential_E1 = np.exp(-1j*T_1*eigvals1)
# exponential_E0 = np.real(np.matrix(np.diag(exponential_E0)))
# exponential_E1 = np.real(np.matrix(np.diag(exponential_E1)))

exponential_E0 = np.matrix(np.diag(exponential_E0))
exponential_E1 = np.matrix(np.diag(exponential_E1))


W = V.H@U
mathcalW = U
W1_T1 = exponential_E1@W
W0_T0 = exponential_E0@(W.H)
W_total = W0_T0@W1_T1

E_density = np.zeros((L, L), dtype =complex)
state = U[:,:int(L**2/2)]@((U.H)[:int(L**2/2),:])


def obtainGroundStateEnergy(state,L):
    Ground_State_Energy_H0 = 0
    for i in range(0, L-1):
        for j in range(0, L-1):
            term1 = state[basis.index((i,j)),basis.index((i+1,j))]
            term2 = state[basis.index((i+1,j)), basis.index((i,j))]
            term3 = state[basis.index((i,j)),basis.index((i,j+1))]
            term4 = state[basis.index((i,j+1)),basis.index((i,j))]
            Ground_State_Energy_H0 += -term1 -term2 -term3 -term4
    return Ground_State_Energy_H0

# for j in range(0, L):
#     for i in range(0, L):
#             if i == L-1:
#                 term1 = state[basis.index((i,j)),basis.index((0,j))]
#                 term2 = state[basis.index((0,j)), basis.index((i,j))]
#             else:
#                 term1 = state[basis.index((i,j)),basis.index((i+1,j))]
#                 term2 = state[basis.index((i+1,j)), basis.index((i,j))]
#             if j == L-1:
#                 term3 = state[basis.index((i,j)),basis.index((i,0))]
#                 term4 = state[basis.index((i,0)),basis.index((i,j))]
#             else:
#                 term3 = state[basis.index((i,j)),basis.index((i,j+1))]
#                 term4 = state[basis.index((i,j+1)),basis.index((i,j))]
#             E_density[i,j] = t_1*term1 + t_1*term2 + t_2*term3 + t_2*term4
#             E_density = np.real(E_density)

for i in range(0,L):
    for j in range(0,L):
        if i == L-1 and j != L-1:
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, 0, j)]
            term2 = state[coord_to_ind(L, 0, j), coord_to_ind(L, i, j)]
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, j+1)]
            term4 = state[coord_to_ind(L, i, j+1), coord_to_ind(L, i, j)]
        elif j == L-1 and i != L-1:
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, 0)]
            term4 = state[coord_to_ind(L, i, 0), coord_to_ind(L, i, j)]
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, i+1, j)]
            term2 = state[coord_to_ind(L, i+1, j), coord_to_ind(L, i, j)]
        elif i == L-1 and j == L-1:
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, 0, j)]
            term2 = state[coord_to_ind(L, 0, j), coord_to_ind(L, i, j)]
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, 0)]
            term4 = state[coord_to_ind(L, i, 0), coord_to_ind(L, i, j)]
        else:
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, i+1, j)]
            term2 = state[coord_to_ind(L, i+1, j), coord_to_ind(L, i, j)]
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, j+1)]
            term4 = state[coord_to_ind(L, i, j+1), coord_to_ind(L, i, j)]
        E_density[i,j] = t_1*term1 + t_1*term2 + t_2*term3 + t_2*term4
        E_density = np.real(E_density)


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis', interpolation = 'nearest')
#     plt.imshow(arr, cmap='winter', interpolation = 'nearest')
#     plt.imshow(arr, cmap= 'magma', interpolation = 'nearest')
    plt.colorbar()
    plt.title( "Energy Density Heat Map" )
    plt.xlim(0, L-1)
    plt.ylim(0, L-1)
    plt.xlabel(r'Lattice Site $x$', fontsize = 18)
    plt.ylabel(r'Lattice Site $y$',fontsize = 18)
    plt.show()

heatmap2d(E_density)

E_density_cycles = np.zeros((L, L), dtype = complex)

for i in range(cycles - 1):
        mathcalW = mathcalW@W_total
        state = mathcalW[:,:int(L/2)]@((mathcalW.H)[:int(L/2),:])
for j in range(0, L):
    for i in range(0, L):
        if i == L-1 and j != L-1:
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, 0, j)]
            term2 = state[coord_to_ind(L, 0, j), coord_to_ind(L, i, j)]
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, j+1)]
            term4 = state[coord_to_ind(L, i, j+1), coord_to_ind(L, i, j)]
        elif j == L-1 and i != L-1:
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, 0)]
            term4 = state[coord_to_ind(L, i, 0), coord_to_ind(L, i, j)]
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, i+1, j)]
            term2 = state[coord_to_ind(L, i+1, j), coord_to_ind(L, i, j)]
        elif i == L-1 and j == L-1:
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, 0, j)]
            term2 = state[coord_to_ind(L, 0, j), coord_to_ind(L, i, j)]
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, 0)]
            term4 = state[coord_to_ind(L, i, 0), coord_to_ind(L, i, j)]
        else:
            term1 = state[coord_to_ind(L, i, j), coord_to_ind(L, i+1, j)]
            term2 = state[coord_to_ind(L, i+1, j), coord_to_ind(L, i, j)]
            term3 = state[coord_to_ind(L, i, j), coord_to_ind(L, i, j+1)]
            term4 = state[coord_to_ind(L, i, j+1), coord_to_ind(L, i, j)]
        E_density_cycles[i,j] = t_1*term1 + t_1*term2 + t_2*term3 + t_2*term4
        E_density_cycles = np.real(E_density_cycles)
        
# for j in range(0,L):
#     for i in range(0,L):
#         if i == L-1 and j != L-1:
#             term1 = state[basis.index((i,j)),basis.index((0,j))]
#             term2 = state[basis.index((0,j)), basis.index((i,j))]
#             term3 = state[basis.index((i,j)),basis.index((i,j+1))]
#             term4 = state[basis.index((i,j+1)),basis.index((i,j))]  
#         elif j == L-1 and i != L-1:
#             term3 = state[basis.index((i,j)),basis.index((i,0))]
#             term4 = state[basis.index((i,0)),basis.index((i,j))]
#             term1 = state[basis.index((i,j)),basis.index((i+1,j))]
#             term2 = state[basis.index((i+1,j)), basis.index((i,j))]
#         elif i == L-1 and j == L-1:
#             term1 = state[basis.index((i,j)),basis.index((0,j))]
#             term2 = state[basis.index((0,j)), basis.index((i,j))]
#             term3 = state[basis.index((i,j)),basis.index((i,0))]
#             term4 = state[basis.index((i,0)),basis.index((i,j))]
#         else:
#             term1 = state[basis.index((i,j)),basis.index((i+1,j))]
#             term2 = state[basis.index((i+1,j)), basis.index((i,j))]
#             term3 = state[basis.index((i,j)),basis.index((i,j+1))]
#             term4 = state[basis.index((i,j+1)),basis.index((i,j))]  
#         E_density_cycles[i,j] = t_1*term1 + t_1*term2 + t_2*term3 + t_2*term4
#         E_density_cycles = np.real(E_density_cycles)

heatmap2d(E_density_cycles)
plt.show()
