'''
!!!!This code does NOT work as intended. Yet.!!!!

It supposed to be an implementation of the Segalman method [1],
using modal analysis information obtained using Nastran. 

Please note that this is not the last revision of the method, 
the latest work is [2].

[1] @link http://www.osti.gov/scitech/biblio/573295
[2] @link http://prod.sandia.gov/techlib/access-control.cgi/2013/133429.pdf
'''
import numpy as np
from pyNastran.op2.op2 import OP2

# input parameters
m = 130                 # mass of the body
PSD = 10 * (9.81) ** 2  # constant PSD in frequency [g^2/Hz=>(m/s^2)^2/Hz]
(f1, f2) = (10, 2000)   # frequency range
N = 1000                # number of integration intervals
damping = 0.02          # critical damping
excitation = 2          # 0-X, 1-Y, 2-Z direction

# open and read OP2
op2Obj = OP2('model-femap-000.op2', debug=False, log=None)
op2Obj.readOP2()


# 1. Get eigenfrequencies
eigenvectors = op2Obj.eigenvectors[1]
eigenvalues = np.array(eigenvectors.eigenvalues())
eigenfrequency = np.sqrt(eigenvalues)  # circular frequency [rad/s]


# 2. Generate modal coordinates matrix
modal_coordinates = []
translations = eigenvectors.translations
modes = sorted(translations.keys())
nids = sorted(translations[1].keys())
for mode in modes:
    for nid in nids:
        modal_coordinates.extend(translations[mode][nid])
# Transform list into the matrix
#   rows - modes
#   columns - dofs [only translations are used]
modal_coordinates = np.array(
    modal_coordinates).reshape(len(modes), len(nids) * 3)


# 2. Generate the dictionary of stress matrices for each element
# [element centre values]
sigma_per_elem = dict()
stress = op2Obj.solidStress[1]
elems = sorted(stress.oxx[1].keys())
for elem in elems:
    sigma_per_elem[elem] = dict()
    for mode in modes:
        s11 = stress.oxx[mode][elem]['C']
        s22 = stress.oyy[mode][elem]['C']
        s33 = stress.ozz[mode][elem]['C']
        s12 = stress.txy[mode][elem]['C']
        s23 = stress.tyz[mode][elem]['C']
        s13 = stress.txz[mode][elem]['C']
        sigma_per_elem[elem][mode] = np.array((s11, s22, s33, s12, s23, s13))


# 3. generate cross-mode stress matrix
A = np.array([[   1,-0.5, -0.5, 0, 0, 0],
              [-0.5,   1, -0.5, 0, 0, 0],
              [-0.5,-0.5,    1, 0, 0, 0],
              [   0,   0,    0, 3, 0, 0],
              [   0,   0,    0, 0, 3, 0],
              [   0,   0,    0, 0, 0, 3]])

T_per_elem = dict()
for elem in elems:
    S = sigma_per_elem[elem]
    T = np.zeros((len(modes), len(modes)))
    for i, mode1 in enumerate(modes):
        for j, mode2 in enumerate(modes):
            # sigmaT * A * sigma // numpy does transpose automatically
            T[i, j] = np.dot(np.dot(S[mode1], A), S[mode2])
    T_per_elem[elem] = T


# 4. Pre-calculate frequency dependent coefficients
omega_1 = f1 * 2 * np.pi  # Hz -> rad/s
omega_2 = f2 * 2 * np.pi  # Hz -> rad/s
delta_omega = (omega_2 - omega_1) / N
gamma = np.zeros((len(modes), len(modes)))
D = np.zeros((len(modes), len(modes)))
for i, mode1 in enumerate(modes):
    for j, mode2 in enumerate(modes):
        omega_i = eigenfrequency[i]
        omega_j = eigenfrequency[j]
        for n in range(N):
            omega = (omega_1 + n * delta_omega)
            Di = 1.0 / (omega_i**2 - omega**2 + complex(0, 2*damping*omega*omega_i))
            Dj = 1.0 / (omega_j**2 - omega**2 + complex(0, 2*damping*omega*omega_j))
            D[i, j] += (Di.conjugate() * Dj).real * delta_omega


# 5. Calculate gamma matrix
for i, mode1 in enumerate(modes):
    for j, mode2 in enumerate(modes):
        for a in range(len(nids) * 3):
            for b in range(len(nids) * 3):
                # (a % 3 = 0) => excitation in X
                # (a % 3 = 1) => excitation in Y
                # (a % 3 = 2) => excitation in Z
                if (a % 3 == excitation) and (b % 3 == excitation):
                    # PSD is provided in [g^2/Hz] -> 1/pi factor shall be
                    # present
                    phi_ab = modal_coordinates[i, a] * modal_coordinates[j, b]
                    gamma[i, j] += phi_ab * m * PSD * D[i, j] / np.pi


# 6. Calculate element stresses
excitation_labels = ['X', 'Y', 'Z']
print('Excitation in', excitation_labels[excitation], 'direction.')
for elem in elems:
    # element-wise multiplication of Gamma and T
    S = np.multiply(gamma, T_per_elem[elem])
    # transform matrix to the vector in order to execute fast summation
    S_vector = S.reshape(1, len(modes) ** 2)
    rms_von_mises = np.sqrt(np.sum(S_vector))
    print('Elem #', elem, ' => ', rms_von_mises, 'MPa')
