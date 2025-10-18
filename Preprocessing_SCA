import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB,
        0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4,
        0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71,
        0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
        0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6,
        0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB,
        0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45,
        0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44,
        0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A,
        0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49,
        0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
        0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 0xBA, 0x78, 0x25,
        0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E,
        0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1,
        0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB,
        0x16,
    ],
    dtype=np.uint32,
)


little_sbox = np.array([0xE, 0x4, 0xD, 0x1,
        0x2, 0xF, 0xB, 0x8,
        0x3, 0xA, 0x6, 0xC,
        0x5, 0x9, 0x0, 0x7])


def hamming_weight(x):
    return bin(x).count('1')

HW = np.vectorize(hamming_weight)

!pip install scalib
from scalib.metrics import SNR, Ttest, MTtest
import scalib.modeling
import scalib.attacks
import scalib.postprocessing

!pip install pyts
from pyts.decomposition import SingularSpectrumAnalysis


def linear_multivariate_dataset(n_d, n_b, sigma, n_leakages, A):
  size = 2**n_b
  classes = np.random.randint(0,size,size=(n_leakages, n_d))

  classesbis = deepcopy(classes)
  for d in range(1, n_d):
    classesbis[:, 0] = classes[:, 0] ^ classes[:, d]

  classesbis = ((classesbis[..., None] & (1 << np.arange(n_b)[::-1])) > 0).astype(int) # (l, d, n)
  ONES = np.ones((*classesbis.shape[:2], 1))  # shape (l, d, 1)

  classesbis = np.concatenate([classesbis, ONES], axis=2)  # shape (l, d, n_b + 1)
  leakages = np.einsum('ldn,n->ld', classesbis, A) + np.random.normal(loc=0, scale=sigma, size=(n_leakages, n_d))


  return leakages, classes

def multivariate_dataset(n_d, n_b, sigma, n_leakages, s_SBOX, s_MASKS, s_useless=0):
  size = 2**n_b
  classes  = np.random.randint(0,size,size=(n_leakages, n_d))

  zi_classes = deepcopy(classes[:, 0])
  for d in range(1, n_d):
    classes[:, 0] = classes[:, 0] ^ classes[:, d]

  leakages = HW(classes)
  leakages_SBOX = np.tile(leakages[:, :1], (1, s_SBOX))
  leakages_MASKS = np.tile(leakages[:, 1:], (1, s_MASKS))
  leakages = np.concatenate([leakages_SBOX, leakages_MASKS], axis=1).astype(float)
  if s_useless>0:
    leakages_useless = np.random.randint(0,size,size=(n_leakages, s_useless))
    leakages_useless = HW(leakages_useless)
    leakages = np.concatenate([leakages, leakages_useless], axis=1)
  leakages += np.random.normal(loc=0, scale=sigma, size=leakages.shape)
  classes[:, 0] = zi_classes

  return leakages, classes



def em_criterion(newparams, params, verbose=False):
  delta_A = np.linalg.norm(newparams[1] - params[1])
  if verbose:
    print(f'delta_A : {delta_A}')
  return delta_A

def standardization(leakages):
  mean = np.mean(leakages)
  std = np.std(leakages)
  return ((leakages - mean)/std), mean, std



def multivariate_gaussian_pdf_1V(leakages_pois, mean, var):
  epsilon = np.eye(var.shape[0], dtype=np.float64) * 1e-6
  #reg_var = var + epsilon
  det = np.linalg.det(var) #(1,)
  if(det <= 0):
    print('Det nul ou negatif')
    print('VOICI LA MATRICE DE COV')
    print(var)
    print("\n")
    print('\n voici la valeur du det')
    print(det)
    print('\n')
    raise ValueError("Determinant nul")

  inv = np.linalg.inv(var) #(s,s)

  #chol = np.linalg.cholesky(var)
  #all_inv_tmp = np.linalg.inv(chol)
  #print(all_inv_tmp.shape)
  #all_inv = np.matmul(np.transpose(all_inv_tmp, axes=(0, 2, 1)), all_inv_tmp)
  coefs = (2*np.pi)**var.shape[0] # puissance s
  coefs = 1/(np.sqrt(det*coefs))
  if(leakages_pois.ndim == 1):
    leakages_pois = leakages_pois[None, :]
  diff = leakages_pois[:, :] - mean[ : , :]
  exponent = -0.5 * np.einsum('li,ij,lj->l', diff, inv, diff)
  return coefs * np.exp(exponent)

from scipy.linalg import block_diag

def template_gaussian(n_d, n_b, leakages_pois, classes, version='1'):

  s = leakages_pois.shape[1]
  mean = np.zeros((2**n_b, s))
  var = np.zeros((s,s))
  copy_leakages = np.zeros((leakages_pois.shape))
  for z in range(2**n_b):
    m = np.mean(leakages_pois[classes==z], axis=0)
    mean[z] = m
    copy_leakages[classes == z] = leakages_pois[classes==z] - m
  var= np.cov(copy_leakages, rowvar=False)

  return [mean,var]

def p_l_z(n_d, n_b, leakages_pois, classes, params):
  probas = multivariate_gaussian_pdf_1V(leakages_pois=leakages_pois, mean=params[0][classes], var=params[1])
  return probas

def all_p_l_z(n_d, n_b, leakages_pois, params):
  all_probas = np.zeros((2**n_b, leakages_pois.shape[0]))
  for z in range(2**n_b):
    all_probas[z] = p_l_z(n_d=n_d, n_b=n_b, leakages_pois=leakages_pois, classes=z*np.ones((leakages_pois.shape[0],), dtype=int), params=params)
  return all_probas

def p_z_l(n_d,n_b,leakages_pois, params):
  res = all_p_l_z(n_d=n_d, n_b=n_b, leakages_pois=leakages_pois, params=params)
  return res / np.sum(res,axis=0)[None , :]


def IPR(eigvecs):
   eigvecs = eigvecs**4
   dispersion = np.sum(eigvecs, axis=1)
   return np.argsort(dispersion)

def ELV(eigvecs, EGV):
  eigvecs = eigvecs**2
  return np.argsort(np.max(eigvecs, axis=1)*EGV)[::-1]

def cumulative_ELV(eigvecs, EGV, treshold):
  # je manie eigvecs**2
  # je trie les samples de la plus grande à la plus petite
  # Je classe de celui qui a besoin du moins de samples à celui qui en a le plus besoin pour dépasser le treshold
  # je ressorts le classement

  #treshold = treshold * np.ones((eigvecs.shape[0],))
  eigvecs = eigvecs**2
  eigvecs = np.sort(eigvecs, axis=1)[:, ::-1]
  eigvecs = eigvecs * EGV[:, None]
  n_samples = 1
  has_stoped = np.zeros(eigvecs.shape[0])
  n_samples_to_stop = np.zeros(eigvecs.shape[0])
  while n_samples < eigvecs.shape[1] and not(np.all(has_stoped)):
    mask = (np.sum(eigvecs[:, :n_samples],axis=1) > treshold) &  (has_stoped==0)
    n_samples_to_stop[mask] = n_samples
    has_stoped[mask] = 1
    n_samples += 1
  n_samples_to_stop[has_stoped==0] = n_samples
  return np.argsort(n_samples_to_stop), n_samples_to_stop

def combined_ELV(eigvecs, EGV, treshold):
    eigvecs = eigvecs**2
    eigvecs = eigvecs * EGV[:, None]

    labels = np.tile(np.arange(eigvecs.shape[0]), ( eigvecs.shape[1], 1)).T
    labels = labels.reshape(-1)
    eigvecs = eigvecs.reshape(-1)

    index = np.argsort(eigvecs)[::-1]

    eigvecs = eigvecs[index]
    labels = labels[index]

    n_samples = 1
    treshold = np.sum(EGV)*treshold
    while(np.sum(eigvecs[:n_samples]) < treshold):
        n_samples += 1
    print(n_samples)
    return np.unique(labels[:n_samples])

def get_cumulative_ELV(eigvecs, EGV):
  eigvecs = eigvecs**2
  eigvecs = eigvecs * EGV[:, None]
  eigvecs = np.sort(eigvecs, axis=1)[:, ::-1]
  cumulative_elv = np.zeros((eigvecs.shape))
  for n_samples in range(eigvecs.shape[1]):
        cumulative_elv[:, n_samples] = np.sum(eigvecs[:, :n_samples], axis=1)
  return cumulative_elv


# KLDA
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
#K = polynomial_kernel(traces, degree=n_d, coef0=1)

def KLDA(n_d, n_b, traces, classes, regularizor, new_dimension):
  l = classes.shape[0]

  #K = rbf_kernel(traces, traces)
  K = polynomial_kernel(traces, traces, degree=n_d, coef0=0)
  #K = (traces @ traces.T)**n_d
  print(K.shape)
  M_T = np.sum(K,axis=1)/l
  print(f"shape de M_T : {M_T.shape}")
  M = np.zeros((l,l))
  N = np.zeros((l,l))

  for z in range(2**n_b):
    K_z = K[:, classes==z]
    print(f"shape de Kz : {K_z.shape}")
    lz = K_z.shape[1]
    Mz = np.sum(K_z, axis=1)/lz
    print(f"shape de Mz : {Mz.shape}")
    delta_M = Mz - M_T
    print(f"shape de delta_M : {delta_M.shape}")
    M += lz*np.matmul(delta_M[:, None], delta_M[None, :])
    N += K_z @ (np.eye(lz) - np.full((lz, lz), 1/lz)) @ K_z.T
    print(f"N shape : {N.shape} formula {K_z.shape} and {(K_z @ (np.eye(lz) - np.full((lz, lz), 1/lz)) @ K_z.T).shape}")

  N += regularizor*np.eye(l)

  if np.linalg.det(N) > 0:
      print("N est inversible")
  else:
      print("N n'est pas inversible")

  #S = np.linalg.solve(N, M)

  #eigvals, eigvecs = np.linalg.eig(S)
  eigvals, eigvecs = eigh(M, N)
  index = np.argsort(eigvals)[::-1]

  index = index[:new_dimension]

  eigvals = eigvals[index]
  eigvecs = eigvecs[:, index]
  eigvecs  = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)

  return eigvals, eigvecs


def KLDA_projection(n_d, n_b, old_traces,new_traces, classes, eigvecs):
  if new_traces.ndim == 1:
    new_traces = new_traces[None, :]
  K = polynomial_kernel(old_traces, new_traces, degree=n_d, coef0=0)
  #K = (old_traces @ new_traces.T)**n_d # (l,p)
  projected_traces = K.T @ eigvecs #(p,c)
  return projected_traces

def pearson_corr(X, y):
    # X: shape (l, s)
    # y: shape (l,)
    X_centered = X - X.mean(axis=0)           # centrer chaque colonne
    y_centered = y - y.mean()                 # centrer y

    numerator = (X_centered * y_centered[:, None]).sum(axis=0)
    denominator = np.sqrt((X_centered**2).sum(axis=0) * (y_centered**2).sum())

    corr = numerator / denominator
    return corr  # shape (s,)


