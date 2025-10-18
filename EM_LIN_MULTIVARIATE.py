import numpy as np
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

def linear_multivariate_dataset(n_d, n_b, sigma, n_leakages, A):
  size = 2**n_b
  classes = np.random.randint(0,size,size=(n_leakages, n_d))

  classesbis = deepcopy(classes)
  for d in range(1, n_d):
    classesbis[:, 0] = classes[:, 0] ^ classes[:, d]

  classesbis = ((classesbis[..., None] & (1 << np.arange(n_b)[::-1])) > 0).astype(int) # (l, d, n)
  ONES = np.ones((*classesbis.shape[:2], 1))  # shape (l, d, 1)

  # Concaténation sur l’axe des bits
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

####################################################################################################################################
####################################################################################################################################
                                                    #INITIALISATION
####################################################################################################################################
####################################################################################################################################

from scipy.stats import norm
from scipy.linalg import block_diag
# Désormais on manipule 2*n_b + 1 bits
def init_ab_nind(n_d, n_b, s_SBOX, s_MASKS, sigma_ab = 0, leakages=None, sigma=1, noise_known = False, covvar=None):
  if noise_known:
    A = []
    if covvar is not None:
      for i in range(s_SBOX):
        leakages_s  = leakages[:, i]
        temp = (np.var(leakages_s) - (covvar[i]))
        if (temp >0):
          a = np.sqrt(temp*4 / (n_b))
        else:
          a = 0
        b = np.mean(leakages_s) - a*(n_b)/2

        a1 = np.full((n_b,), a)
        a2 = np.full((n_b*(n_d-1),), 0)
        a = np.concatenate([a1, a2], axis=0)
        b = np.array([b])
        ab = np.hstack([a,b])
        A.append(ab)
      for index in range(1,n_d):

        for i in range(s_SBOX+((index-1)*s_MASKS), s_SBOX + (index)*s_MASKS):
          leakages_s  = leakages[:, i]
          temp = (np.var(leakages_s) - (covvar[i]))
          if (temp >0):
            a = np.sqrt(temp*4 / (n_b))
          else:
            a = 0
          b = np.mean(leakages_s) - a*(n_b)/2
          a1 = np.full((n_b,), 0)
          a = np.full((n_b,), a)
          for l in range(index):
            a = np.concatenate([a1, a], axis=0)
          for l in range(index+1, n_d):
            a = np.concatenate([a,a1], axis=0)
          b = np.array([b])
          ab = np.hstack([a,b])
          A.append(ab)
    else:
      for i in range(s_SBOX):
        leakages_s  = leakages[:, i]
        temp = (np.var(leakages_s) - (sigma)**2)
        if (temp >0):
          a = np.sqrt(temp*4 / (n_b))
        else:
          a = 0
        b = np.mean(leakages_s) - a*(n_b)/2

        a1 = np.full((n_b,), a)
        a2 = np.full((n_b*(n_d-1),), 0)
        a = np.concatenate([a1, a2], axis=0)
        b = np.array([b])
        ab = np.hstack([a,b])
        A.append(ab)
      for index in range(1, n_d):
        for i in range(s_SBOX + (index-1)*s_MASKS, s_SBOX + index*s_MASKS):
          leakages_s  = leakages[:, i]
          a = np.sqrt((np.var(leakages_s) - (sigma)**2)*4 / (n_b))
          b = np.mean(leakages_s) - a*(n_b)/2

          a1 = np.full((n_b,), 0)
          a = np.full((n_b,), a)
          for l in range(index):
            a = np.concatenate([a1, a], axis=0)
          for l in range(index+1, n_d):
            a = np.concatenate([a, a1], axis=0)
          b = np.array([b])
          ab = np.hstack([a,b])
          A.append(ab)
    A = np.array(A)
  else:
    a = np.full((n_d*n_b,), 1)
    b = np.array([0])
    ab = np.hstack([a,b])
    A = [ab.copy() for _ in range(s)]
    A = np.array(A)
  if sigma_ab>0:
    A += np.random.normal(loc=0, scale=sigma_ab, size=(s_SBOX+s_MASKS, n_d*n_b+1))
  return A #(s , n)


def init_ab_error(n_d, n_b, s, sigma_ab = 0, leakages=None, sigma=1, noise_known = False):
  if noise_known:
    A = []
    for i in range(s):
      leakages_s  = leakages[:, i]
      temp = (np.var(leakages_s) - (sigma)**2)
      if (temp >0):
        a = np.sqrt(temp*4 / (n_b))
      else:
        a = 0
      b = np.mean(leakages_s) - a*(n_b*n_d)/2
      #a = (np.mean(leakages_s)-b) * (2/(n_d*n_b)) # La moyenne des AX vaut moyenne des leakages après calcul on a a*n_d*n_b/2 + b = mean(leakages) d'où a = (mean -b) * 2 / (n_b * n_d)
      a1 = np.full((n_b,), a)
      a2 = np.full((n_b,), a)
      a = np.concatenate([a1, a2], axis=0)
      b = np.array([b])
      ab = np.hstack([a,b])
      A.append(ab)
    A = np.array(A)
  else:
    a = np.full((n_d*n_b,), 1)
    b = np.array([0])
    ab = np.hstack([a,b])
    A = [ab.copy() for _ in range(s)]
    A = np.array(A)
  if sigma_ab>0:
    A += np.random.normal(loc=0, scale=sigma_ab, size=(s_SBOX+s_MASKS, n_d*n_b+1))
  return A #(s , n)


def init_ab_ind(n_d, n_b, sigma_ab = 0, leakages=None,  sigma=1, noise_known=False, covvar=None):

  s = 1 if leakages.ndim==1 else leakages.shape[1]
  if noise_known:
    A = []
    if covvar is not None:
      for i in range(s):
        leakages_s = leakages[:, i]
        #p = 0.01**(1/leakages_s.shape[0])
        #b = min(leakages_s) + sigma* norm.ppf(p)
        #a = (np.mean(leakages_s)-b) * (2/n_b)
        temp = (np.var(leakages_s) - (covvar[i]))
        if (temp >0):
          a = np.sqrt(temp*4 / (n_b))
        else:
          a = 0
        b = np.mean(leakages_s) - a*n_b/2
        a = np.full((n_b,), a)
        b = np.array([b])
        ab = np.hstack([a, b])
        A.append(ab)
    else:
      for i in range(s):
        leakages_s = leakages[:, i]
        #p = 0.01**(1/leakages_s.shape[0])
        #b = min(leakages_s) + sigma* norm.ppf(p)
        #a = (np.mean(leakages_s)-b) * (2/n_b)
        temp = (np.var(leakages_s) - (sigma)**2)
        if (temp >0):
          a = np.sqrt(temp*4 / (n_b))
        else:
          a = 0
        b = np.mean(leakages_s) - a*n_b/2
        a = np.full((n_b,), a)
        b = np.array([b])
        ab = np.hstack([a, b])
        A.append(ab)
    A = np.array(A)
  else:
    a = np.full((n_b,), 1)
    b = np.array([0])
    ab = np.hstack([a,b])
    A = [ab.copy() for _ in range(s)]
    A = np.array(A)
  if sigma_ab > 0:
    A += np.random.normal(loc=0, scale=sigma_ab, size=(s, n_b+1))
  return A


def init_V(n_d, n_b, s, sigma=1, covvar=None):
  if covvar is not None:
    return np.diag(covvar)
  return sigma**2*np.eye(s)

def lin_nind_init(n_d, n_b, leakages_pois, s_SBOX, s_MASKS, sigma=1, sigma_ab=0, noise_known=False, covvar=None):
  s = leakages_pois.shape[1]
  n = n_d*n_b+1
  size = 2**n_b
  n_MASKS = size**(n_d-1)
  a_j = np.full((n_MASKS,), 1/n_MASKS)
  A = init_ab_nind(n_d=n_d, n_b=n_b, s_SBOX=s_SBOX, s_MASKS=s_MASKS, sigma_ab=sigma_ab, leakages=leakages_pois, sigma=sigma, noise_known=noise_known, covvar=covvar)
  V = init_V(n_d=n_d, n_b=n_b, s=s, sigma=sigma, covvar=covvar)
  return [(a_j, A, V)]

def lin_nind_init_error(n_d, n_b, leakages_pois, s_SBOX, s_MASKS, sigma=1, sigma_ab=0, noise_known=False, covvar=None):
  s = leakages_pois.shape[1]
  n = n_d*n_b+1
  size = 2**n_b
  n_MASKS = size**(n_d-1)
  a_j = np.full((n_MASKS,), 1/n_MASKS)
  A = init_ab_error(n_d=n_d, n_b=n_b, s=s, sigma_ab=sigma_ab, leakages=leakages_pois, sigma=sigma, noise_known=noise_known)
  V = init_V(n_d=n_d, n_b=n_b, s=s, sigma=sigma, covvar=covvar)
  return [(a_j, A, V)]


def lin_ind_init(n_d, n_b, l_leakages_pois,sigma=1, sigma_ab=0, noise_known=False, covvar=None):
  params = []
  for l in range(len(l_leakages_pois)):
    leakages_pois = l_leakages_pois[l]
    leakages_pois = leakages_pois[:, None] if leakages_pois.ndim==1 else leakages_pois
    s = leakages_pois.shape[1]
    n = n_b+1
    size = 2**n_b
    n_MASKS = size**(n_d-1)
    a_j = np.full((n_MASKS,), 1/n_MASKS)
    if covvar is not None:
      A = init_ab_ind(n_d=n_d, n_b=n_b, sigma_ab=sigma_ab, leakages=leakages_pois, sigma=sigma, noise_known=noise_known, covvar=covvar[l])
    else:
      A = init_ab_ind(n_d=n_d, n_b=n_b, sigma_ab=sigma_ab, leakages=leakages_pois, sigma=sigma, noise_known=noise_known)
    if  A.ndim==1:
      A = A[:, None]
    if covvar is not None:
      V = init_V(n_d=n_d, n_b=n_b, s=s, sigma=sigma, covvar=covvar[l])
    else:
      V = init_V(n_d=n_d, n_b=n_b, s=s, sigma=sigma)
    params.append((a_j, A, V))
  return params

# Dans le cas ind on aura une matrice de taille s1 + s2 * 2n
def recombination(n_d, n_b, params1, params2):
  size = 2**n_b
  n_MASKS = size**(n_d-1)
  a_j = np.full((n_MASKS,), 1/n_MASKS)
  A1 = params1[0][1] # matrice de taille s1 n1
  A2 = params2[0][1] # matrice de taille s2 n2
  if (A1.ndim==1):
    A1 = A1[None, :]
  if (A2.ndim==1):
    A2 = A2[None, :]
  _A1 = np.concatenate([A1, np.zeros((A1.shape[0], A2.shape[1]))], axis=1)
  _A2 = np.concatenate([np.zeros((A2.shape[0], A1.shape[1])), A2], axis=1)
  A = np.concatenate([_A1,_A2], axis=0)

  V1 = params1[0][2]
  V2 = params2[0][2]

  V = block_diag(V1, V2)

  return [(a_j, A, V)]

def get_full_params(n_d, n_b, params):
  new_params = [params[0]]
  for n in range(1,len(params)):
    new_params = recombination(n_d=n_d, n_b=n_b, params1=new_params, params2 = [params[n]])
  return new_params


def get_original_params(n_d, n_b, sigma, s_SBOX, s_MASKS, mean_profiling, std_profiling,  version='nind'):
  n=n_b+1
  size = 2**n_b
  n_MASKS = size**(n_d-1)
  a_j = np.full((n_MASKS,), 1/n_MASKS)
  if version == "nind":
      A = np.ones((s_SBOX, n_b)) * (1/std_profiling)
      A = np.concatenate([A,  np.zeros((s_SBOX,(n_d-1)*n_b)), np.zeros((s_SBOX,1))], axis=1)
      for index in range(1,n_d):
          A1 = np.ones((s_MASKS, n_b)) * (1/std_profiling)
          for l in range(0, index):
              A1 = np.concatenate([np.zeros((s_SBOX,n_b)), A1], axis=1)
          for l in range(index+1, n_d):
              A1 = np.concatenate([ A1, np.zeros((s_SBOX,n_b))], axis=1)
          A1 = np.concatenate([A1, np.zeros((s_SBOX,1))], axis=1)
          A = np.concatenate([A, A1], axis=0)
      A[:, -1] = -mean_profiling/std_profiling
  if version == "ind":
      A = np.ones((s_SBOX, n)) * (1/std_profiling)
      A[:, -1] = -mean_profiling/std_profiling
      A = np.concatenate([A,  np.zeros((s_SBOX,(n_d-1)*n))], axis=1)
      for index in range(1, n_d):
        A1 = np.ones((s_SBOX, n)) * (1/std_profiling)
        A1[:, -1] = -mean_profiling/std_profiling
        for l in range(0, index):
            A1 = np.concatenate([np.zeros((s_SBOX,n)), A1], axis=1)
        for l in range(index+1, n_d):
            A1 = np.concatenate([A1, np.zeros((s_SBOX,n))], axis=1)
        A = np.concatenate([A, A1], axis=0)
  V = np.eye(s_SBOX+s_MASKS*(n_d-1))*(sigma/std_profiling)**2
  return [(a_j, A, V)]


####################################################################################################################################
####################################################################################################################################
                                                    #MULTIVARIATE_GAUSSIAN
####################################################################################################################################
####################################################################################################################################



def multivariate_gaussian_3D( n_d, n_b, mean, cov, leakages_pois):
  det = np.linalg.det(cov)
  inv = np.linalg.inv(cov)
  l = leakages_pois.shape[0]
  n_probas = mean.shape[0]
  probas = np.zeros((n_probas, l))
  for zi in range(n_probas):
    mean_zi = mean[zi]
    diff = leakages_pois - mean_zi # (l, s)
    var = (-1/2) * np.einsum('li,ij,lj -> l', diff, inv, diff, optimize=True) # (l)
    coef = (np.pi*2) ** (cov.shape[0])
    coef = np.sqrt(coef *det)
    coef = 1/coef
    probas[zi, :] = coef*np.exp(var) # (l)
  return probas # (n_MASKS, l)

def all_combinations(n_d, size):
    grids = np.meshgrid(*[np.arange(size)] * (n_d-1), indexing='ij')
    combinations = np.stack(grids, axis=-1).reshape(-1, n_d-1)
    return combinations



def get_X_nind(n_d, n_b, classes):
  l = classes.shape[0]
  size = 2**n_b
  n_MASKS = size**(n_d-1)
  n = n_d * n_b

  ALL_MASKS = all_combinations(n_d=n_d, size=size)  # shape (n_MASKS, n_d)

  xor_vals = classes[:, None] ^ ALL_MASKS[:, 0][None, :]
  for d in range(1, n_d - 1):
      xor_vals ^= ALL_MASKS[:, d][None, :]

  bits = ((xor_vals[..., None] >> np.arange(n_b - 1, -1, -1)) & 1).astype(np.int8)  # (l, n_MASKS, n_b)
  bits = bits.transpose(1, 0, 2)  # (n_MASKS, l, n_b)

  for d in range(0, n_d-1):
      mask = ALL_MASKS[:, d]  # (n_MASKS,)
      mask_bits = ((mask[:, None] >> np.arange(n_b - 1, -1, -1)) & 1).astype(np.int8)  # (n_MASKS, n_b)
      mask_bits = np.broadcast_to(mask_bits[:, None, :], (n_MASKS, l, n_b))  # (n_MASKS, l, n_b)
      bits = np.concatenate([bits, mask_bits], axis=2)  # concatène sur les features


  ones = np.ones((n_MASKS, l, 1), dtype=np.int8)

  X = np.concatenate([bits, ones], axis=2)
  return X



def get_X_ind(n_d,n_b,classes, share='sbox'):
    l = classes.shape[0]
    size = 2 ** n_b
    n_MASKS = size**(n_d-1)

    ALL_MASKS = all_combinations(n_d=n_d, size=size)


    if share == 'sbox' or share=='all':
      xor_vals = classes[:, None] ^ (ALL_MASKS[:, 0])[None, :]
      for d in range(1, n_d-1):
          xor_vals = xor_vals ^ (ALL_MASKS[:, d])[None, :]
      bits = ((xor_vals[..., None] >> np.arange(n_b-1, -1, -1)) & 1)  # (l, n_MASKS, n_b)
      bits = bits.transpose(1, 0, 2)  # (n_MASKS, l, n_b)
      ones_sbox = np.ones((n_MASKS, l, 1), dtype=np.int8)
      X_SBOX = np.concatenate([bits, ones_sbox], axis=-1)  # (n_MASKS, l, n_b+1)

      if share=='sbox':
        return X_SBOX


    if share=="all":
      X = X_SBOX
    for d in range(n_d-1):
      if share == str(d) or share=='all':
        mask_bits = (((ALL_MASKS[:, d])[:, None] >> np.arange(n_b-1, -1, -1)) & 1)  # (n_MASKS, n_b)
        mask_bits = mask_bits.astype(np.int8)
        ones_mask = np.ones((n_MASKS, 1), dtype=np.int8)
        X_MASKS = np.concatenate([mask_bits, ones_mask], axis=1)  # (n_MASKS, n)
        X_MASKS = X_MASKS[:, None, :]  # (n_MASKS, 1, n)
        X_MASKS = np.broadcast_to(X_MASKS, (n_MASKS, l, X_MASKS.shape[-1]))
        if share == str(d):
          return X_MASKS
        X = np.concatenate([X, X_MASKS], axis=2)  # (n_MASKS, l, 2*n)
    return X
  # on crée deux tableaux composés de bit de 1 et 0
  # On concatène
  # On ajoute le 1 à la fin


####################################################################################################################################
####################################################################################################################################
                                                    #TRAINING
####################################################################################################################################
####################################################################################################################################


import psutil

def lin_nind_e_step(n_d, n_b, a_j, mean, cov, leakages_pois):
  probas = multivariate_gaussian_3D(n_d=n_d, n_b=n_b, mean=mean, cov=cov, leakages_pois=leakages_pois) #(n_MASKS, l)
  #probas += 1e-12
  probas = probas/np.sum(probas, axis=0)[None, :] #(n_MASKS, l)
  return probas
# X est de taille (n_MASKS, size, n)
# A est de taille (s, n)
def lin_nind_m_step(n_d, n_b, params, leakages_pois, classes, X, X_classes):
  (a_j, A, V) = params[0]

  size= 2**n_b
  n = n_d*n_b+1
  l = leakages_pois.shape[0]
  n_MASKS = (size)**(n_d-1)

  AX = np.einsum('mln, sn -> mls',X_classes, A)
  AX = AX[:, classes.flatten(), :]

  ajn = lin_nind_e_step(n_d=n_d, n_b=n_b, a_j=a_j, mean=AX, cov=V, leakages_pois=leakages_pois)
  XX = np.einsum('ml, mli, mlj -> ij', ajn, X, X, optimize=True) # (n, n)
  inv_XX = np.linalg.inv(XX) # (n,n)
  Y = np.einsum('mln, ls, ml -> sn', X, leakages_pois, ajn) #(s,n)
  A = np.matmul(Y, inv_XX) # (s,n)
  AX = np.einsum('mln, sn -> mls', X_classes, A)
  AX = AX[:, classes.flatten(), :] #(n_masks, l, s)
  Y = leakages_pois[None, :, :] - AX
  V = np.einsum('ml, mli, mlj -> ij', ajn, Y, Y, optimize=True)/ l # (s,s)
  return (a_j, A, V)

def lin_ind_e_step(n_d, n_b, params, l_leakages_pois,classes,  l_X_classes):
  (a_j, A, V) = params[0]
  AX = np.einsum('mln, sn -> mls',l_X_classes[0], A)
  AX = AX[:, classes.flatten(), :]
  probas = multivariate_gaussian_3D(n_d=n_d, n_b=n_b, mean=AX, cov=V, leakages_pois=l_leakages_pois[0])
  for n in range(1, len(params)):
    (a_j, A, V) = params[n]
    AX = np.einsum('mln, sn -> mls',l_X_classes[n], A)
    AX = AX[:, classes.flatten(), :]
    probas *= multivariate_gaussian_3D(n_d=n_d, n_b=n_b, mean=AX, cov=V, leakages_pois=l_leakages_pois[n])
  #probas += 1e-12
  probas = probas/np.sum(probas, axis=0)[None, :] #(n_MASKS, l)
  return probas


def lin_ind_m_step(n_d, n_b, params, l_leakages_pois, classes, l_X, l_X_classes):

  a = params[0][0]
  ajn = lin_ind_e_step(n_d=n_d, n_b=n_b, params=params, l_leakages_pois=l_leakages_pois, classes=classes, l_X_classes=l_X_classes)
  new_params = []
  for n in range(len(params)):
    XX = np.einsum('ml, mli, mlj -> ij', ajn, l_X[n], l_X[n], optimize='optimal')
    inv_XX = np.linalg.inv(XX)
    Y = np.einsum('mln, ls, ml -> sn', l_X[n], l_leakages_pois[n], ajn)
    A = np.matmul(Y, inv_XX)

    AX = np.einsum('mln, sn -> mls', l_X_classes[n], A)
    AX = AX[:, classes.flatten(), :]
    Y = l_leakages_pois[n][None, :, :] - AX
    V = np.einsum('ml, mli, mlj -> ij', ajn, Y, Y, optimize=True)/l_leakages_pois[n].shape[0]

    new_params.append((deepcopy(a), deepcopy(A), deepcopy(V)))
  return new_params

def lin_nind_training(n_d, n_b, params, leakages_pois, classes, epochs, leakages_test=None, classes_test=None, bestparams=[], criterion=[], round_without_increase=10, criterion_bound=1e-4, verbose=False):
  s = leakages_pois.shape[1]
  size = 2**n_b
  l = leakages_pois.shape[0]
  X_classes = get_X_nind(n_d=n_d, n_b=n_b, classes=np.arange(size)) #(n_MASKS, size, n)
  X = get_X_nind(n_d=n_d, n_b=n_b, classes=classes)
  pi=-10
  rwi = 0
  for epoch in range(epochs):
    newparams = lin_nind_m_step(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, classes=classes, X=X, X_classes=X_classes)
    params[0] = newparams
    if leakages_test is not None:
      new_pi, _ = PI(n_d=n_d, n_b=n_b, params=newparams, leakages_pois=leakages_test, classes=classes_test, version='nind')
      if verbose:
        print(f'new pi : {new_pi}')
      if new_pi > pi:
        pi = new_pi
        bestparams[0] = tuple(param for param in newparams)
        rwi = 0
      else :
        rwi += 1
        if rwi == round_without_increase and 'pi' in criterion:
            print(f'PI ne c est plus améliorée depuis {round_without_increase} round')
            return 1
  return 0


def lin_ind_training(n_d, n_b, params, l_leakages_pois, classes, epochs, leakages_test=None, classes_test=None, bestparams=[], criterion=[], round_without_increase=10, criterion_bound=1e-4, verbose=False):
  size = 2**n_b
  l_X_classes= []
  l_X_classes.append(get_X_ind(n_d=n_d, n_b=n_b, classes=np.arange(size), share='sbox'))
  for d in range (0,n_d-1):
    l_X_classes.append(get_X_ind(n_d=n_d, n_b=n_b, classes=np.arange(size), share=str(d)))
  l_X = []
  l_X.append(get_X_ind(n_d=n_d, n_b=n_b, classes=classes, share='sbox'))
  for d in range(0, n_d-1):
    l_X.append(get_X_ind(n_d=n_d, n_b=n_b, classes=classes, share=str(d)))
  pi = -10
  rwi=0
  for epoch in range(epochs):
    newparams = lin_ind_m_step(n_d=n_d, n_b=n_b, params=params, l_leakages_pois=l_leakages_pois, classes=classes, l_X=l_X, l_X_classes=l_X_classes)
    for i in range(len(params)):
      params[i] = newparams[i]
    if leakages_test is not None:
      fullparams = get_full_params(n_d=n_d, n_b=n_b, params=params)
      new_pi, _ = PI(n_d=n_d, n_b=n_b, params=fullparams[0], leakages_pois=leakages_test, classes=classes_test, version='ind')
      if verbose:
        print(f'new pi : {new_pi}')
      if new_pi > pi:
        pi = new_pi
        bestparams.clear()
        bestparams.extend(newparams)
        rwi = 0
      else:
        rwi += 1
        if rwi == round_without_increase and 'pi' in criterion:
          print(f"PI ne c'est plus améliorée depuis {round_without_increase} round")
          return 1
  return 0

####################################################################################################################################
####################################################################################################################################
                                                    #METRICS
####################################################################################################################################
####################################################################################################################################

import time as time

def p_l_z(n_d, n_b, params, leakages_pois, zi , version='nind'):
  size = 2**n_b
  l = leakages_pois.shape[0]
  s = leakages_pois.shape[1]
  if version=='nind':
    X = get_X_nind(n_d=n_d, n_b=n_b, classes=np.array([zi]))
  elif version=='ind':
    X = get_X_ind(n_d=n_d, n_b=n_b, classes=np.array([zi]), share='all')
  AX = np.einsum('mln, sn -> mls', X, params[1]) #(n_MASKS, 1, s)
  AX = np.broadcast_to(AX, (AX.shape[0], l, s))
  probas= multivariate_gaussian_3D(n_d=n_d, n_b=n_b, mean=AX, cov=params[2], leakages_pois=leakages_pois)
  probas = params[0][:, None] * probas
  return np.sum(probas, axis=0)

def log2_p_l_z(n_d, n_b, params, leakages_pois, zi, version='nind'):
  probas = p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, zi=zi, version=version)
  probas += 1e-12
  return np.log2(probas)

def log_p_l_z(n_d, n_b, params, leakages_pois, zi, version='nind'):
  probas = p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, zi=zi, version=version)
  probas += 1e-12
  return np.log(probas)

def all_p_l_z(n_d, n_b, params, leakages_pois, version='nind'):
  size = 2**n_b
  probas = np.zeros((size, leakages_pois.shape[0]))
  for zi in range(size):
    probas[zi, :] = p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, zi=zi, version=version)
  return probas

def all_log2_p_l_z(n_d, n_b, params, leakages_pois, version='nind'):
  probas = all_p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, version=version)
  probas += 1e-12
  return np.log2(probas)

def all_log_p_l_z(n_d, n_b, params, leakages_pois, version='nind'):
  probas = all_p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, version=version)
  probas += 1e-12
  return np.log(probas)

def p_z_l(n_d, n_b, params, leakages_pois, classes, version='nind'):
  size = 2**n_b
  probas = all_p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, version=version)
  probas += 1e-12
  return probas/np.sum(probas, axis=0)

def log2_p_z_l(n_d, n_b, params, leakages_pois, classes, version='nind'):
  return np.log2(p_z_l(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, classes=classes, version=version))

def log_p_z_l(n_d, n_b, params, leakages_pois, classes, version="nind"):
  return np.log(p_z_l(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, classes=classes, version=version))

def PI(n_d, n_b, params, leakages_pois, classes, version='nind'):
  size = 2**n_b
  non_existing=0
  log2_probas = log2_p_z_l(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, classes=classes, version=version)
  H_zi_l = 0
  H_zi_l_2 = 0
  for zi in range(size):
    if(zi in classes):
      log2_probas_zi = log2_probas[zi, classes==zi]
      H_zi_l += np.sum(log2_probas_zi)/log2_probas_zi.shape[0]
      H_zi_l_2 += (np.sum(log2_probas_zi)/log2_probas_zi.shape[0])**2
    else:
      non_existing += 1
  H_zi_l = H_zi_l = H_zi_l / (size-non_existing)
  H_zi_l_2 = H_zi_l_2/(size-non_existing)
  std = np.sqrt((H_zi_l_2 - H_zi_l**2)/leakages_pois.shape[0])
  return n_b + H_zi_l, std

####################################################################################################################################
####################################################################################################################################
                                                    #PLOTTING
####################################################################################################################################
####################################################################################################################################

import matplotlib.pyplot as plt
def plot_A_zi(n_d, n_b, params, zi, masks=[], version='nind'):
  size = 2**n_b
  if len(masks) == 0:
    n_masks = size**(n_d-1)
    MASKS = np.arange(n_masks)
  else :
    n_masks = len(masks)
    MASKS = np.array(masks)
  A = params[1]
  ZI = np.ones((n_masks,)) * zi
  SBOX = ZI.astype(int) ^ MASKS.astype(int)
  MASKS = ((MASKS[:, None] & (1 << np.arange(n_b)[::-1])) > 0).astype(int)
  SBOX = ((SBOX[:, None] & (1 << np.arange(n_b)[::-1])) > 0).astype(int)

  if (version=='nind'):
    ALL = np.concatenate([SBOX, MASKS, np.ones((n_masks, 1))], axis=1)
    ASBOX = np.einsum('n,sn->s', A[0], ALL)
    AMASKS = np.einsum('n,sn->s', A[1], ALL )
    plt.scatter(ASBOX, AMASKS)
    return
  if (version == 'ind'):
    ALL = np.concatenate([SBOX, np.ones((n_masks, 1)), MASKS, np.ones((n_masks, 1))], axis=1)
    ASBOX = np.einsum('n,sn->s', A[0], ALL)
    AMASKS = np.einsum('n,sn->s', A[1], ALL)
    print(ASBOX, AMASKS)
    plt.scatter(ASBOX, AMASKS)


from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
def plot_A_zi_2(n_d, n_b, params, zi, masks=[], version='nind', xc=0, yc=0):
  size = 2**n_b
  if len(masks) == 0:
    n_masks = size**(n_d-1)
    MASKS = np.arange(n_masks)
  else :
    n_masks = len(masks)
    MASKS = np.array(masks)
  A = params[1]
  ZI = np.ones((n_masks,)) * zi
  SBOX = ZI.astype(int) ^ MASKS.astype(int)
  MASKS = ((MASKS[:, None] & (1 << np.arange(n_b)[::-1])) > 0).astype(int)
  SBOX = ((SBOX[:, None] & (1 << np.arange(n_b)[::-1])) > 0).astype(int)

  if (version=='nind'):
    ALL = np.concatenate([SBOX, MASKS, np.ones((n_masks, 1))], axis=1)
    ASBOX = np.einsum('n,sn->s', A[0], ALL)
    AMASKS = np.einsum('n,sn->s', A[1], ALL )
    plt.scatter(ASBOX, AMASKS)

  elif (version == 'ind'):
    ALL = np.concatenate([SBOX, np.ones((n_masks, 1)), MASKS, np.ones((n_masks, 1))], axis=1)
    ASBOX = np.einsum('n,sn->s', A[0], ALL)
    AMASKS = np.einsum('n,sn->s', A[1], ALL)
    print(ASBOX, AMASKS)
    plt.scatter(ASBOX, AMASKS)
  cov = params[2]  # Matrice 2x2
  mean = [np.mean(ASBOX), np.mean(AMASKS)]

  def plot_cov_ellipse(cov, pos=None, n_std=2.0, xc=None, yc=None, **kwargs):

      eigvals, eigvecs = np.linalg.eigh(cov)
      order = eigvals.argsort()[::-1]
      eigvals, eigvecs = eigvals[order], eigvecs[:, order]

      theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
      width, height = 2 * n_std * np.sqrt(eigvals)

      # Priorité : xc/yc > pos > [0, 0]
      if xc is not None and yc is not None:
          center = [xc, yc]
      elif pos is not None:
          center = pos
      else:
          center = [0, 0]

      ellip = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)
      ax = plt.gca()
      ax.add_patch(ellip)
      plt.show()

  plot_cov_ellipse(cov, xc=xc, yc=yc, n_std=2.0,
                    edgecolor='red', facecolor='none', linewidth=2)


%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import plotly.graph_objects as go

def plot_distribution_3D(n_d, n_b, n_samples, params, n_points, zi, version,  xy1=[-3, 3], xy2=[-3, 3]):
    points1 = np.random.uniform(low=xy1[0], high=xy1[1], size=(n_points, 1))
    points2 = np.random.uniform(low=xy2[0], high=xy2[1], size=(n_points, 1))
    points = np.concatenate([points1, points2], axis=1)
    if version == 'ind':
      X = get_X_ind(n_d=n_d, n_b=n_b, classes=np.array([zi]), share='all')
    if version == 'nind':
      X = get_X_nind(n_d=n_d, n_b=n_b, classes=np.array([zi]))
    print(X.shape)
    print
    AX = np.einsum('mln,sn->mls', X, params[0][1]) #(n_MASKS, 1, s)
    probas = multivariate_gaussian_3D(n_d=n_d, n_b=n_b, mean=AX, cov=params[0][2], leakages_pois=points) #(n_MASKS,points)
    probas = params[0][0][:, None] * probas #(n_MASKS,points)
    probas = np.sum(probas, axis=0) #(points)

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=probas,
        mode='markers',
        marker=dict(
            size=4,
            color=probas,
            colorscale='plasma',
            colorbar=dict(title='Proba'),
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=f"3D PDF (zi = {zi})",
        scene=dict(
            xaxis_title='HW(SBOX ^ MASKS)',
            yaxis_title='HW(MASKS)',
            zaxis_title='Probability'
        ),
        height=600
    )

    fig.show()

####################################################################################################################################
####################################################################################################################################
                                                    #ATTACK
####################################################################################################################################
####################################################################################################################################

def multivariate_attack_dataset(n_d, n_b, sigma, n_leakages, s_SBOX=1, s_MASKS=1, s_useless=0):
  size = 2**n_b
  pts = np.random.randint(0, size, size=(n_leakages,))
  keys = np.random.randint(0, size) * np.ones((n_leakages,))
  if n_b==8:
    zi_classes = SBOX[pts.astype(np.int16) ^ keys.astype(np.int16)]
  elif n_b==4:
    zi_classes = little_sbox[pts.astype(np.int16) ^ keys.astype(np.int16)]
  MASKS_classes = np.random.randint(0,size,size=(n_leakages, n_d-1))


  xor_classes = zi_classes.copy()
  for d in range(n_d-1):
    xor_classes = xor_classes ^ MASKS_classes[:, d]

  leakages_SBOX = HW(xor_classes)
  leakages_MASKS =  HW(MASKS_classes)
  leakages_SBOX = np.tile(leakages_SBOX[:,None], (1,s_SBOX))
  leakages_MASKS = np.tile(leakages_MASKS, (1,s_MASKS))
  leakages = np.hstack((leakages_SBOX, leakages_MASKS))

  # Random data
  random_variable = None
  if s_useless > 0:
    random_variable = HW(np.random.randint(0,size,size=(n_leakages, s_useless)))
    leakages = np.hstack((leakages, random_variable))

  leakages = leakages + np.random.normal(loc = 0, scale=sigma, size=(n_leakages, leakages.shape[1]))
  return leakages, zi_classes, MASKS_classes, pts, keys

def attack(n_d, n_b, list_params, leakages_pois, pts, version):
  size = 2**n_b
  l=leakages_pois.shape[0]
  probas = np.zeros((size, l))
  for params in list_params:
    probas += all_log_p_l_z(n_d=n_d, n_b=n_b, params=params, leakages_pois=leakages_pois, version=version)
  p_k = np.zeros((size,))
  for key in range(size):
    _key = key * np.ones(pts.shape, dtype=np.int16)
    if n_b==8:
      zi_classes = SBOX[_key ^ pts.astype(np.int16)]
    elif n_b==4:
      zi_classes = little_sbox[_key ^ pts.astype(np.int16)]
    p_k[key] = np.sum(probas[zi_classes, np.arange(l)])
  ranking = np.argsort(p_k)
  return ranking[::-1]

def success_rate(n_d, n_b, list_params, n_tours, n_a, leakages_attack, pts_attack, keys_attack, version):
  sr = []
  for n_leakages in n_a:

    p=0
    for tour in range(n_tours):
      leakages_A, pts_A, key_A = leakages_attack[tour*(n_leakages):(tour+1)*n_leakages],  pts_attack[tour*(n_leakages):(tour+1)*n_leakages], keys_attack[tour*(n_leakages):(tour+1)*n_leakages]
      top_keys = []
      ranking = attack(n_d=n_d, n_b=n_b, list_params=list_params, leakages_pois = leakages_A, pts=pts_A, version=version)
      top_keys.append(ranking[0])
      top_key = np.bincount(top_keys).argmax()
      if(top_key == key_A[0]):
       p += 1
    sr.append(p/n_tours)
    print(f'n_leakages = {n_leakages}, p = {p/n_tours}')
  return sr