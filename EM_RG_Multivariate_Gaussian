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


def multivariate_dataset(n_d, n_b, sigma, n_leakages, s_SBOX, s_MASKS, s_useless=0):
  size = 2**n_b
  classes  = np.random.randint(0,size,size=(n_leakages, n_d))

  zi_classes = classes[:, 0].copy()
  for d in range(1, n_d):
    classes[:, 0] = classes[:, 0] ^ classes[:, d]

  leakages = HW(classes)
  leakages_SBOX = np.tile(leakages[:, :1], (1, s_SBOX))
  leakages_MASKS = np.tile(leakages[:, 1:], (1, s_MASKS))
  leakages = np.concatenate([leakages_SBOX, leakages_MASKS], axis=1).astype(float)
  if s_useless > 0:
    leakages_useless = np.random.randint(0, size, size=(leakages.shape[0], s_useless))
    leakages_useless = HW(leakages_useless)
    leakages = np.concatenate([leakages, leakages_useless], axis=1)
  leakages += np.random.normal(loc=0, scale=sigma, size=leakages.shape)
  classes[:, 0] = zi_classes

  return leakages, classes

def standardization(leakages):
  mean = np.mean(leakages)
  std = np.std(leakages)
  return ((leakages - mean)/std), mean, std
  
def em_criterion(newparams, params, verbose=False):
  delta_var = 0
  delta_mean = 0
  low_var = 0
  if len(params[1]) == len(params[2]):
    for zi in range(len(params[1])):
      delta_var += np.linalg.norm(params[2][zi] - newparams[2][zi])**2
      delta_mean += np.linalg.norm(params[1][zi] - newparams[1][zi])**2
  else:
    delta_var += np.linalg.norm(params[2]- newparams[2])**2
    for zi in range(len(params[1])):
      delta_mean += np.linalg.norm(params[1][zi] - newparams[1][zi])**2

      if(np.any(np.diag(params[2][zi]) < 10**-2)):
        low_var = 1
  if(low_var == 1):
    print('var < 1e-2')
  if verbose:
    print(f'delta mean = {delta_mean}')
    print(f'delta var = {delta_var}')
  return delta_var + delta_mean


####################################################################################################################################
####################################################################################################################################
                                                    #MULTIVARIATE_GAUSSIAN
####################################################################################################################################
####################################################################################################################################



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
  diff = leakages_pois[:, None, :] - mean[None, : , :]
  exponent = -0.5 * np.einsum('lni,ij,lnj->ln', diff, inv, diff)
  return coefs * np.exp(exponent)



def multivariate_gaussian_pdf_NV(leakages_pois, mean, var):
  epsilon = np.eye(var.shape[0], dtype=np.float64) * 1e-6
  #reg_var = var + epsilon
  det = np.linalg.det(var) #(1,)
  if(np.any(det <= 0)):
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
  diff = leakages_pois[:, None, :] - mean[None, : , :]
  exponent = -0.5 * np.einsum('lni,nij,lnj->ln', diff, inv, diff)
  return coefs * np.exp(exponent)



####################################################################################################################################
####################################################################################################################################
                                                    #INITIALISATION
####################################################################################################################################
####################################################################################################################################

def multivariate_gaussian_pdf_NV(leakages_pois, mean, var):
  epsilon = np.eye(var.shape[0], dtype=np.float64) * 1e-6
  #reg_var = var + epsilon
  det = np.linalg.det(var) #(1,)
  if(np.any(det <= 0)):
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
  diff = leakages_pois[:, None, :] - mean[None, : , :]
  exponent = -0.5 * np.einsum('lni,nij,lnj->ln', diff, inv, diff)
  return coefs * np.exp(exponent)



def multivariate_init_a(n_d, n_b, n_clusters):
  size = 2**n_b
  return [ np.full((n_clusters,), 1/n_clusters, dtype=np.float64) for _ in range(size)]


def multivariate_init_V(n_d, n_b, samples, sigma=1,n_clusters=1, version='1'):
  var = sigma**2*np.eye(len(samples), dtype=np.float64)
  size = 2**n_b
  if version == '1':
    return [var.copy() for _ in range(size)]
  elif version == 'N':
    size = 2**n_b
    return [  np.array([var.copy() for _ in range(n_clusters)]) for _ in range(size)]


from sklearn.cluster import KMeans
# leakages_pois (l,s)
def kmeans_multivariate_init_E_j(n_d, n_b, n_clusters, leakages_pois, classes):
  size = 2**n_b
  clusters = []
  for zi in range(size):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit( leakages_pois[classes==zi])
    clusters.append(kmeans.cluster_centers_)
  return clusters

def kmeans_multivariate_init(n_d, n_b, n_clusters, samples, leakages, classes, subkeys, sigma=1, version='1'):
  samples = samples[None, :] if samples.ndim==1 else samples
  classes = classes[:, None] if classes.ndim==1 else classes
  a = multivariate_init_a(n_d, n_b, n_clusters)
  return [ ( a.copy(), kmeans_multivariate_init_E_j(n_d, n_b, n_clusters, leakages[:, samples[subkey,:]], classes[:, subkey]), multivariate_init_V(n_d, n_b, samples[subkey, :], sigma=sigma, n_clusters=n_clusters, version=version)) for subkey in range(subkeys)]


def gaussian_multivariate_init_E_j(n_d, n_b, n_clusters, leakages_pois, classes):
  size = 2**n_b
  clusters = []
  for zi in range(size):
    means = np.mean(leakages_pois[classes==zi], axis=0) #(s,)
    std = np.std(leakages_pois[classes==zi], axis=0) #(s,)
    clusters.append(np.random.normal(loc=means, scale=std, size=(n_clusters, leakages_pois.shape[1])))
  return clusters

def gaussian_multivariate_init(n_d, n_b, n_clusters, samples, leakages, classes, subkeys, sigma=1, version='1'):
  samples = samples[None, :] if samples.ndim==1 else samples
  classes = classes[:, None] if classes.ndim==1 else classes
  a = multivariate_init_a(n_d, n_b, n_clusters)
  return [ ( a.copy(), gaussian_multivariate_init_E_j(n_d, n_b, n_clusters, leakages[:, samples[subkey,:]], classes[:, subkey]), multivariate_init_V(n_d, n_b, samples[subkey, :], sigma=sigma, n_clusters=n_clusters, version=version)) for subkey in range(subkeys)]



def init_noisy_T(n_d, n_b, s, sigma, std):
  size = 2**n_b
  T = np.eye(s) * sigma**2 / (std**2)
  return [T.copy() for _ in range(size)]

def init_diff_T(n_d,n_b, s, sigma, std):
  size = 2**n_b
  var = sigma*sigma
  T = var/(std**2)
  return [T.copy() for _ in range(size)]



####################################################################################################################################
####################################################################################################################################
                                                    #TRAINING
####################################################################################################################################
####################################################################################################################################


def multivariate_e_step_1(n_d, n_b, samples, params, leakages):
  (a_j_zi, E_j_zi, V_zi) = params
  if(leakages.ndim==1):
    leakages = leakages[None :]
  num = a_j_zi[None, :] * multivariate_gaussian_pdf_1V(leakages[:,samples], E_j_zi, V_zi) #(l,d)
  return num / np.sum(num, axis=1, keepdims=True)

from sklearn.model_selection import KFold
# Inutile car on a des besoins des masques pour que ça marche
def L_fold_1(n_d, n_b, samples, leakages,n_splits, candidates, T):
  leakages_pois = leakages[:, samples]

  epsilon = np.eye(len(samples), dtype=np.float64) * 1e-7 # paramètre pouvant combattre les singularités
  error = np.zeros((len(candidates),))

  kf = KFold(n_splits=n_splits, shuffle=True)

  for i, (tr_index, val_index) in enumerate(kf.split(leakages_pois)):

    leakages_pois_tr, leakages_pois_val = leakages_pois[tr_index], leakages_pois[val_index] # on divise,

    len_tr = leakages_pois_tr.shape[0] # tot/n
    len_val = leakages_pois_val.shape[0] # (tot * n-1 / n)

    mean_tr = np.mean(leakages_pois_tr, axis=0) # (s)
    mean_val = np.mean(leakages_pois_val, axis=0) # (s)

    diff_val = leakages_pois_val - mean_val # (lv,s)
    diff_tr = leakages_pois_tr - mean_tr # (lt,s)

    S_val = np.sum(np.matmul(diff_val[:,:, None], diff_val[:, None, :]), axis=0)/(len_val) # (l, s, s)

    E_tr = np.sum(np.matmul(diff_tr[:, :, None], diff_tr[:, None, :]), axis=0)/(len_tr) # (s, s)

    E_eta = (len_tr/(candidates+len_tr))[:, None, None] * E_tr[None, :, :] + ( candidates/(candidates+len_tr))[:, None, None] * T[None, :, :] + epsilon[None, :, :]
    #(eta, s, s)
    det_E_eta = np.linalg.det(E_eta ) #(eta,)
    inv_E_eta = np.linalg.inv(E_eta)  # (eta, s, s)

    trace = np.trace(np.matmul(inv_E_eta, S_val), axis1=1, axis2=2) #(eta,)
    #trace = np.trace(np.matmul(inv_E_eta, T), axis1=1, axis2=2)
    log_det = np.log(det_E_eta) #(eta, )

    error += trace + log_det # (eta, )
  print(f'error = {error}')
  return candidates[np.argmin(error)]

# C'est EM-classique avec en plus le coup du eta et T,
# Faire le K-fold, sinon simple
def multivariate_m_step_rg_1(n_d, n_b, samples, params, leakages, classes, T, verbose=False):
  size = 2**n_b
  s=len(samples)
  non_existing = 0
  (a_j , E_j, V) = params
  new_E_j = []
  new_a_j = []
  new_V = []
  epsilon = np.eye(len(samples), dtype=np.float64) * 1e-7 # paramètre pouvant combattre les singularités
  for zi in range(size):
    leakages_zi = leakages[classes == zi] # (l,stot)
    if(leakages_zi.shape[0] != 0):
      if(leakages_zi.ndim == 1):
        leakages_zi = leakages_zi[None, :]
      #print("e-step start...")
      a_j_n= multivariate_e_step_1(n_d, n_b, samples, (a_j[zi], E_j[zi], V[zi]), leakages_zi) #(l,d)
      for ajn in a_j_n[0]:
        if np.isnan(ajn):
          print(f'zi : {zi}')
          print(a_j_n[0])
          raise ValueError("NaN founded in m_step after e_step")
      #print("e-step end...")
      #print("m-step start...")
      if(a_j_n.ndim==1):
        raise ValueError(f'a_j_n a dim ==1 apres e-step')
      sum_a_j_n = np.sum(a_j_n, axis=0) #(d,)

      new_a_j.append(sum_a_j_n / leakages_zi.shape[0])
      new_E_j.append(np.dot(a_j_n.T, leakages_zi[:, samples]) / sum_a_j_n.reshape((a_j_n.shape[1],1)))

      diff = leakages_zi[:,samples][:, None, :] - new_E_j[zi][None, :, :]

      #eta = np.log(np.linalg.norm(np.trace(V[zi]- T[zi]))**2 / np.linalg.norm(np.trace(T[zi])) + 1)
      eta = np.log(np.linalg.norm(np.diag(V[zi]- T[zi]))**2 / np.linalg.norm(np.diag(T[zi]))**2 + 1)
      #if verbose :
      #  print(f'candidate = {eta}')
      cov = a_j_n[:, :, None, None] * np.matmul(diff[:, :, :, None], diff[:, :, None, :]) # (l,d,s,s)
      V1 = np.sum(np.sum(cov, axis=1), axis=0) / (leakages_zi.shape[0])
      V1 = V1 / (eta + 1)
      V2 = (eta/(eta+1)) * T[zi]
      new_V.append( V1 + V2 + epsilon )
      # autre régularisation possible new_V_j[zi] = 0.5 * (new_V_j[zi] + np.transpose(new_V_j[zi], axes=(0, 2, 1)))
      #print("m-step end...")
    else :
      non_existing += 1
      new_V.append(np.array([1]))
      new_E_j.append(np.array([0]))
      new_a_j.append(np.array([1e-6]))
      print(f'No Leakage value for zi = {zi}')

  return (new_a_j, new_E_j, new_V)


def multivariate_m_step_mix_1(n_d, n_b, samples, params, leakages, classes, T1, T2, verbose=False):
  size = 2**n_b
  s=len(samples)
  non_existing = 0
  (a_j , E_j, V) = params
  new_E_j = []
  new_a_j = []
  new_V = []
  epsilon = np.eye(len(samples), dtype=np.float64) * 1e-7 # paramètre pouvant combattre les singularités
  for zi in range(size):
    leakages_zi = leakages[classes == zi] # (l,stot)
    if(leakages_zi.shape[0] != 0):
      if(leakages_zi.ndim == 1):
        leakages_zi = leakages_zi[None, :]
      #print("e-step start...")
      a_j_n= multivariate_e_step_1(n_d, n_b, samples, (a_j[zi], E_j[zi], V[zi]), leakages_zi) #(l,d)
      for ajn in a_j_n[0]:
        if np.isnan(ajn):
          print(f'zi : {zi}')
          print(a_j_n[0])
          raise ValueError("NaN founded in m_step after e_step")
      #print("e-step end...")
      #print("m-step start...")
      if(a_j_n.ndim==1):
        raise ValueError(f'a_j_n a dim ==1 apres e-step')
      sum_a_j_n = np.sum(a_j_n, axis=0) #(d,)

      new_a_j.append(sum_a_j_n / leakages_zi.shape[0])
      new_E_j.append(np.dot(a_j_n.T, leakages_zi[:, samples]) / sum_a_j_n.reshape((a_j_n.shape[1],1)))

      diff = leakages_zi[:,samples][:, None, :] - new_E_j[zi][None, :, :]

      #eta = np.log(np.linalg.norm(np.trace(V[zi]- T[zi]))**2 / np.linalg.norm(np.trace(T[zi])) + 1)
      delta = np.linalg.norm(np.diag(T2[zi]- T1[zi]))**2
      e1 = np.linalg.norm(np.diag(V[zi]- T1[zi]))**2
      e2 = np.linalg.norm(np.diag(V[zi]- T2[zi]))**2
      if e1 >= delta :
        eta = np.log(np.linalg.norm(np.diag(V[zi]- T2[zi]))**2 / np.linalg.norm(np.diag(T2[zi]))**2 + 1)
        T_zi = T2[zi]
      elif e2 >= delta :
        eta = np.log(np.linalg.norm(np.diag(V[zi]- T1[zi]))**2 / np.linalg.norm(np.diag(T1[zi]))**2 + 1)
        T_zi = T1[zi]
      else :
        eta = 0
        T_zi = T1[zi] 
      #if verbose :
        #print(f'candidate = {eta}')
      cov = a_j_n[:, :, None, None] * np.matmul(diff[:, :, :, None], diff[:, :, None, :]) # (l,d,s,s)
      V1 = np.sum(np.sum(cov, axis=1), axis=0) / (leakages_zi.shape[0])
      V1 = V1 / (eta + 1)
      V2 = (eta/(eta+1)) * T_zi
      new_V.append( V1 + V2 + epsilon)
      # autre régularisation possible new_V_j[zi] = 0.5 * (new_V_j[zi] + np.transpose(new_V_j[zi], axes=(0, 2, 1)))
      #print("m-step end...")
    else :
      non_existing += 1
      new_V.append(np.array([1]))
      new_E_j.append(np.array([0]))
      new_a_j.append(np.array([1e-6]))
      print(f'No Leakage value for zi = {zi}')

  return (new_a_j, new_E_j, new_V)


def multivariate_training_rg_1(n_d, n_b, samples, params, leakages, classes, subkeys, epochs, T, leakages_test=None,classes_test=None, bestparams=[],criterion=[], round_without_increase=10, criterion_bound=1e-4, verbose=False):
  for subkey in range(subkeys):
    pi = -10
    rwi = 0
    for epoch in range(epochs):
      sample = samples if samples.ndim == 1 else samples[subkey, :]
      classe = classes if classes.ndim == 1 else classes[:,subkey]
      newparams = multivariate_m_step_rg_1(n_d, n_b, sample, params[subkey], leakages, classe, T=T, verbose=verbose)
      if('bound' in criterion):
        if(em_criterion(newparams, params[subkey], verbose=verbose) < criterion_bound):
          print('Critère atteint')
          return 1
      if leakages_test is not None:
        new_pi, _ = PI(n_d=n_d, n_b=n_b, params=newparams, leakages_pois=leakages_test, classes=classes_test)
        if verbose:
          print(f'new_pi {new_pi}')
        if new_pi > pi:
          pi = new_pi
          bestparams[subkey] = tuple([param for param in newparams])
          rwi = 0
        else:
          rwi += 1
          if rwi == round_without_increase and 'pi' in criterion:
            print(f'PI ne c est plus améliorée depuis {round_without_increase} round')
            return 1
      params[subkey] = newparams
  return 0

#Dans criterion j'aurai 'bound' et 'pi'
def multivariate_training_mix_1(n_d, n_b, samples, params, leakages, classes, subkeys, epochs, T1, T2, leakages_test=None,classes_test=None, bestparams=[], criterion=[], round_without_increase=10, criterion_bound=1e-4, verbose=False):
  for subkey in range(subkeys):
    pi = -10
    rwi = 0
    for epoch in range(epochs):
      sample = samples if samples.ndim == 1 else samples[subkey, :]
      classe = classes if classes.ndim == 1 else classes[:,subkey]
      newparams = multivariate_m_step_rg_1(n_d, n_b, sample, params[subkey], leakages, classe, T1=T1, T2=T2, verbose=verbose)
      if('bound' in criterion):
        if(em_criterion(newparams, params[subkey], verbose=verbose) < criterion_bound):
          print('Critère atteint')
          return 1
      if leakages_test is not None:
        new_pi, _ = PI(n_d=n_d, n_b=n_b, params=newparams, leakages_pois=leakages_test, classes=classes_test)
        if verbose:
          print(f'new_pi {new_pi}')
        if new_pi > pi:
          pi = new_pi
          bestparams[subkey] = tuple([param for param in newparams])
          rwi = 0
        else:
          rwi += 1
          if rwi == round_without_increase and 'pi' in criterion:
            print(f'PI ne c est plus améliorée depuis {round_without_increase} round')
            return 1
      params[subkey] = newparams
  return 0



####################################################################################################################################
####################################################################################################################################
                                                    #PLOTTING
####################################################################################################################################
####################################################################################################################################


def plot_pi_ti(n_d, n_b, n_clusters, sigma, n_test, samples, epochs, profiling_points, criterion=['bound'], round_without_increase=10, s_SBOX=1, s_MASKS=1, s_useless=0, init='kmeans'):
  leakages_profiling, classes_profiling = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=max(profiling_points), s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless)
  zi_classes_profiling, m_classes_profiling = classes_profiling[:, 0], classes_profiling[:, 1:]

  leakages_test, classes_test = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=n_test, s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless)
  zi_classes_test, m_classes_test = classes_test[:, 0], classes_test[:, 1:]

  size = 2**n_b
  plt.figure(figsize=(10, 6))
  pi = []
  ti = []
  pi_std = []
  ti_std = []

  for point in profiling_points:

    leakages_train = leakages_profiling[:point]
    zi_classes_train, m_classe_train = zi_classes_profiling[:point], m_classes_profiling[:point]

    leakages_train, mean_train, std_train = standardization(leakages_train)
    leakages_test_norm = (leakages_test - mean_train)/std_train

    if init == 'kmeans':
      params_train = kmeans_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_train, classes=zi_classes_train, subkeys=1)
    elif init == 'gaussian':
      params_train = gaussian_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_train, classes=zi_classes_train, subkeys=1)

    T_train = init_noisy_T(n_d=n_d, n_b=n_b, s=len(samples), sigma=sigma, std=std_train)
    if 'pi' not in criterion:
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs, T=T_train, criterion=criterion)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_train, classes=zi_classes_train)
    else:
      bestparams = deepcopy(params_train)
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs, T=T_train, criterion=criterion, round_without_increase=round_without_increase, bestparams=bestparams)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_train, classes=zi_classes_train)
    pi.append(_pi)
    pi_std.append(_pi_std)
    ti.append(_ti)
    ti_std.append(_ti_std)

  plt.plot(profiling_points, pi, label='PI', c='r')
  plt.plot(profiling_points, ti, label='TI', c='b')
  plt.fill_between(profiling_points, np.array(pi) - np.array(pi_std), np.array(pi) + np.array(pi_std), color='red', alpha=0.2)
  plt.fill_between(profiling_points, np.array(ti) - np.array(ti_std), np.array(ti) + np.array(ti_std), color='blue', alpha=0.2)
  plt.title(f'PI and TI for n_d = {n_d} n_b = {n_b} sigma = {sigma}')
  plt.xlabel('nb of profiling points')
  plt.ylabel(f'PI (red) TI(blue)')
  plt.legend()
  plt.show()
  return (pi, pi_std, ti, ti_std)


def plot_pi_ti_2(n_d, n_b, n_clusters, sigma, n_test, samples, epochs, profiling_points, criterion=['bound'], round_without_increase=10, s_SBOX=1, s_MASKS=1, s_useless=0, init='kmeans'):
  leakages_profiling, classes_profiling = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=max(profiling_points), s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless )
  zi_classes_profiling, m_classes_profiling = classes_profiling[:, 0], classes_profiling[:, 1:]

  leakages_test, classes_test = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=n_test, s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless)
  zi_classes_test, m_classes_test = classes_test[:, 0], classes_test[:, 1:]

  size = 2**n_b
  plt.figure(figsize=(10, 6))
  pi = []
  ti = []
  pi_std = []
  ti_std = []

  min_point = min(profiling_points)
  leakages_init = leakages_profiling[:min_point]
  zi_classes_init, m_classe_init = zi_classes_profiling[:min_point], m_classes_profiling[:min_point]
  leakages_init, _, _ = standardization(leakages_init)

  if init == 'kmeans':
    params_init = kmeans_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_init, classes=zi_classes_init, subkeys=1)
  elif init == 'gaussian':
    params_init = gaussian_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_init, classes=zi_classes_init, subkeys=1)


  for point in profiling_points:

    leakages_train = leakages_profiling[:point]
    zi_classes_train, m_classe_train = zi_classes_profiling[:point], m_classes_profiling[:point]

    leakages_train, mean_train, std_train = standardization(leakages_train)
    leakages_test_norm = (leakages_test - mean_train)/std_train

    params_train = params_init.copy()

    T_train = init_noisy_T(n_d=n_d, n_b=n_b, s=len(samples), sigma=sigma, std=std_train)
    if 'pi' not in criterion:
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs,T=T_train, criterion=True)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_train, classes=zi_classes_train)
    else:
      bestparams = deepcopy(params_train)
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs, T=T_train, criterion=criterion, round_without_increase=round_without_increase, bestparams=bestparams)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_train, classes=zi_classes_train)
    pi.append(_pi)
    pi_std.append(_pi_std)
    ti.append(_ti)
    ti_std.append(_ti_std)

  plt.plot(profiling_points, pi, label='PI', c='r')
  plt.plot(profiling_points, ti, label='TI', c='b')
  plt.fill_between(profiling_points, np.array(pi) - np.array(pi_std), np.array(pi) + np.array(pi_std), color='red', alpha=0.2)
  plt.fill_between(profiling_points, np.array(ti) - np.array(ti_std), np.array(ti) + np.array(ti_std), color='blue', alpha=0.2)
  plt.title(f'PI and TI for n_d = {n_d} n_b = {n_b} sigma = {sigma}')
  plt.xlabel('nb of profiling points')
  plt.ylabel(f'PI (red) TI(blue)')
  plt.legend()
  plt.show()
  return (pi, pi_std, ti, ti_std)


def get_pi_ti(n_d, n_b, n_clusters, sigma, n_test, epochs, profiling_points, criterion=['bound'], round_without_increase=10, s_SBOX=1, s_MASKS=1, s_useless=0, init='kmeans'):
  leakages_profiling, classes_profiling = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=max(profiling_points), s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless)
  zi_classes_profiling, m_classes_profiling = classes_profiling[:, 0], classes_profiling[:, 1:]

  leakages_test, classes_test = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=n_test, s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless)
  zi_classes_test, m_classes_test = classes_test[:, 0], classes_test[:, 1:]

  size = 2**n_b
  samples = np.arange(s_SBOX + s_MASKS + s_useless)
  pi = []
  ti = []
  pi_std = []
  ti_std = []

  for point in profiling_points:

    leakages_train = leakages_profiling[:point]
    zi_classes_train, m_classe_train = zi_classes_profiling[:point], m_classes_profiling[:point]

    leakages_train, mean_train, std_train = standardization(leakages_train)
    leakages_test_norm = (leakages_test - mean_train)/std_train

    if init == 'kmeans':
      params_train = kmeans_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_train, classes=zi_classes_train, subkeys=1)
    elif init == 'gaussian':
      params_train = gaussian_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_train, classes=zi_classes_train, subkeys=1)

    T_train = init_noisy_T(n_d=n_d, n_b=n_b, s=len(samples), sigma=sigma, std=std_train)
    if 'pi' not in criterion:
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs, T=T_train, criterion=criterion)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_train, classes=zi_classes_train)
    else:
      bestparams = deepcopy(params_train)
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs, T=T_train, criterion=criterion, round_without_increase=round_without_increase, bestparams=bestparams)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_train, classes=zi_classes_train)
    pi.append(_pi)
    pi_std.append(_pi_std)
    ti.append(_ti)
    ti_std.append(_ti_std)

  return (pi, pi_std, ti, ti_std)


def get_pi_ti_2(n_d, n_b, n_clusters, sigma, n_test, epochs, profiling_points, criterion=['bound'], round_without_increase=10, s_SBOX=1, s_MASKS=1, s_useless=0, init='kmeans'):
  leakages_profiling, classes_profiling = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=max(profiling_points), s_SBOX=s_SBOX, s_MASKS=s_MASKS, s_useless=s_useless )
  zi_classes_profiling, m_classes_profiling = classes_profiling[:, 0], classes_profiling[:, 1:]

  leakages_test, classes_test = multivariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=n_test)
  zi_classes_test, m_classes_test = classes_test[:, 0], classes_test[:, 1:]

  size = 2**n_b
  samples = np.arange(s_SBOX + s_MASKS + s_useless)
  pi = []
  ti = []
  pi_std = []
  ti_std = []

  min_point = min(profiling_points)
  leakages_init = leakages_profiling[:min_point]
  zi_classes_init, m_classe_init = zi_classes_profiling[:min_point], m_classes_profiling[:min_point]
  leakages_init, _, _ = standardization(leakages_init)

  if init == 'kmeans':
    params_init = kmeans_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_init, classes=zi_classes_init, subkeys=1)
  elif init == 'gaussian':
    params_init = gaussian_multivariate_init(n_d=n_d, n_b=n_b, n_clusters=n_clusters, samples=samples, leakages=leakages_init, classes=zi_classes_init, subkeys=1)


  for point in profiling_points:

    leakages_train = leakages_profiling[:point]
    zi_classes_train, m_classe_train = zi_classes_profiling[:point], m_classes_profiling[:point]

    leakages_train, mean_train, std_train = standardization(leakages_train)
    leakages_test_norm = (leakages_test - mean_train)/std_train

    params_train = params_init.copy()

    T_train = init_noisy_T(n_d=n_d, n_b=n_b, s=len(samples), sigma=sigma, std=std_train)
    if 'pi' not in criterion:
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs,T=T_train, criterion=criterion)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=params_train[0], leakages_pois=leakages_train, classes=zi_classes_train)
    else:
      bestparams = deepcopy(params_train)
      multivariate_training_rg_1(n_d=n_d, n_b=n_b, samples=samples, params=params_train, leakages=leakages_train, classes=zi_classes_train, subkeys=1, epochs=epochs, T=T_train, criterion=criterion, round_without_increase=round_without_increase, bestparams=bestparams)
      _pi, _pi_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_test_norm, classes=zi_classes_test)
      _ti, _ti_std = PI(n_d=n_d, n_b=n_b, params=bestparams[0], leakages_pois=leakages_train, classes=zi_classes_train)
    pi.append(_pi)
    pi_std.append(_pi_std)
    ti.append(_ti)
    ti_std.append(_ti_std)

  return (pi, pi_std, ti, ti_std)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

def plot_all_clusters(n_d, n_b, n_clusters, params, zi, n_points, n_samples, mean, std,  xlim=5, ylim=5, version='1'):

    if version == '1':
        multivariate_gaussian_pdf = multivariate_gaussian_pdf_1V
    elif version == 'N':
        multivariate_gaussian_pdf = multivariate_gaussian_pdf_NV
    else:
        raise ValueError("Version should be '1' or 'N'")


    fig, ax = plt.subplots(facecolor='lightblue')       
    ax.set_facecolor('lightyellow')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)




    fig.tight_layout()

    n = 256
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i / n) for i in range(n)]
    vals, vecs = np.linalg.eigh(params[2][zi])
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    thetah = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 *  2.45 * np.sqrt(vals)

    border_sup = 1/n_clusters + np.std(params[0][zi])
    border_inf = 1/n_clusters - np.std(params[0][zi])

    for di in range(n_b+1):
      for dj in range(n_b+1):
        ax.scatter((di-mean)/std, (dj-mean)/std, c='red',marker='x', linewidths=2)


    for d in range(n_clusters):

      if(params[0][zi][d] > border_sup):
        cc = 'lightgreen'
      elif (params[0][zi][d] < border_inf):
        cc='cyan'
      else:
        cc = 'black'

      ax.scatter(*params[1][zi][d], c=cc, label=f'Mean {d}')
      ellipse = Ellipse(xy=params[1][zi][d], width=width, height=height, angle=thetah,
                        facecolor='None', edgecolor='orange')
      ax.add_patch(ellipse)
      ax.set_title(f'clusters for classes = {zi}')





def plot_all_clusters_2(n_d, n_b, n_clusters, params, zi, n_points, n_samples, mean, std,  xlim=5, ylim=5, version='1'):

    if version == '1':
        multivariate_gaussian_pdf = multivariate_gaussian_pdf_1V
    elif version == 'N':
        multivariate_gaussian_pdf = multivariate_gaussian_pdf_NV
    else:
        raise ValueError("Version should be '1' or 'N'")




    fig, ax = plt.subplots(facecolor='lightblue')       
    ax.set_facecolor('lightyellow')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)


    n = 256
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i / n) for i in range(n)]

    global_cov = params[2][zi]
    global_mean = np.array([0,0])
    vals, vecs = np.linalg.eigh(global_cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    thetah = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 *  2.45 * np.sqrt(vals)

    border_sup = 1/n_clusters + np.std(params[0][zi])
    border_inf = 1/n_clusters - np.std(params[0][zi])


    for di in range(n_b+1):
      for dj in range(n_b+1):
        ax.scatter((di-mean)/std, (dj-mean)/std, c='red',marker='x', linewidths=2)

    ellipse = Ellipse(xy=global_mean, width=width, height=height, angle=thetah,
                        facecolor='None', edgecolor='orange')
    for d in range(n_clusters):

      if(params[0][zi][d] > border_sup):
        cc = 'lightgreen'
      elif (params[0][zi][d] < border_inf):
        cc='cyan'
      else:
        cc = 'black'

      ax.scatter(*params[1][zi][d], c=cc, label=f'Mean {d}')

      ax.add_patch(ellipse)
      ax.set_title(f'clusters for classes = {zi}')




####################################################################################################################################
####################################################################################################################################
                                                    #METRICS
####################################################################################################################################
####################################################################################################################################


# leakages (l,), params((zi, d,), (zi, d,), (1,)), le zi cible
def p_l_zi(params, leakages_pois, zi, version='1'):
  if version == '1':
    probas = params[0][zi] * multivariate_gaussian_pdf_1V(leakages_pois, params[1][zi], params[2][zi]) # (l,d)
  elif version == 'N':
    probas = params[0][zi][None, :] * multivariate_gaussian_pdf_NV(leakages_pois, params[1][zi], params[2][zi]) # (l,d)
  return np.sum(probas, axis=1) # (l,)
# la PI

def log2_p_l_zi(params, leakages_pois, zi, version='1'):
  return np.log2(p_l_zi(params, leakages_pois, zi, version=version) + 1e-12) #(l,)

def log_p_l_zi(params, leakages_pois, zi, version='1'):
  return np.log(p_l_zi(params, leakages_pois, zi, version=version) + 1e-12) #(l,)

def all_p_l_zi(n_d, n_b, params, leakages_pois, version='1'):
  size = 2**n_b
  probas_l_zi = np.zeros((size, leakages_pois.shape[0]), dtype=np.float64) #(size, l)
  for zi in range(size):
    probas_l_zi[zi] = p_l_zi(params, leakages_pois, zi, version=version) #(l,)
  return probas_l_zi #(size, l)

def all_log2_p_l_zi(n_d, n_b, params, leakages_pois, version='1'):
  return np.log2(all_p_l_zi(n_d, n_b, params, leakages_pois, version=version) + 1e-12) #(size, l)

def all_log_p_l_zi(n_d, n_b, params, leakages_pois, version='1'):
  return np.log(all_p_l_zi(n_d, n_b, params, leakages_pois, version=version) + 1e-12) #(size, l)
# j'ai p(zi / l) = p(l / zi) * p(zi) / sum( p(l/zj) * p(zj))
# je dois donc calculer tous les p(l/jz)

# j'ai mes leakages, je prends tous les zi possibles (taille zi) je dois ressortir un truc de taille (zi, l)
# chaque ligne c'est p(zi /l)
def p_zi_l(n_d, n_b, params, leakages_pois, version='1'):
  size = 2**n_b
  probas_l_zi = all_p_l_zi(n_d, n_b, params, leakages_pois, version=version) #(size, l)
  sum_probas_l_zi = np.sum(probas_l_zi, axis=0) + 1e-12 #(l,)
  return probas_l_zi / sum_probas_l_zi[None, :]  #(size, l)

def log_p_zi_l(n_d, n_b, params, leakages_pois, version='1'):
  return np.log(p_zi_l(n_d, n_b, params, leakages_pois, version=version) + 1e-12) #(size, l)

def log2_p_zi_l(n_d, n_b, params, leakages_pois, version='1'):
  return np.log2(p_zi_l(n_d, n_b, params, leakages_pois, version=version) + 1e-12) #(size, l)


def PI(n_d, n_b, params, leakages_pois, classes, version='1'):
  size = 2**n_b
  non_existing = 0
  log2_probas = log2_p_zi_l(n_d, n_b, params, leakages_pois, version=version) #(size, l)
  H_zi_l = 0
  H_zi_l_2 = 0
  for zi in range(size):
    if(zi in classes):
      log2_probas_zi = log2_probas[zi, classes==zi] #(lzi,) ou lzi est le nombre de traces ayant la classe ==zi
      H_zi_l += np.sum(log2_probas_zi)/log2_probas_zi.shape[0]
      H_zi_l_2 += (np.sum(log2_probas_zi)/log2_probas_zi.shape[0])**2
      #print(f'Pi existing zi={zi}')
    else:
      non_existing += 1
      print(f'Pi non-existing zi={zi}')
  H_zi_l = H_zi_l / (size - non_existing)
  H_zi_l_2 = H_zi_l_2 / (size - non_existing)
  std = np.sqrt( H_zi_l_2 - H_zi_l**2)
  return n_b + H_zi_l, std


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
  masks_classes = np.random.randint(0,size,size=(n_leakages, n_d-1))


  xor_classes = zi_classes.copy()
  for d in range(n_d-1):
    xor_classes = xor_classes ^ masks_classes[:, d]

  leakages_SBOX = HW(xor_classes)
  leakages_MASKS =  HW(masks_classes)
  leakages_SBOX = np.tile(leakages_SBOX[:,None], (1,s_SBOX))
  leakages_MASKS = np.tile(leakages_MASKS, (1,s_MASKS))
  leakages = np.hstack((leakages_SBOX, leakages_MASKS))

  # Random data
  random_variable = None
  if s_useless > 0:
    random_variable = HW(np.random.randint(0,size,size=(n_leakages, s_useless)))
    leakages = np.hstack((leakages, random_variable))

  leakages = leakages + np.random.normal(loc = 0, scale=sigma, size=(n_leakages, leakages.shape[1]))
  return leakages, zi_classes, masks_classes, pts, keys


def attack(n_d, n_b, list_params, leakages_attack_pois, pts_attack, version='1'):
  size = 2**n_b
  probas = np.zeros((size, leakages_attack_pois.shape[0]), dtype=np.float64)
  for params in list_params:
    probas+= all_log_p_l_zi(n_d, n_b, params, leakages_attack_pois, version=version) # (size, l)
  p_k = np.zeros((size,), dtype=np.float64)
  for key in range(size):
    key_ = key*np.ones((pts_attack.shape[0],))
    if n_b == 8:
      zi_classes = SBOX[key_.astype(np.int16) ^ pts_attack]
    elif n_b==4:
      zi_classes = little_sbox[key_.astype(np.int16) ^ pts_attack]
    p_k[key] = np.sum(probas[zi_classes, np.arange(leakages_attack_pois.shape[0])])
  ranking = np.argsort(p_k)
  return ranking[::-1]


def success_rate(n_d, n_b, list_params, n_tours, n_a,  leakages_attack_pois, pts_attack, keys_attack, version='1'):

  sr=[]
  for n_leakages in n_a:

    p=0
    for tour in range(n_tours):
      leakages_A, pts_A, keys_A = leakages_attack_pois[tour*(n_leakages):(tour+1)*n_leakages], pts_attack[tour*(n_leakages):(tour+1)*n_leakages], keys_attack[tour*(n_leakages):(tour+1)*n_leakages]
      top_keys = []
      ranking = attack(n_d=n_d, n_b=n_b, list_params=list_params, leakages_attack_pois=leakages_A, pts_attack=pts_A)
      top_keys.append(ranking[0])
      top_key = np.bincount(top_keys).argmax()
      if top_key == keys_attack[0]:
        p+=1
    sr.append(p/n_tours)
    print(f'n_leakages = {n_leakages}, p = {p/n_tours}')
  return sr

  

