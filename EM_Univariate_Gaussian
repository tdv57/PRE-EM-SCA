import numpy as np

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

def hamming_weight(x):
    return bin(x).count('1')

HW = np.vectorize(hamming_weight)


def univariate_dataset(n_d, n_b, sigma, n_leakages):
  size = 2**n_b
  classes  = np.random.randint(0,size,size=(n_leakages, n_d))

  zi_classes = classes[:, 0].copy()
  for d in range(1, n_d):
    zi_classes = zi_classes ^ classes[:, d]
  leakages = HW(zi_classes)
  for d in range(1, n_d):
    leakages += HW(classes[:, d])

  leakages = leakages + np.random.normal(loc = 0, scale=sigma, size=(n_leakages,))
  return leakages, classes


def normalization(leakages):
  mean = np.mean(leakages)
  std = np.std(leakages)
  return ((leakages - mean)/std), mean, std

def em_criterion(newparams, params):
  delta_var = np.linalg.norm(params[2] - newparams[2])**2
  delta_mean = np.linalg.norm(params[1] - newparams[1])**2
  if(params[2] < 10**-2):
    print('variance < 1e-2')
  return delta_var + delta_mean

def save_params(params, filename):
  np.save(filename, np.array(params, dtype=object))

####################################################################################################################################
####################################################################################################################################
                                                    #PLOTTING
####################################################################################################################################
####################################################################################################################################

# plot KDE je donne en entrée une liste de samples je plot le KDE
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
def plot_KDE(leakages, bandwidth=1):
  x = np.linspace(-3, 3, 1000)[:,None]
  kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(leakages)
  log_dens = kde.score_samples(x)
  plt.fill(x, np.exp(log_dens), c='black')
  plt.show()

def plot_all_distribution(n_d, n_b, params):
  size = 2**n_b
  points = x = np.linspace(-3, 3, 10000)
  plt.figure(figsize=(10, 6))
  for zi in range(size):
    probas = p_l_zi(params[0], points, zi) #(l , ) ou chaque l est p( l / zi)
    plt.plot(points, probas, label=f'zi = {zi}')
  plt.legend()
  plt.show()

def plot_each_distribution(n_d, n_b, params):
  size = 2**n_b
  points = x = np.linspace(-3, 3, 10000)
  for zi in range(size):
    plt.figure(figsize=(10, 6))
    probas = p_l_zi(params[0], points, zi) #(l , ) ou chaque l est p( l / zi)
    plt.plot(points, probas, label=f'zi = {zi}')
    plt.legend()
    plt.show()

def plot_each_cluster(n_d, n_b, params, zi):
  size = 2**n_b
  points = x = np.linspace(-3, 3, 10000)
  plt.figure(figsize=(10, 6))
  probas = params[0][zi][None, :] * univariate_gaussian_pdf(points, params[1][zi], params[2]) # (l,d)
  for d in range(size**(n_d-1)):
    plt.plot(points, probas[:,d])
  plt.legend()
  plt.show()

def plot_pi(n_d, n_b, sigma, n_test, epochs, profiling_points, init='standard'):

  leakages_profiling, classes_profiling = univariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=max(profiling_points))
  zi_classes_profiling, m_classes_profiling = classes_profiling[:, 0], classes_profiling[:, 1:]

  leakages_test, test_classes = univariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=n_test)
  zi_classes_test, m_classes_test = test_classes[:, 0], test_classes[:, 1:]

  if init=='standard':
    params_init = standard_univariate_init(n_d=n_d, n_b=n_b, subkeys=1)
  elif init=='kmeans':
    params_init = kmeans_univariate_init(n_d=n_d, n_b=n_b, leakages=leakages_profiling, classes=zi_classes_profiling, subkeys=1)

  size = 2**n_b
  plt.figure(figsize=(10, 6))
  pi = []
  for point in profiling_points:

    leakages_train = leakages_profiling[:point]
    zi_classes_train, m_classes_train = classes_profiling[:point, 0], classes_profiling[:point, 1:]
    leakages_train, mean_train, std_train = normalization(leakages_train)

    leakages_test_norm = (leakages_test - mean_train)/ std_train

    params_train = params_init.copy()
    univariate_training(n_d=n_d, n_b=n_b, params=params_train, leakages=leakages_train, classes=zi_classes_train, epochs=epochs)

    pi.append(PI(n_d, n_b, params_train[0], leakages_test_norm, zi_classes_test))

  plt.plot(profiling_points, pi)
  plt.title(f'n_d = {n_d} n_b = {n_b} sigma = {sigma}')
  plt.xlabel('nb of profiling points')
  plt.ylabel(f'PI pi[-1] = {pi[-1]}')
  last_pi = [pi[-1]]*len(pi)
  plt.plot(profiling_points, last_pi, c='red', linestyle='--')
  ax.set_aspect('equal')  # Pour que le cercle ne soit pas aplati
  plt.show()
  return [pi, profiling_points]


####################################################################################################################################
####################################################################################################################################
                                                    #UNIVARIATE_GAUSSIAN
####################################################################################################################################
####################################################################################################################################

def univariate_gaussian_pdf(leakages, mean, var):
    coef = 1/(np.sqrt((var)*(2*np.pi)))
    diff = leakages[:, None] - mean[None, :]
    diff = -0.5 * ((diff*diff)/var)
    return coef * np.exp(diff)

####################################################################################################################################
####################################################################################################################################
                                                    #INIT
####################################################################################################################################
####################################################################################################################################


def univariate_init_a(n_d, n_b):
    size = 2**n_b
    d = (size**(n_d-1))
    return np.full((size, d), 1/d, dtype=np.float64)

def univariate_init_V(sigma = 1):
    return sigma


def standard_univariate_init_E_j(n_d, n_b):
  size = 2**n_b
  d = (size**(n_d-1))
  return np.random.normal(loc=0, scale=1, size=(size, d)).astype(np.float64)


def standard_univariate_init(n_d, n_b, subkeys, sigma=1):
   a = univariate_init_a(n_d, n_b)
   return [(a.copy(), standard_univariate_init_E_j(n_d, n_b), univariate_init_V(sigma)) for _ in range(subkeys)]

from sklearn.cluster import KMeans
def kmeans_univariate_init_E_j(n_d, n_b, leakages, classes):
    size = 2**n_b
    n_clusters = size**(n_d-1)
    clusters = []
    for zi in range(size):
      kmeans  = KMeans(n_clusters=n_clusters)
      kmeans.fit(leakages[classes == zi][:,None])
      clusters.append(kmeans.cluster_centers_.reshape((n_clusters,)))
    return(np.array(clusters))


def kmeans_univariate_init(n_d, n_b, leakages, classes, subkeys, sigma=1):
    return[(univariate_init_a(n_d, n_b), kmeans_univariate_init_E_j(n_d, n_b, leakages, classes), univariate_init_V(sigma)) for _ in range(subkeys)]


####################################################################################################################################
####################################################################################################################################
                                                    #TRAINING
####################################################################################################################################
####################################################################################################################################


# prend les params ( (d,), (d,), 1), les leakages (l,)
# ressort les a_j (l,d)
def univariate_e_step(params, leakages):
  (a_j_zi, E_j_zi, V) = params
  num = a_j_zi[None,:] * univariate_gaussian_pdf(leakages, E_j_zi, V) #(l,d)
  return num / np.sum(num, axis=1, keepdims=True) #(l,d)

# version a_j constant (univariate_m_step_a  )
# a_j de taille (zi, d)
def univariate_m_step(n_d, n_b,params, leakages, classes):
  size = 2**n_b
  (a_j , E_j, V) = params
  new_E_j = np.zeros((E_j.shape[0], E_j.shape[1]), dtype=np.float64)
  new_V = 0
  for zi in range(size):
    leakages_zi = leakages[classes == zi] # (l,stot)
    if(leakages_zi.shape[0] != 0):
      #print("e-step start...")
      a_j_n= univariate_e_step((a_j[zi], E_j[zi], V), leakages_zi) #(l,d)
      #print("e-step end...")
      #print("m-step start...")
      sum_a_j_n = np.sum(a_j_n, axis=0) #(d,)

      new_E_j[zi] = np.dot(a_j_n.T, leakages_zi) / sum_a_j_n

      #diff = leakages[:,samples][:, None, :] - E_j[zi,None, :, :]  #(taille l , d , s)
      diff = leakages_zi[:, None] - new_E_j[zi][None, :] #(l, d)
      cov = a_j_n * diff*diff # (l,d)
      new_V += (np.sum(np.sum(cov, axis=1), axis=0) / leakages_zi.shape[0])
      #print("m-step end...")
    else :
      new_E_j[zi] = E_j[zi]
      print(f'No Leakage value for zi = {zi}')

  new_V = new_V / size
  return (a_j, new_E_j, new_V)

def univariate_training(n_d, n_b, params, leakages, classes, epochs, criterion=False, criterion_bound=1e-4):
  for epoch in range(epochs):
    newparams = univariate_m_step(n_d, n_b, params[0], leakages, classes)
    if(criterion):
      if(em_criterion(newparams, params[0]) < criterion_bound):
        print('critère de fin atteint')
        return
    params[0] = newparams


####################################################################################################################################
####################################################################################################################################
                                                    #METRICS
####################################################################################################################################
####################################################################################################################################


#la proba d'avoir zi sachant l = p(l/zi)

def p_l_zi(params, leakages, zi):
  probas = params[0][zi][None, :] * univariate_gaussian_pdf(leakages, params[1][zi], params[2]) # (l,d)
  return np.sum(probas, axis=1) # (l,)

def log2_p_l_zi(params, leakages, zi):
  return np.log2(p_l_zi(params, leakages, zi) + 1e-12) #(l,)

def log_p_l_zi(params, leakages, zi):
  return np.log(p_l_zi(params, leakages, zi) + 1e-12) #(l,)

def all_p_l_zi(n_d, n_b, params, leakages):
  size = 2**n_b
  probas_l_zi = np.zeros((size, leakages.shape[0]), dtype=np.float64) #(size, l)
  for zi in range(size):
    probas_l_zi[zi] = p_l_zi(params, leakages, zi) #(l,)
  return probas_l_zi #(size, l)

def all_log2_p_l_zi(n_d, n_b, params, leakages):
  return np.log2(all_p_l_zi(n_d, n_b, params, leakages) + 1e-12) #(size, l)

def all_log_p_l_zi(n_d, n_b, params, leakages):
  return np.log(all_p_l_zi(n_d, n_b, params, leakages) + 1e-12) #(size, l)
# j'ai p(zi / l) = p(l / zi) * p(zi) / sum( p(l/zj) * p(zj))
# je dois donc calculer tous les p(l/jz)

# j'ai mes leakages, je prends tous les zi possibles (taille zi) je dois ressortir un truc de taille (zi, l)
# chaque ligne c'est p(zi /l)
def p_zi_l(n_d, n_b, params, leakages):
  size = 2**n_b
  probas_l_zi = all_p_l_zi(n_d, n_b, params, leakages) #(size, l)
  sum_probas_l_zi = np.sum(probas_l_zi, axis=0) + 1e-12 #(l,)
  return probas_l_zi / sum_probas_l_zi[None, :]  #(size, l)

def log_p_zi_l(n_d, n_b, params, leakages):
  return np.log(p_zi_l(n_d, n_b, params, leakages) + 1e-12) #(size, l)

def log2_p_zi_l(n_d, n_b, params, leakages):
  return np.log2(p_zi_l(n_d, n_b, params, leakages) + 1e-12) #(size, l)


def PI(n_d, n_b, params, leakages, classes):
  size = 2**n_b
  non_existing = 0
  log2_probas = log2_p_zi_l(n_d, n_b, params, leakages)
  H_zi_l = 0
  for zi in range(size):
    if( (classes==zi).shape[0] != 0):

      log2_probas_zi = log2_probas[zi, classes==zi] #(lzi,) ou lzi est le nombre de traces ayant la classe ==zi
      H_zi_l += np.sum(log2_probas_zi)/log2_probas_zi.shape[0]
    else:
      non_existing += 1
  return np.log2(size) + H_zi_l/(size-non_existing)


####################################################################################################################################
####################################################################################################################################
                                                    #ATTACK
####################################################################################################################################
####################################################################################################################################


def univariate_attack_dataset(n_d, n_b, sigma, n_leakages):
  size = 2**n_b
  pts = np.random.randint(0,size,size=(n_leakages,))
  keys = np.random.randint(0, size) * np.ones((n_leakages,))
  zi_classes = SBOX[pts.astype(np.int16) ^ keys.astype(np.int16)]
  masks_classes = np.random.randint(0,size,size=(n_leakages, n_d-1))

  xor_classes = zi_classes.copy()
  for d in range(n_d-1):
    xor_classes = xor_classes ^ masks_classes[:, d]

  leakages = HW(xor_classes)
  for d in range(n_d-1):
    leakages += HW(masks_classes[:, d])
  leakages = leakages + np.random.normal(loc = 0, scale=sigma, size=(n_leakages,))
  return leakages, zi_classes, masks_classes, pts, keys

# j'ai une clé des pts, des leakages, les params de mon model je dois retrouver la clée
def attack(n_d, n_b, params, leakages_attack, pts_attack, keys_attack):
  size = 2**n_b
  probas = all_log_p_l_zi(n_d, n_b, params[0], leakages_attack) # (size, l)
  p_k = np.zeros((size,), dtype=np.float64)
  for key in range(size):
    key_ = key * np.ones((pts_attack.shape[0],))
    zi_classes = SBOX[pts_attack.astype(np.int16) ^ key_.astype(np.int16)]
    p_k[key] = np.sum(probas[zi_classes, np.arange(leakages_attack.shape[0])])
  ranking = np.argsort(p_k)
  return ranking[::-1], p_k[ranking][::-1]


def success_rate(n_d, n_b, sigma, subkeys, n_tours,n_p, n_a, init='kmeans'):

  # JE crée le dataset
  # J'entraine les params dessus
  # J'évalue
  leakages_profiling, classes = univariate_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=n_p)
  zi_classes_profiling = classes[:, 0]
  leakages_profiling, mean_profiling, std_profiling = normalization(leakages_profiling)
  if init=='kmeans':
    params_init = kmeans
  elif init=='standard':
    params_init = standard_univariate_init(n_d=n_d, n_b=n_b, subkeys=subkeys)
  univariate_training(n_d=n_d, n_b=n_b, params=params_init, leakages=leakages_profiling, classes=zi_classes_profiling, epochs=10, criterion=True, criterion_bound=1e-4)


  success_rate = []
  leakages_attack, zi_classes_attack, masks_classes_attack, pts_attack, keys_attack = univariate_attack_dataset(n_d=n_d, n_b=n_b, sigma=sigma, n_leakages=max(n_a)*n_tours)
  leakages_attack = (leakages_attack - mean_profiling)/std_profiling
  for n_leakages in n_a:
    p=0
    for tour in range(n_tours):
      leakages_A, zi_classes_A, masks_classes_A, pts_A, keys_A = leakages_attack[tour*n_leakages:(tour+1)*n_leakages], zi_classes_attack[tour*n_leakages:(tour+1)*n_leakages], masks_classes_attack[tour*n_leakages:(tour+1)*n_leakages], pts_attack[tour*n_leakages:(tour+1)*n_leakages], keys_attack[tour*n_leakages:(tour+1)*n_leakages]
      ranking, _ = attack(n_d=n_d, n_b=n_b, params=params_init, leakages_attack=leakages_A, subkeys=1, pts_attack=pts_A, keys_attack=keys_A)
      if ranking[0] == keys_attack[0]:
        p += 1
    success_rate.append(p/n_tours)
    print(f'n_leakages = {n_leakages}, p = {p/n_tours}')
  return success_rate, n_a


