"""Data for unit tests."""
import numpy as np

# Seed for PRG.
seed = 0

# Data for testing nonconformity measures and p-values calculation.
x = [0.227606514087, 3.19756309191, 3.99676324207, 2.2618963427, 4.9096869819,
     9.36106523154, 9.2442539595, 5.69251642812, 1.74739742778, 5.12675778168]

X = [[0.241379816431, 3.07772645586, 0.266095771065, 6.96496842189, 2.05604416791,
      4.53735749296, 0.678355345234, 5.11754104077, 7.36610695022, 0.220108218584],
    [1.07744114334, 7.58963734813, 2.80211737997, 7.74580533536, 2.55748547234,
     6.59702669393, 0.0217942493459, 2.87192707468, 8.1254030898, 0.12792624235],
    [6.43718139931, 4.63421513939, 8.83766794716, 2.13536933171, 5.00237682514,
     1.88638458727, 0.495493646824, 8.26242412914, 4.66560025836, 4.19530119089],
    [6.37843328926, 4.62934620171, 7.15672856837, 6.88007474688, 9.08575265749,
     8.20903783949, 9.77601785473, 7.01985296071, 4.26624048173, 2.15619714096],
    [1.26382287509, 3.91332311469, 5.94718827214, 0.605738745779, 4.64205323548,
     7.16015438495, 6.32264821969, 9.61285117591, 4.90480877531, 8.64990456322]]

knn_score = 61.606577966581696
knn_k = 3
kde_h = 0.1
kde_score =  -664903800.6690541

cp_knn_pvalue = 0.8333333333333334
cp_kde_pvalue = 1.0


# Generators for testing prediction.
# The data is normally distributed. We consider three distributions,
# d1, d2, d3, which can be trivially distinguished.
d1 = {'mu': 0.0, 'sigma': 1.0}
d2 = {'mu': 2.0, 'sigma': 0.5}
d3 = {'mu': 3.0, 'sigma': 0.005}
d = [d1, d2, d3]

N = 20          # Number of objects/examples.
K = 10          # Size of each object.

epsilon = 0.05

def set_seed(seed):
    """Set seed for PRNG.
    """
    np.random.seed(seed)

def generate_unlabelled_dataset(n, k):
    """Returns an array of n objects from the first
    distribution, a test object that comes from the same
    distribution.
    Each object has size k.
    """
    X = []
    for i in range(n):
        x = sample_distribution(d[0], k)
        X.append(x)
    X = np.array(X)
    x_test = sample_distribution(d[0], k)
    
    return X, x_test

def generate_labelled_dataset(n, k):
    """Generates an array of n objects and labels
    (dataset), and a test object.
    Each object has size k.
    """
    X = []
    Y = []
    for i in range(n):
        y = i%3
        x = sample_distribution(d[y], k)
        Y.append(y)
        X.append(x)
    y_test = 2
    x_test = sample_distribution(d[y_test], k)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, x_test, y_test
        


def sample_distribution(d, k):
    """Returns k samples from a normal distribution with mean
    d['mu'] and standard deviation d['sigma'].
    """
    return np.random.normal(d['mu'], d['sigma'], k)
