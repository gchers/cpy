import utils
import unittest
from ..CP import CP
from ..nonconformity_measures import kNN, KDE

class TestNCM(unittest.TestCase):
    
    def setUp(self):
        self.x = utils.x
        self.X = utils.X
        
    
    def test_kNN(self):
        k = len(self.X)
        knn = kNN.KNN(k)
        score = knn.compute(self.x, self.X)
        self.assertAlmostEqual(score, utils.knn_score)

    def test_KDE(self):
        h = utils.kde_h
        kde = KDE.KDE(h, 'gaussian')
        score = kde.compute(self.x, self.X)
        self.assertAlmostEqual(score, utils.kde_score)

class TestCP(unittest.TestCase):
    
    def setUp(self):
        k = utils.knn_k
        h = utils.kde_h
        knn = kNN.KNN(k)
        kde = KDE.KDE(h, 'gaussian')
        self.cp_knn = CP.CP(knn, smooth=True)
        self.cp_kde = CP.CP(kde, smooth=True)
    
    def test_CP_pvalue(self):
        pass
    
    def test_CP_labelled(self):
        pass
    
    def test_CP_unlabelled(self):
        pass


if __name__ == '__main__':
    unittest.main()
