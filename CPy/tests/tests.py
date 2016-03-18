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
        self.assertAlmostEqual(score, utils.knn_score, 'kNN NCM returned wrong score.')

    def test_KDE(self):
        h = utils.kde_h
        kde = KDE.KDE(h, 'gaussian')
        score = kde.compute(self.x, self.X)
        self.assertAlmostEqual(score, utils.kde_score, 'KDE NCM returned wrong score.')

class TestCP(unittest.TestCase):
    
    def setUp(self):
        self.seed = 0
        k = utils.knn_k
        h = utils.kde_h
        knn = kNN.KNN(k)
        kde = KDE.KDE(h, 'gaussian')
        self.cp_knn = CP.CP(knn, smooth=False)
        self.cp_kde = CP.CP(kde, smooth=False)
        self.epsilon = utils.epsilon
    
    def test_CP_knn_pvalue(self):
        x = utils.x
        X = utils.X
        pvalue = self.cp_knn.calculate_pvalue(x, X)
        self.assertAlmostEqual(pvalue, utils.cp_knn_pvalue,
                               'P-value for kNN NCM has a wrong value.')
    
    def test_CP_kde_pvalue(self):
        x = utils.x
        X = utils.X
        pvalue = self.cp_kde.calculate_pvalue(x, X)
        self.assertAlmostEqual(pvalue, utils.cp_kde_pvalue,
                               'P-value for KDE NCM has a wrong value.')

    def test_CP_knn_unlabelled(self):
        utils.set_seed(self.seed)
        N = utils.N
        K = utils.K
        X, x_test = utils.generate_unlabelled_dataset(N, K)
        pred = self.cp_knn.predict_unlabelled(x_test, X, self.epsilon)
        self.assertTrue(pred, 'CP with kNN NCM mispredicted an object in unlabelled setting.')

    def test_CP_kde_unlabelled(self):
        utils.set_seed(self.seed)
        N = utils.N
        K = utils.K
        X, x_test = utils.generate_unlabelled_dataset(N, K)
        pred = self.cp_kde.predict_unlabelled(x_test, X, self.epsilon)
        self.assertTrue(pred, 'CP with KDE NCM mispredicted an object in unlabelled setting.')

    def test_CP_knn_labelled_1(self):
        """Consider objects with dimension 1.
        """
        utils.set_seed(self.seed)
        N = utils.N
        K = 1
        X, Y, x_test, y_test = utils.generate_labelled_dataset(N, K)
        pred = self.cp_knn.predict_labelled(x_test, X, Y, self.epsilon)
        self.assertIn(y_test, pred,
                      'CP with kNN NCM mispredicted an object. Objects had dimension 1.')

    def test_CP_kde_labelled_1(self):
        """Consider objects with dimension 1.
        """
        utils.set_seed(self.seed)
        N = utils.N
        K = 1
        X, Y, x_test, y_test = utils.generate_labelled_dataset(N, K)
        pred = self.cp_kde.predict_labelled(x_test, X, Y, self.epsilon)
        self.assertIn(y_test, pred,
                      'CP with KDE NCM mispredicted an object. Objects had dimension 1.')
    
    def test_CP_knn_labelled(self):
        utils.set_seed(self.seed)
        N = utils.N
        K = utils.K
        X, Y, x_test, y_test = utils.generate_labelled_dataset(N, K)
        pred = self.cp_knn.predict_labelled(x_test, X, Y, self.epsilon)
        self.assertIn(y_test, pred,
                      'CP with kNN NCM mispredicted an object. Objects had dimension {}'.format(K))
        
    def test_CP_kde_labelled(self):
        utils.set_seed(self.seed)
        N = utils.N
        K = utils.K
        X, Y, x_test, y_test = utils.generate_labelled_dataset(N, K)
        pred = self.cp_kde.predict_labelled(x_test, X, Y, self.epsilon)
        self.assertIn(y_test, pred,
                      'CP with KDE NCM mispredicted an object. Objects had dimension {}'.format(K))


if __name__ == '__main__':
    unittest.main()
