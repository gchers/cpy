# CPy -- Conformal Prediction

Python implementation of Conformal Predictors [1,2].


## Example

```python
import numpy as np
from cpy import cp
from cpy.nonconformity_measures import knn

# Creating a dataset.
np.random.seed(0)
X = np.random.random_sample((20, 2))
X[:10,] += 1.5
Y = [0]*10 + [1]*10
x_test = np.random.random_sample(2)
y_test = 1

# Conformal Prediction using k-NN as a nonconformity measure,
# and 10% significance level.
ncm = knn.KNN(k=2)
epsilon = 0.1
cp = cp.CP(ncm)
prediction = cp.predict_labelled(x_test, X, Y, epsilon)

print 'Prediction set: {}'.format(prediction)
```


## References
[1] Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer Science & Business Media, 2005.

[2] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction." The Journal of Machine Learning Research 9 (2008): 371-421.
