# CPy -- Conformal Prediction

Python implementation of Conformal Predictors [1,2].

**Disclaimer** This is code I used for some experiments. You can use it,
but I do not plan to extend it nor maintain it.
If you're looking for a stable CP library, I am maintaining
[a new one](https://github.com/gchers/random-world) in *Rust*,
which is faster and more complete, provides standalone binaries, and soon
include *Python* bindings.

## Example

```python
import numpy as np
from cpy.cp import CP
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
cp = CP(ncm)
prediction = cp.predict_labelled(x_test, X, Y, epsilon)

print 'Prediction set: {}'.format(prediction)
```


## References
[1] Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer Science & Business Media, 2005.

[2] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction." The Journal of Machine Learning Research 9 (2008): 371-421.
