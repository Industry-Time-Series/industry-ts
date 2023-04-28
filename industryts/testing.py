"""
    Temporary file for testing purposes.
"""

import models.univariate as unimodels
import matplotlib.pyplot as plt
import numpy as np

from generation.synthetic import ar_process


N_EXPERIMENTS = 400
TRUE_COEFS = [0.4]
SAMPLES = 500

coefs = []

for _ in range(N_EXPERIMENTS):
    data = ar_process(TRUE_COEFS, samples=SAMPLES, noise=0)
    ar_model = unimodels.AutoRegressive(p=1, bias=False)
    ar_model.fit(data)
    coefs.append(ar_model.coef)

coefs = np.array(coefs).squeeze()
plt.hist(coefs, bins=100)

plt.axvline(x=TRUE_COEFS[0], color='red')
plt.axvline(x=np.mean(coefs), color='green')

plt.legend(["True coefficient", "Mean of coefficients"])
plt.title("Histogram of coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Frequency")
plt.show()


print("Mean: ", np.mean(coefs))
print("Standard deviation: ", np.std(coefs))
