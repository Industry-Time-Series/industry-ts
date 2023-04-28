"""
    Temporary file for testing purposes.
"""

import models.univariate as unimodels
import matplotlib.pyplot as plt
import numpy as np

from generation.synthetic import ar_process, ma_process


N_EXPERIMENTS = 500
TRUE_COEFS = [0.4]
SAMPLES = 600
PROCESS = 'MA'
BIAS = True

coefs = []
if BIAS:
    biases = []

for _ in range(N_EXPERIMENTS):
    if PROCESS == 'AR':
        data = ar_process(TRUE_COEFS, samples=SAMPLES, noise=0)
        ar_model = unimodels.AutoRegressive(p=1, bias=BIAS)
        ar_model.fit(data)
        if BIAS:
            biases.append(ar_model.coef[0])
        coefs.append(ar_model.coef[1:])
    elif PROCESS == 'MA':
        data = ma_process(TRUE_COEFS, samples=SAMPLES, noise=0)
        ma_model = unimodels.MovingAverage(q=1, bias=BIAS)
        ma_model.fit(data)
        if BIAS:
            biases.append(ma_model.coef[0])
        coefs.append(ma_model.coef[1:])
    else:
        raise ValueError("Invalid process type.")

coefs = np.array(coefs).squeeze()

plt.hist(coefs, bins=100)

plt.axvline(x=TRUE_COEFS[0], color='red')
plt.axvline(x=np.mean(coefs), color='green')

plt.legend(["True coefficient", "Mean of coefficients"])
plt.title("Histogram of coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Frequency")
plt.show()


print("Mean of coefficient estimates: ", np.mean(coefs))
print("Standard deviation: ", np.std(coefs))

if BIAS:
    print("Mean of bias estimates: ", np.mean(biases))
    print("Standard deviation: ", np.std(biases))

# Plot the data
plt.plot(data)
plt.title(PROCESS + f"({len(TRUE_COEFS)}) process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
