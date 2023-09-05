"""
    Temporary file for testing purposes.
"""

import models.univariate as unimodels
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from generation.synthetic import ar_process, ma_process


N_EXPERIMENTS = 2
TRUE_COEFS = [0.4, 0.1]
SAMPLES = 500
PROCESS = 'AR'
BIAS = True

PREDICT_WITH_AVG_COEFS = False

coefs = []
if BIAS:
    biases = []

for _ in range(N_EXPERIMENTS):
    if PROCESS == 'AR':
        data = ar_process(TRUE_COEFS, samples=SAMPLES, noise=0)
        ar_model = unimodels.AutoRegressive(p=len(TRUE_COEFS), bias=BIAS)
        ar_model.fit(data)
        if BIAS:
            biases.append(ar_model.coef[0])
        coefs.append(ar_model.coef[1:])
    elif PROCESS == 'MA':
        data = ma_process(TRUE_COEFS, samples=SAMPLES, noise=0)
        ma_model = unimodels.MovingAverage(q=len(TRUE_COEFS), bias=BIAS)
        ma_model.fit(data)
        if BIAS:
            biases.append(ma_model.coef[0])
        coefs.append(ma_model.coef[1:])
    else:
        raise ValueError("Invalid process type.")

coefs = np.array(coefs).squeeze()
if BIAS:
    biases = np.array(biases).squeeze()

for i in range(len(TRUE_COEFS)):
    plt.hist(coefs[:, i], bins=100)

    plt.axvline(x=TRUE_COEFS[i], color='red')
    plt.axvline(x=np.mean(coefs[:, i]), color='green', linestyle='--')

    plt.legend(["True coefficient", "Mean of coefficients"])
    plt.title("Histogram of coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Frequency")
    plt.show()

    print("Mean of coefficient estimates: ", np.mean(coefs[:, i]))
    print("Standard deviation: ", np.std(coefs[:, i]))

if BIAS:
    plt.hist(np.array(biases), bins=100)

    plt.axvline(x=TRUE_COEFS[0], color='red')
    plt.axvline(x=np.mean(biases), color='green', linestyle='--')

    plt.legend(["True bias", "Mean of bias estimates"])
    plt.title("Histogram of biases")
    plt.xlabel("Bias")
    plt.ylabel("Frequency")
    plt.show()

    print("Mean of bias estimates: ", np.mean(biases))
    print("Standard deviation: ", np.std(biases))

# Make predictions using average coefficients
if PREDICT_WITH_AVG_COEFS:
    avg_coefs = np.mean(coefs, axis=0)
    if BIAS:
        avg_bias = np.mean(biases)
    else:
        avg_bias = 0

    if PROCESS == 'AR':
        ar_model.coef = np.concatenate([[avg_bias], avg_coefs])
    elif PROCESS == 'MA':
        pass
    else:
        raise ValueError("Invalid process type.")

ar_model.coef = np.concatenate([[0], TRUE_COEFS])

# Make predictions using one step ahead forecasting
predictions = np.zeros(len(data))
predictions[:len(TRUE_COEFS)] = data[:len(TRUE_COEFS)]
for i in range(len(TRUE_COEFS), len(data)):
    if PROCESS == 'AR':
        initial_condition = data[(i - len(TRUE_COEFS)):i]
        predictions[i] = ar_model.forecast(initial_condition)
    elif PROCESS == 'MA':
        pass
    else:
        raise ValueError("Invalid process type.")

plt.plot(data)
plt.plot(predictions, linestyle='--', color='red')
plt.title(PROCESS + f"({len(TRUE_COEFS)}) process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(["Data", "Predictions", "Free run predictions"])
plt.show()

# Calculate mean absolute error
mae = np.mean(np.abs(data - predictions))
print("Mean absolute error: ", mae)

# Plot the residuals, their histogram, acf and qq plot in subplots
residuals = data - predictions

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(residuals)
axes[0, 0].set_title("Residuals")
axes[0, 1].hist(residuals, bins=50)
axes[0, 1].set_title("Histogram of residuals")
axes[1, 0].acorr(residuals, maxlags=10)
axes[1, 0].set_title("Autocorrelation of residuals")
axes[1, 1].set_title("QQ plot of residuals")
sm.qqplot(residuals, ax=axes[1, 1], line='s')
# set figure size
fig.set_size_inches(10, 8)

plt.show()

# Check normality of residuals
jb_test = sm.stats.stattools.jarque_bera(residuals)
print("Jarque-Bera test: ")
print("Statistic: ", jb_test[0])
print("p-value: ", jb_test[1])
print("Skewness: ", jb_test[2])
print("Kurtosis: ", jb_test[3])

if jb_test[1] < 0.05:
    print("Residuals are not normally distributed.")
else:
    print("Residuals are normally distributed.")
