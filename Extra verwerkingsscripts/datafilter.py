#just try out some filters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns


# create random data
N_SAMPLES = 100
x = np.linspace(0, 6, N_SAMPLES)
signal = np.sin(x)
noise = np.random.normal(0, 0.15, N_SAMPLES)
measurements = signal + noise


# # plot noisy data
# plt.plot(x, measurements, 'o', color='black')
# plt.plot(x, signal, 'o', color='green')
# plt.plot(x, noise, 'o', color='red')
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.title('Example dataset')
# plt.show()

# Let's try the LOESS filter
filtered = lowess(measurements, x, frac=0.2)
plt.plot(filtered[:, 0], filtered[:, 1], 'r-', linewidth=2)
plt.plot(x, measurements, 'o', color='black')
plt.title('Example dataset with LOESS filter (frac=0.2)')
plt.show()

# Example of some random data distributions
data = np.random.randn(3, 100)
sns.displot(data=pd.DataFrame({"data": data.ravel(),
                               "column": np.repeat(np.arange(data.shape[0]), data.shape[1])}),
            x="data", col="column", kde=True, color='blueviolet', height=3)
plt.show()

# Let's find the data distribution
d = {'measurements': measurements, 'time': x}
df = pd.DataFrame(data=d)
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.displot(data=df, x='measurements', kind="hist", bins = 10, aspect = 1.5)
plt.title('Distribution of data')
plt.show()


