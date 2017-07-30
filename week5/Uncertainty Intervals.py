import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('./example_wp_peyton_manning.csv')
#print(df.head())

df['y'] = np.log(df['y'])
#print(df.head())

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)

#Uncertainty in the trend
forecast = Prophet(interval_width=0.95).fit(df).predict(future)

#Uncertainty in seasonality
m = Prophet(mcmc_samples=500)
forecast = m.fit(df).predict(future)
m.plot_components(forecast)
plt.show()
