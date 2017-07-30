import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('/home/yeolpyeong/pragmaticML/week5/example_wp_peyton_manning.csv')
#print(df.head())

df['y'] = np.log(df['y'])
#print(df.head())

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)

#Adjusting trend flexibility
m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
m.plot(forecast)
plt.show()

m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(df).predict(future)
m.plot(forecast)
plt.show()

#Specifying the locations of the changepoints
m = Prophet(changepoints=['2014-01-01'])
forecast = m.fit(df).predict(future)
m.plot(forecast)
plt.show()