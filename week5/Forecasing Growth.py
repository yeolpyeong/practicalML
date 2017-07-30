import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('./example_wp_peyton_manning.csv')
#print(df.head())

df['y'] = np.log(df['y'])
#print(df.head())

df['cap'] = 8.5

m = Prophet(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
m.plot(fcst)
plt.show()
