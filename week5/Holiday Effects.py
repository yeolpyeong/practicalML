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


#Modeling Holidays
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

#print(playoffs)
#print(superbowls)
#print(holidays)

m = Prophet(holidays=holidays)
forecast = m.fit(df).predict(future)

forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-10:]

m.plot_components(forecast)
#plt.show()

#Prior scale for holidays and seasonality
m = Prophet(holidays=holidays, holidays_prior_scale=1).fit(df)
forecast = m.predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
    ['ds', 'playoff', 'superbowl']][-10:]
