# technicalmethods
## A collection of Technical Analysis tools

&nbsp;

### The library currently provides methods for:
  - RSI
  - Bollinger Bands
  - MACD
  - ADX
  - Williams %R
  - Stochastics
  - ATR
  - EMA
  - Trend Channel 

&nbsp;  

### Installation
Install from PyPI:
```
$ pip install technicalmethods
```

&nbsp;

To install in new environment using anaconda:
```
$ conda create --name techmeth
```
Activate new environment
```
$ activate techmeth
```
Install Python
```
(techmeth) $ conda install python==3.8.8
```
Install Spyder
```
(techmeth) $ conda install spyder==4.2.5
```
Install Pandas
```
(techmeth) $ conda install pandas==1.1.4
```


Install technicalmethods
```
(techmeth) $ python -m pip install technicalmethods
```

&nbsp;

### Setup
Import csv of OHLC data using pandas
```
import pandas as pd
msft = pd.read_csv('MSFT.csv')
```
Import technicalmethods and initialize an Indicators object 
```
import technicalmethods.methods as meth
ind = meth.Indicators()
```
Calculate a 14 day RSI on the closing prices
```
msft['RSI14'] = ind.RSI(close=msft['Close'], time_period=14)
```