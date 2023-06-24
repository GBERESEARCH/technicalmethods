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

Install in a new environment using Python venv:

Create base environment of Python 3.11
```
$ py -3.11 -m venv .venv
```
Activate new environment
```
$ .venv\scripts\activate
```
Ensure pip is up to date
``` 
$ (.venv) python -m pip install --upgrade pip
```
Install Spyder
```
$ (.venv) python -m pip install spyder
```
Install package
```
$ (.venv) python -m pip install technicalmethods
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
(techmeth) $ conda install python==3.9
```
Install Spyder
```
(techmeth) $ conda install spyder
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