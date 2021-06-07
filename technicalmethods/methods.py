import numpy as np
import pandas as pd
# Suppress SettingWithCopyWarning caused by slicing DataFrame
pd.options.mode.chained_assignment = None


class Indicators():    
   
    @classmethod
    def MACD(cls, close, fast, slow, signal):
        """
        Calculate Moving Average Convergence Divergence 

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        fast : Int
            Speed of fast moving average.
        slow : Int
            Speed of slow moving average.
        signal : Int
            Speed of signal line moving average.

        Returns
        -------
        Series
            Time series of MACD line (price velocity).
        Series
            Time series of MACD Signal line.
        Series
            Time series of MACD Histogram.

        """
        # Create DataFrame from the closing prices
        df = pd.DataFrame(close)
        
        # Calculate fast Exponential Moving Average
        df['EMA_fast'] = cls.EMA(
            input_series=df['Close'], time_period=fast, slow_macd=slow)
        
        # Calculate slow Exponential Moving Average
        df['EMA_slow'] = cls.EMA(input_series=df['Close'], time_period=slow)
        
        # Calculate Price Velocity as the difference between the fast and slow 
        # averages
        df['MACD'] =  df['EMA_fast'] - df['EMA_slow']
        
        # Calculate the Signal line as the exponentially smoothed Price 
        # Velocity (using the signal parameter for period)
        df['Signal'] = cls.EMA(df['MACD'], time_period=signal)
        
        # Calculate the MACD Histogram as the Price Velocity less the Signal 
        # line
        df['Histogram'] = df['MACD'] - df['Signal']
        
        return df['MACD'], df['Signal'], df['Histogram']
    
    
    @classmethod
    def RSI(cls, close, time_period):
        """
        Calculate Relative Strength Index

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        time_period : Int
            Lookback period to average over.

        Returns
        -------
        Series
            Time series of RSI values.

        """
        # Create DataFrame from the closing prices
        df = pd.DataFrame(close)

        # Create columns of 1-day gains and losses
        df['Up'] = np.where(
            np.isnan(df['Close'] - df['Close'].shift(1)), 
            np.nan, 
            np.where(df['Close'] - df['Close'].shift(1) > 0, 
                     df['Close'] - df['Close'].shift(1), 
                     0))
        
        df['Down'] = np.where(
            np.isnan(df['Close'] - df['Close'].shift(1)), 
            np.nan, 
            np.where(df['Close'] - df['Close'].shift(1) < 0, 
                     -(df['Close'] - df['Close'].shift(1)), 
                     0))
        
        # Create columns to average the gains and losses over the specified 
        # period
        df['Up_Avg'] = cls.EMA(
            input_series=df['Up'], time_period=time_period, wilder=True)
        
        df['Down_Avg'] = cls.EMA(
            input_series=df['Down'], time_period=time_period, wilder=True)
               
        # Calculate Relative Strength
        df['Relative_Strength'] = df['Up_Avg'] / df['Down_Avg']
        df['RSI'] = 100 - 100 / (1 + df['Relative_Strength'])
    
        return df['RSI']


    @classmethod
    def ADX(cls, high, low, close, time_period):
        """
        Calculate Average Directional Movement Index

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        time_period : Int
            Lookback period.

        Returns
        -------
        Series
            Time series of ADX values.

        """
        
        # Create DataFrame with Price Fields
        df = cls._create_dataframe(close, high, low)
        
        # Add True Range calculation
        df['True_Range'] = cls._true_range(df)
        
        # Calculate True Range for the day
        df['TR_1'] = cls.EMA(
            input_series=df['True_Range'], time_period=1, wilder=True, 
            average=False)
        
        # Calculate Directional Movement columns  
        df['pos_shift'] = np.where(
            df['High'] - df['High'].shift() > 0, 
            df['High'] - df['High'].shift(), 
            np.where(np.isnan(df['High'] - df['High'].shift()), 
                     np.nan, 
                     0))
        df['neg_shift'] = np.where(
            df['Low'] - df['Low'].shift() < 0, 
            -(df['Low'] - df['Low'].shift()), 
            np.where(np.isnan(df['Low'] - df['Low'].shift()), 
                     np.nan, 
                     0))
        
        df['DM_plus_1'] = np.where(
            np.isnan(df['pos_shift']), 
            np.nan,
            np.where(df['pos_shift'] > df['neg_shift'],
                     df['pos_shift'], 
                     0))
        
        df['DM_minus_1'] = np.where(
            np.isnan(df['neg_shift']),
            np.nan,
            np.where(df['pos_shift'] < df['neg_shift'],
                     df['neg_shift'], 
                     0))
        
        # Calculate True Range for the specified period
        df['TR_period'] = cls.EMA(
            input_series=df['True_Range'], time_period=time_period, 
            wilder=True, average=False)
        
        # Calculate Directional Movement for the specified period
        df['DM_plus_period'] = cls.EMA(
            input_series=df['DM_plus_1'], time_period=time_period, wilder=True, 
            average=False)
        
        df['DM_minus_period'] = cls.EMA(
            input_series=df['DM_minus_1'], time_period=time_period, 
            wilder=True, average=False)
        
        # Calculate Directional Indicator as Directional Movement / True Range 
        df['DI_plus_period'] = (
            df['DM_plus_period'] / df['TR_period'] ) * 100
        
        df['DI_minus_period'] = (
            df['DM_minus_period'] / df['TR_period'] ) * 100
        
        # Calculate the Directional Indicator difference
        df['DI_diff'] = np.abs(df['DI_plus_period'] - df['DI_minus_period'])
        
        # Calculate the Directional Indicator sum
        df['DI_sum'] = df['DI_plus_period'] + df['DI_minus_period']
        
        # Calculate the Directional Movement Index as the ratio of DI_diff and 
        # DI_sum
        df['DX'] = (df['DI_diff'] / df['DI_sum']) * 100
        
        # Calculate the Average Directional Movement Index taking an EMA over 
        # the selected period 
        df['ADX'] = cls.EMA(
            input_series=df['DX'], time_period=time_period, wilder=True)
        
        df['ADXR'] = (df['ADX'] + df['ADX'].shift(time_period)) / 2
    
        return df['ADX']
    
    
    @classmethod
    def williams_r(cls, high, low, close, time_period):
        """
        Calculate Williams %R

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        time_period : Int
            Lookback period.

        Returns
        -------
        Series
            Time series of %R.

        """
        # Create DataFrame with Price Fields
        df = cls._create_dataframe(close, high, low)
        
        # Calculate n-day highs and lows
        df['nd_low'] = df['Low'].rolling(time_period).min()
        df['nd_high'] = df['High'].rolling(time_period).max()
        
        
        df['%R'] = -100 * ((df['nd_high'] - df['Close']) / 
                                  (df['nd_high'] - df['nd_low']))
    
        return df['%R']
    
    
    @classmethod
    def stochastic(cls, high, low, close, fast_k_period, fast_d_period=3, 
                   slow_k_period=3, slow_d_period=3, output_type='slow'):
        """
        Calculate Stochastic Oscillator

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        fast_k_period : Int
            Lookback period to calculate %K.
        fast_d_period : Int
            Lookback period to calculate %D by smoothing %K. The default is 3.
        slow_k_period : Int
            Lookback period to calculate Slow %K by smoothing %K. The default 
            is 3.
        slow_d_period : Int
            Lookback period to calculate Slow %D by smoothing %D. The default 
            is 3.
        output_type : STR, optional
            Whether to output Fast or Slow stochastics. The default is 'slow'.

        Returns
        -------
        Series
            Time Series of %K and %D.

        """
        # Create DataFrame with Price Fields
        df = cls._create_dataframe(high, low, close)
                
        # Calculate n-day highs and lows
        df['nd_low'] = df['Low'].rolling(fast_k_period).min()
        df['nd_high'] = df['High'].rolling(fast_k_period).max()
                
        # Fast stochastics
        # Calculate %K 
        df['%K'] = 100 * (df['Close'] - df['nd_low']) / (df['nd_high'] - df['nd_low'])
        
        # Calculate %D by smoothing %K
        df['%D'] = df['%K'].rolling(fast_d_period).mean()
        
        # Slow stochastics apply smoothing to %K and %D
        df['%K_slow'] = df['%K'].rolling(slow_k_period).mean()
        df['%D_slow'] = df['%D'].rolling(slow_d_period).mean()
        
        if output_type == 'slow':
            return df['%K_slow'], df['%D_slow']
        else:
            return df['%K'], df['%D']
    
    
    @classmethod
    def ATR(cls, high, low, close, time_period):
        """
        Calculate Average True Range

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        time_period : Int
            Lookback period to average over.

        Returns
        -------
        Series
            Time series of Average True Range values.

        """
        # Create DataFrame with Price Fields
        df = cls._create_dataframe(close, high, low)
        
        # Add True Range calculation
        df['True_Range'] = cls._true_range(df)
        
        # Take the Exponentially Weighted Moving Average (using Welles Wilder's 
        # specific technique of 1/N of the new value plus (N-1)/N of the 
        # previous average)
        df['ATR'] = cls.EMA(
            input_series=df['True_Range'], time_period=time_period, 
            wilder=True)
        
        return df['ATR']
    
    
    @staticmethod
    def breakout(df, time_period=20):
        """
        Calculate n-day breakout 

        Parameters
        ----------
        df : DataFrame
            The DataFrame of historical prices.
        time_period : Int, optional
            The lookback window. The default is 20.

        Returns
        -------
        nd_low : Series
            The array of n-day lows.
        nd_high : Series
            The array of n-day highs.
        flag : Series
            Indicator whether to be long (1) or short (-1).

        """
        
        # Calculate n-day highs and lows
        nd_low = np.array(df['Low'].rolling(time_period).min())
        nd_high = np.array(df['High'].rolling(time_period).max())
        
        # Create start point from first valid number
        start = np.where(~np.isnan(nd_high))[0][0]
       
        # Create numpy array of zeros to store positions
        flag = np.array([0]*len(nd_high))
        
        for row in range(start + 1, len(nd_high)):
            if (df['High'][row] > nd_high[row-1]) or (
                    flag[row-1] == 1 and df['Low'][row] > nd_low[row-1]):
                flag[row] = 1
    
            if (df['Low'][row] < df['nd_low'][row-1]) or (
                    flag[row-1] == -1 and df['High'][row] < nd_high[row-1]):
                flag[row] = -1
              
        return nd_low, nd_high, flag
    
    
    @staticmethod
    def _true_range(df):
        """
        Calculates the 1 day True Range given High, Low and Close prices.

        Parameters
        ----------
        df : DataFrame
            Input dataframe of High, Low and Close data.

        Returns
        -------
        Series
            1 day True Range.

        """
        # Suppress SettingWithCopyWarning caused by slicing DataFrame
        pd.options.mode.chained_assignment = None
       
        # Calculate the high minus low, absolute value of high minus yesterdays 
        # close and the absolute value of the low minus yesterdays close
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Close_Low'] = np.abs(df['Low'] - df['Close'].shift())
        
        # Concatenate these 3 series
        ranges = pd.concat([df['High_Low'], 
                            df['High_Close'], 
                            df['Close_Low']], 
                           axis=1)
        
        # Calculate the True Range as the maximum of these 3 series
        df['True_Range'] = ranges.max(skipna=False, axis=1)
        
        # Suppress SettingWithCopyWarning caused by slicing DataFrame
        #pd.options.mode.chained_assignment = "warn"
        
        return df['True_Range']
    
    
    @staticmethod
    def _create_dataframe(*fields):
        """
        Create new DataFrame from price series'

        Parameters
        ----------
        *fields : Tuple of Series
            Takes a number of price fields and concatenates into a Pandas 
            DataFrame.

        Returns
        -------
        df : DataFrame
            Combine price data fields.

        """
        # Create DataFrame from concatenating price columns
        df = pd.concat(fields, axis=1)
        
        return df
    
    
    @staticmethod
    def EMA(input_series, time_period, wilder=False, average=True, 
            slow_macd=None):
        """
        Calculate Exponentially Weighted Moving Average  

        Parameters
        ----------
        input_series : Series
            Prices to be smoothed.
        time_period : Int
            Length of lookback period.
        wilder : Bool, optional
            Whether to use Welles Wilder's smoothing method (used in ATR, ADX 
            and RSI). The default is False.
        average : Bool, optional
            Whether to calculate an average (the sum is used in the ADX). The 
            default is True.
        slow_macd : Int, optional
            If the fast MACD is being calculated, this ensures that the fast 
            and slow lines are correctly synchronised. The default is None.

        Returns
        -------
        output_series : Series
            The smoothed price data.

        """
        # Calculate exponential smoothing factor     
        alpha = (2 / (time_period + 1))
        
        # Set starting point to first valid true range calculation
        start = np.where(~np.isnan(input_series))[0][0] 

        if slow_macd is not None:
            start = start + slow_macd - time_period
        
        # Create empty array to store atr values
        output_series = np.array([np.nan]*len(input_series))
        
        if average:
            # Initialize with Simple Moving Average
            output_series[start + time_period - 1] = input_series[
                start:(start + time_period)].mean() 
            
            # Smooth the series using periods specified
            for row in range(start + time_period, len(input_series)):
                # (using Welles Wilder's specific technique of 1/N of the new 
                # value plus (N-1)/N of the previous average)
                if wilder:
                    output_series[row] = (
                        (input_series[row] 
                         + output_series[row - 1] * (time_period - 1)) 
                        / time_period)
                        
                else:
                    output_series[row] = (
                        input_series[row] * alpha 
                        + (output_series[row - 1] * (1 - alpha)))

        # Exponential sum used in ADX calculation
        else:
            # Initialize with sum
            output_series[start + time_period - 1] = (
                input_series[start:(start + time_period - 1)].sum() 
                - (input_series[start:(start + time_period - 1)].sum() 
                   / time_period)
                + input_series[start + time_period - 1])
        
            # Smooth the True Range using periods specified
            for row in range(start + time_period, len(input_series)):
                output_series[row] = (input_series[row]
                                      + output_series[row - 1] 
                                      - (output_series[row - 1] / time_period))
        
        return output_series

 