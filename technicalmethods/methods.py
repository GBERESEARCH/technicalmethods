"""
Calculate various Technical Indicators

"""

import copy
import numpy as np
import pandas as pd
from scipy.stats import linregress
# Suppress SettingWithCopyWarning caused by slicing DataFrame
#pd.options.mode.chained_assignment = None
# pylint: disable=invalid-name

class Indicators():
    """
    Calculate various Technical Indicators

    """
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

        # Calculate fast Exponential Moving Average
        ema_fast = cls.EMA(
            input_series=close, time_period=fast, slow_macd=slow)

        # Calculate slow Exponential Moving Average
        ema_slow = cls.EMA(input_series=close, time_period=slow)

        # Calculate Price Velocity as the difference between the fast and slow
        # averages
        macd =  ema_fast - ema_slow

        # Calculate the Signal line as the exponentially smoothed Price
        # Velocity (using the signal parameter for period)
        signal = cls.EMA(macd, time_period=signal)

        # Calculate the MACD Histogram as the Price Velocity less the Signal
        # line
        histogram = macd - signal

        return macd, signal, histogram


    @classmethod
    def bbands(cls, close, high=None, low=None, time_period=20, sd_up=2,
               sd_down=2, simple_ma=True, only_close=True):
        """
        Calculate Bollinger Bands - Upper, Lower and Mid Moving Average

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        high : Series, optional
            Time series of high prices.
        low : Series, optional
            Time series of low prices.
        time_period : Int
            Lookback period to average over.
        sd_up : float, optional
            The number of standard deviations above the ma the upper band
            should be. The default is 2.
        sd_down : float, optional
            The number of standard deviations below the ma the upper band
            should be. The default is 2.
        simple_ma : Bool, optional
            Whether to calculate a simple or exponential moving average.
            The default is True.
        only_close : Bool, optional
            Whether to use just the closing price or the typical price as the
            average of high, low and close. The default is True.

        Returns
        -------
        upper_band : Series
            The band of prices above the moving average.
        ma : Series
            The moving average calculated for the chosen lookback period
        lower_band : Series
            The band of prices below the moving average.

        """
        if only_close:
            sd = close.rolling(window=time_period).std(ddof=0)
            if simple_ma:
                ma = close.rolling(window=time_period).mean()
            else:
                ma = cls.EMA(input_series=close, time_period=time_period)

        else:
            typical_price = (high + low + close) / 3
            sd = typical_price.rolling(window=time_period).std(ddof=0)
            if simple_ma:
                ma = typical_price.rolling(window=time_period).mean()
            else:
                ma = cls.EMA(
                    input_series=typical_price, time_period=time_period)

        upper_band = ma + (sd * sd_up)
        lower_band = ma - (sd * sd_down)

        return upper_band, ma, lower_band


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

        # Create columns of 1-day gains and losses
        gain = np.where(
            np.isnan(close - close.shift(1)),
            np.nan,
            np.where(close - close.shift(1) > 0,
                     close - close.shift(1),
                     0))

        loss = np.where(
            np.isnan(close - close.shift(1)),
            np.nan,
            np.where(close - close.shift(1) < 0,
                     -(close - close.shift(1)),
                     0))

        # Create columns to average the gains and losses over the specified
        # period
        gain_avg = cls.EMA(
            input_series=gain, time_period=time_period, wilder=True)

        loss_avg = cls.EMA(
            input_series=loss, time_period=time_period, wilder=True)

        # Calculate Relative Strength
        relative_strength = gain_avg / loss_avg
        rsi = 100 - 100 / (1 + relative_strength)

        return rsi


    @staticmethod
    def CCI(high, low, close, time_period):
        """
        Calculate Commodity Channel Index

        Parameters
        ----------
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        close : Series
            Time series of closing prices.
        time_period : Int
            Lookback period.

        Returns
        -------
        Series
            Time series of CCI values.

        """
        # 4 Steps

        # 1. Compute todays average using High, Low and Close
        typical_price = (high + low + close) / 3

        # 2. Compute a moving average of the n most recent average prices
        moving_average = typical_price.rolling(time_period).mean()

        # 3. Compute the mean deviation of the n most recent typical prices
        mean_deviation = np.array([0.0]*len(close))
        for i in range(len(close)):
            #mean_deviation[i] = typical_price[i - time_period+1:i+1].mad()
            data = typical_price[i - time_period+1:i+1]
            mean_deviation[i] = np.abs(data - data.mean(axis=0)).mean(axis=0)

        # 4. Compute the Commodity Channel Index
        cci = ((typical_price - moving_average) / (0.015 * mean_deviation))

        return cci


    @classmethod
    def ADX(cls, high, low, close, time_period, dmi=False):
        """
        Calculate Average Directional Movement Index

        Parameters
        ----------
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        close : Series
            Time series of closing prices.
        time_period : Int
            Lookback period.

        Returns
        -------
        Series
            Time series of ADX values.

        """

        # Add True Range calculation
        t_range = cls.true_range(high, low, close)

        # Calculate True Range for the specified period
        tr_period = cls.EMA(
            input_series=t_range, time_period=time_period,
            wilder=True, average=False)

        # Calculate Directional Movement
        dm_plus_period, dm_minus_period = cls._directional_movement(
            high=high, low=low, time_period=time_period)

        # Calculate Directional Indicator as Directional Movement / True Range
        di_plus_period = (dm_plus_period / tr_period) * 100

        di_minus_period = (dm_minus_period / tr_period) * 100

        # Calculate the Directional Indicator difference
        di_diff = np.abs(di_plus_period - di_minus_period)

        # Calculate the Directional Indicator sum
        di_sum = di_plus_period + di_minus_period

        # Calculate the Directional Movement Index as the ratio of DI_diff and
        # DI_sum
        dir_index = (di_diff / di_sum) * 100

        # Calculate the Average Directional Movement Index taking an EMA over
        # the selected period
        adx = cls.EMA(
            input_series=dir_index, time_period=time_period, wilder=True)

        #adxr = (adx + adx.shift(time_period)) / 2

        if dmi:
            return adx, di_plus_period, di_minus_period

        else:
            return adx


    @classmethod
    def _directional_movement(cls, high, low, time_period):
        """
        Calculate Directional Movement

        Parameters
        ----------

        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
        close : Series
            Time series of closing prices.
        time_period : Int
            Lookback period.

        Returns
        -------
        dm_plus_period : TYPE
            DESCRIPTION.
        dm_minus_period : TYPE
            DESCRIPTION.

        """

        # Calculate Directional Movement columns
        pos_shift = np.where(
            high - high.shift() > 0,
            high - high.shift(),
            np.where(np.isnan(high - high.shift()),
                     np.nan,
                     0))
        neg_shift = np.where(
            low - low.shift() < 0,
            -(low - low.shift()),
            np.where(np.isnan(low - low.shift()),
                     np.nan,
                     0))

        dm_plus_1 = np.where(
            np.isnan(pos_shift),
            np.nan,
            np.where(pos_shift > neg_shift,
                     pos_shift,
                     0))

        dm_minus_1 = np.where(
            np.isnan(neg_shift),
            np.nan,
            np.where(pos_shift < neg_shift,
                     neg_shift,
                     0))

        # Calculate Directional Movement for the specified period
        dm_plus_period = cls.EMA(
            input_series=dm_plus_1, time_period=time_period, wilder=True,
            average=False)

        dm_minus_period = cls.EMA(
            input_series=dm_minus_1, time_period=time_period,
            wilder=True, average=False)

        return dm_plus_period, dm_minus_period


    @staticmethod
    def williams_r(high, low, close, time_period):
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

        # Calculate n-day highs and lows
        nd_low = low.rolling(time_period).min()
        nd_high = high.rolling(time_period).max()

        percent_r = -100 * ((nd_high - close) / (nd_high - nd_low))

        return percent_r


    @staticmethod
    def stochastic(high, low, close, fast_k_period, slow_k_period=3,
                   slow_d_period=3, output_type='slow'):
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

        # Calculate n-day highs and lows
        nd_low = low.rolling(fast_k_period).min()
        nd_high = high.rolling(fast_k_period).max()

        # Fast stochastics
        # Calculate %K
        percent_k = (100 * (close - nd_low) / (nd_high - nd_low))

        # Calculate %D by smoothing %K
        percent_d = percent_k.rolling(slow_k_period).mean()

        # Slow stochastics apply smoothing to %K and %D
        percent_k_slow = percent_k.rolling(slow_k_period).mean()
        percent_d_slow = percent_d.rolling(slow_d_period).mean()

        if output_type == 'slow':
            return percent_k_slow, percent_d_slow

        return percent_k, percent_d


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

        # Add True Range calculation
        t_range = cls.true_range(high, low, close)

        # Take the Exponentially Weighted Moving Average (using Welles Wilder's
        # specific technique of 1/N of the new value plus (N-1)/N of the
        # previous average)
        atr = cls.EMA(
            input_series=t_range, time_period=time_period, wilder=True)

        return atr


    @staticmethod
    def breakout(high, low, time_period=20):
        """
        Calculate n-day breakout

        Parameters
        ----------
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.
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
        nd_low = np.array(low.rolling(time_period).min())
        nd_high = np.array(high.rolling(time_period).max())

        # Create start point from first valid number
        start = np.where(~np.isnan(nd_high))[0][0]

        # Create numpy array of zeros to store positions
        flag = np.array([0]*len(nd_high))

        for row in range(start + 1, len(nd_high)):
            if (high[row] >= nd_high[row-1]) or (
                    flag[row-1] == 1 and low[row] > nd_low[row-1]):
                flag[row] = 1

            if (low[row] <= nd_low[row-1]) or (
                    flag[row-1] == -1 and high[row] < nd_high[row-1]):
                flag[row] = -1

        return nd_low, nd_high, flag


    @staticmethod
    def trend_channel(close, high, low):
        """
        Calculate Trend lines above and below the price data to form a channel

        Parameters
        ----------
        close : Series
            Time series of closing prices.
        high : Series
            Time series of high prices.
        low : Series
            Time series of low prices.

        Returns
        -------
        low_trend : Series.
            Lower trend line
        high_trend : Series.
            High trend line

        """
        prices = pd.concat([close, high, low], axis=1)

        output = copy.deepcopy(prices)

        output['counter'] = (
            (output.index.date
             - output.index.date.min())).astype('timedelta64[D]')
        output['counter'] = output['counter'].dt.days + 1

        # high trend line
        tmp_data = copy.deepcopy(output)

        while len(tmp_data)>3:

            reg_line = linregress(x=tmp_data['counter'], y=tmp_data['High'])
            tmp_data = tmp_data.loc[tmp_data['High'] > reg_line[0]
                                    * tmp_data['counter'] + reg_line[1]]


        reg_line = linregress(x=tmp_data['counter'], y=tmp_data['High'])

        high_trend = reg_line[0] * output['counter'] + reg_line[1]

        # low trend line
        tmp_data = output.copy()

        while len(tmp_data)>3:

            reg_line = linregress(x=tmp_data['counter'], y=tmp_data['Low'])
            tmp_data = tmp_data.loc[tmp_data['Low'] < reg_line[0]
                                    * tmp_data['counter'] + reg_line[1]]

        reg_line = linregress(x=tmp_data['counter'], y=tmp_data['Low'])

        low_trend = reg_line[0] * output['counter'] + reg_line[1]

        return low_trend, high_trend


    @staticmethod
    def true_range(high, low, close):
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
        #pd.options.mode.chained_assignment = None

        # Calculate the high minus low, absolute value of high minus yesterdays
        # close and the absolute value of the low minus yesterdays close
        high_low = high - low
        high_close = np.abs(high - close.shift())
        close_low = np.abs(low - close.shift())

        # Concatenate these 3 series
        ranges = pd.concat([high_low, high_close, close_low], axis=1)

        # Calculate the True Range as the maximum of these 3 series
        t_range = ranges.max(skipna=False, axis=1)

        # Suppress SettingWithCopyWarning caused by slicing DataFrame
        #pd.options.mode.chained_assignment = "warn"

        return t_range


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
        dataframe = pd.concat(fields, axis=1)

        return dataframe


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
