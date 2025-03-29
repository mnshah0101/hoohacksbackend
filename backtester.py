from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import math

# A simple RSI function used as an indicator


def compute_rsi(prices, period):
    prices = pd.Series(prices)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# A helper function to compute ADX along with +DI and -DI


def compute_adx(high, low, close, period=14):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Directional Movements
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Directional Indicators
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

    # DX and ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di

# Base class that adds risk management functionality


class RiskManagedStrategy(Strategy):
    # These will be set dynamically from money_management_params
    risk_fraction = 0.02         # default: risk 2% of equity per trade
    max_position_size = None     # default: no cap

    def calc_position_size(self):
        # If no stop loss is defined, return a default size of 1 unit
        if self.stop_loss is None:
            return 1
        
        equity = self.equity
        current_price = self.data.Close[-1]
        # Convert stop_loss from percentage to price
        stop_loss_price = current_price * (1 - self.stop_loss)
        risk_per_share = abs(current_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 1
            
        pos_size = (equity * self.risk_fraction) / risk_per_share
        if self.max_position_size is not None:
            pos_size = min(pos_size, self.max_position_size)
        return math.ceil(pos_size)

    def calculate_sl_tp(self, price):
        """Calculate stop-loss and take-profit prices based on percentages"""
        if self.stop_loss is not None:
            # Convert percentage to price difference below entry
            sl = price * (1 - self.stop_loss)
        else:
            sl = None
            
        if self.take_profit is not None:
            # Convert percentage to price difference above entry
            tp = price * (1 + self.take_profit)
        else:
            tp = None
            
        # Validate that SL < Entry < TP
        if sl is not None and tp is not None:
            if not (sl < price < tp):
                # If validation fails, disable SL and TP
                sl, tp = None, None
        
        return sl, tp


class BacktestRunner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def create_strategy_class(self, strategy_type: str, strategy_params: dict, money_management_params: dict):
        # Extract common money management parameters
        risk_fraction = money_management_params.get('risk_fraction', 0.02)
        max_position_size = money_management_params.get(
            'max_position_size', None)

        if strategy_type == 'ma_crossover':
            class MACrossoverStrategy(RiskManagedStrategy):
                short_window = strategy_params.get('short_window', 10)
                long_window = strategy_params.get('long_window', 20)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.short_sma = self.I(lambda x: pd.Series(x).rolling(
                        self.short_window).mean(), self.data.Close)
                    self.long_sma = self.I(lambda x: pd.Series(x).rolling(
                        self.long_window).mean(), self.data.Close)

                def next(self):
                    if (self.short_sma[-1] > self.long_sma[-1] and
                            self.short_sma[-2] <= self.long_sma[-2]):
                        price = self.data.Close[-1]
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif (self.short_sma[-1] < self.long_sma[-1] and
                          self.short_sma[-2] >= self.long_sma[-2]):
                        if self.position:
                            self.position.close()
            MACrossoverStrategy.risk_fraction = risk_fraction
            MACrossoverStrategy.max_position_size = max_position_size
            return MACrossoverStrategy

        elif strategy_type == 'rsi':
            class RSIStrategy(RiskManagedStrategy):
                rsi_period = strategy_params.get('rsi_period', 14)
                overbought = strategy_params.get('overbought', 70)
                oversold = strategy_params.get('oversold', 30)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.rsi = self.I(
                        compute_rsi, self.data.Close, self.rsi_period)

                def next(self):
                    if self.rsi[-1] < self.oversold:
                        price = self.data.Close[-1]
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif self.rsi[-1] > self.overbought:
                        if self.position:
                            self.position.close()
            RSIStrategy.risk_fraction = risk_fraction
            RSIStrategy.max_position_size = max_position_size
            return RSIStrategy

        elif strategy_type == 'bollinger':
            class BollingerReversionStrategy(RiskManagedStrategy):
                period = strategy_params.get('period', 20)
                std_multiplier = strategy_params.get('std_multiplier', 2)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.middle_band = self.I(lambda x: pd.Series(
                        x).rolling(self.period).mean(), self.data.Close)
                    self.std = self.I(lambda x: pd.Series(x).rolling(
                        self.period).std(), self.data.Close)
                    self.upper_band = self.I(
                        lambda ma, std: ma + self.std_multiplier * std, self.middle_band, self.std)
                    self.lower_band = self.I(
                        lambda ma, std: ma - self.std_multiplier * std, self.middle_band, self.std)

                def next(self):
                    price = self.data.Close[-1]
                    if price < self.lower_band[-1]:
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif price > self.upper_band[-1]:
                        if self.position:
                            self.position.close()
            BollingerReversionStrategy.risk_fraction = risk_fraction
            BollingerReversionStrategy.max_position_size = max_position_size
            return BollingerReversionStrategy

        elif strategy_type == 'mean_reversion_simple':
            class MeanReversionSimpleStrategy(RiskManagedStrategy):
                period = strategy_params.get('period', 20)
                threshold = strategy_params.get('threshold', 0.05)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.ma = self.I(lambda x: pd.Series(x).rolling(
                        self.period).mean(), self.data.Close)

                def next(self):
                    price = self.data.Close[-1]
                    ma_value = self.ma[-1]
                    if price < ma_value * (1 - self.threshold):
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif price > ma_value * (1 + self.threshold):
                        if self.position:
                            self.position.close()
            MeanReversionSimpleStrategy.risk_fraction = risk_fraction
            MeanReversionSimpleStrategy.max_position_size = max_position_size
            return MeanReversionSimpleStrategy

        elif strategy_type == 'mean_reversion_zscore':
            class MeanReversionZScoreStrategy(RiskManagedStrategy):
                period = strategy_params.get('period', 20)
                zscore_threshold = strategy_params.get('zscore_threshold', 1.0)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.ma = self.I(lambda x: pd.Series(x).rolling(
                        self.period).mean(), self.data.Close)
                    self.std = self.I(lambda x: pd.Series(x).rolling(
                        self.period).std(), self.data.Close)

                def next(self):
                    price = self.data.Close[-1]
                    if self.std[-1] == 0:
                        zscore = 0
                    else:
                        zscore = (price - self.ma[-1]) / self.std[-1]
                    if zscore < -self.zscore_threshold:
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif zscore > self.zscore_threshold:
                        if self.position:
                            self.position.close()
            MeanReversionZScoreStrategy.risk_fraction = risk_fraction
            MeanReversionZScoreStrategy.max_position_size = max_position_size
            return MeanReversionZScoreStrategy

        elif strategy_type == 'macd':
            class MACDStrategy(RiskManagedStrategy):
                fast_period = strategy_params.get('fast_period', 12)
                slow_period = strategy_params.get('slow_period', 26)
                signal_period = strategy_params.get('signal_period', 9)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.ema_fast = self.I(lambda x: pd.Series(x).ewm(
                        span=self.fast_period, adjust=False).mean(), self.data.Close)
                    self.ema_slow = self.I(lambda x: pd.Series(x).ewm(
                        span=self.slow_period, adjust=False).mean(), self.data.Close)
                    self.macd_line = self.ema_fast - self.ema_slow
                    self.signal_line = self.I(lambda x: pd.Series(x).ewm(
                        span=self.signal_period, adjust=False).mean(), self.macd_line)

                def next(self):
                    if (self.macd_line[-1] > self.signal_line[-1] and
                            self.macd_line[-2] <= self.signal_line[-2]):
                        price = self.data.Close[-1]
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif (self.macd_line[-1] < self.signal_line[-1] and
                          self.macd_line[-2] >= self.signal_line[-2]):
                        if self.position:
                            self.position.close()
            MACDStrategy.risk_fraction = risk_fraction
            MACDStrategy.max_position_size = max_position_size
            return MACDStrategy

        elif strategy_type == 'trend_following_adx':
            class TrendFollowingADXStrategy(RiskManagedStrategy):
                adx_period = strategy_params.get('adx_period', 14)
                adx_threshold = strategy_params.get('adx_threshold', 25)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    self.adx = self.I(lambda high, low, close: compute_adx(high, low, close, self.adx_period)[0],
                                      self.data.High, self.data.Low, self.data.Close)
                    self.di_plus = self.I(lambda high, low, close: compute_adx(high, low, close, self.adx_period)[1],
                                          self.data.High, self.data.Low, self.data.Close)
                    self.di_minus = self.I(lambda high, low, close: compute_adx(high, low, close, self.adx_period)[2],
                                           self.data.High, self.data.Low, self.data.Close)

                def next(self):
                    if self.adx[-1] > self.adx_threshold:
                        if self.di_plus[-1] > self.di_minus[-1]:
                            if not self.position:
                                price = self.data.Close[-1]
                                sl, tp = self.calculate_sl_tp(price)
                                self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                        elif self.di_plus[-1] < self.di_minus[-1]:
                            if self.position:
                                self.position.close()
            TrendFollowingADXStrategy.risk_fraction = risk_fraction
            TrendFollowingADXStrategy.max_position_size = max_position_size
            return TrendFollowingADXStrategy

        # ----- New Strategies -----
        elif strategy_type == 'breakout':
            class BreakoutStrategy(RiskManagedStrategy):
                breakout_window = strategy_params.get('breakout_window', 20)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    # Use previous window's highest high as breakout level
                    self.breakout_high = self.I(lambda x: pd.Series(x).rolling(
                        self.breakout_window).max().shift(1), self.data.High)

                def next(self):
                    price = self.data.Close[-1]
                    if not self.position and price > self.breakout_high[-1]:
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif self.position and price < self.breakout_high[-1]:
                        self.position.close()
            BreakoutStrategy.risk_fraction = risk_fraction
            BreakoutStrategy.max_position_size = max_position_size
            return BreakoutStrategy

        elif strategy_type == 'momentum':
            class MomentumStrategy(RiskManagedStrategy):
                momentum_period = strategy_params.get('momentum_period', 10)
                momentum_threshold = strategy_params.get(
                    'momentum_threshold', 0.01)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    # Calculate rate of change (momentum)
                    self.momentum = self.I(lambda x: (
                        pd.Series(x)/pd.Series(x).shift(self.momentum_period) - 1), self.data.Close)

                def next(self):
                    # Enter if momentum exceeds the threshold
                    if not self.position and self.momentum[-1] > self.momentum_threshold:
                        price = self.data.Close[-1]
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    # Exit if momentum falls below 0
                    elif self.position and self.momentum[-1] < 0:
                        self.position.close()
            MomentumStrategy.risk_fraction = risk_fraction
            MomentumStrategy.max_position_size = max_position_size
            return MomentumStrategy

        elif strategy_type == 'vwap':
            class VWAPStrategy(RiskManagedStrategy):
                vwap_period = strategy_params.get('vwap_period', 20)
                stop_loss = money_management_params.get('stop_loss', None)
                take_profit = money_management_params.get('take_profit', None)

                def init(self):
                    # Calculate rolling VWAP over the vwap_period; data must contain a 'Volume' column
                    self.vwap = self.I(lambda p, v: (pd.Series(p * v).rolling(self.vwap_period).sum() /
                                                     pd.Series(v).rolling(self.vwap_period).sum()),
                                       self.data.Close, self.data.Volume)

                def next(self):
                    # Buy when price crosses above VWAP; exit when it crosses below
                    if (not self.position and
                        self.data.Close[-1] > self.vwap[-1] and
                            self.data.Close[-2] <= self.vwap[-2]):
                        price = self.data.Close[-1]
                        sl, tp = self.calculate_sl_tp(price)
                        self.buy(size=self.calc_position_size(), sl=sl, tp=tp)
                    elif (self.position and
                          self.data.Close[-1] < self.vwap[-1] and
                          self.data.Close[-2] >= self.vwap[-2]):
                        self.position.close()
            VWAPStrategy.risk_fraction = risk_fraction
            VWAPStrategy.max_position_size = max_position_size
            return VWAPStrategy

        else:
            raise ValueError("Unsupported strategy type provided.")

    def run_backtest(self, strategy_type: str, strategy_params: dict, money_management_params: dict):
        StrategyClass = self.create_strategy_class(
            strategy_type, strategy_params, money_management_params)
        bt = Backtest(self.data, StrategyClass, cash=10000, commission=0.002)
        stats = bt.run()
        return stats

""" 
# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200)
    prices = np.cumsum(np.random.randn(200)) + 100
    # For demonstration, we include a Volume column for VWAP strategy.
    volume = np.random.randint(100, 1000, size=200)
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices,
        'Low': prices,
        'High': prices,
        'Volume': volume
    }).set_index('Date')

    runner = BacktestRunner(data)

    # Define common parameters for strategies
    strategy_params = {
        'short_window': 5,            # for ma_crossover
        'long_window': 20,            # for ma_crossover
        'rsi_period': 14,             # for rsi
        'period': 20,                 # for bollinger and mean reversion strategies
        'std_multiplier': 2,          # for bollinger
        'threshold': 0.05,            # for mean_reversion_simple
        'zscore_threshold': 1.0,      # for mean_reversion_zscore
        'fast_period': 12,            # for macd
        'slow_period': 26,            # for macd
        'signal_period': 9,           # for macd
        'adx_period': 14,             # for trend_following_adx
        'adx_threshold': 25,          # for trend_following_adx
        'breakout_window': 20,        # for breakout strategy
        'momentum_period': 10,        # for momentum strategy
        'momentum_threshold': 0.01,   # for momentum strategy
        'vwap_period': 20             # for VWAP strategy
    }
    money_management_params = {
        'stop_loss': 50,
        'take_profit': 110,
        'risk_fraction': 0.02,        # risk 2% of equity per trade
        'max_position_size': 10       # maximum position size
    }

    # Change the strategy type to test any one of the following:
    # 'ma_crossover', 'rsi', 'bollinger', 'mean_reversion_simple',
    # 'mean_reversion_zscore', 'macd', 'trend_following_adx', 'breakout',
    # 'momentum', or 'vwap'
    results = runner.run_backtest(
        'breakout', strategy_params, money_management_params)
    print(results)
"""
