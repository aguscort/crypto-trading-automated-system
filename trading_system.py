import pandas as pd
import numpy as np
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode

class BinanceFuturesClient:
    def __init__(self, testnet=True, api_key=None, api_secret=None):
        """
        Initialize the Binance Futures Client.
        
        :param testnet: Boolean, use testnet if True, real network if False
        :param api_key: String, your Binance API key
        :param api_secret: String, your Binance API secret
        """
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        
        if self.testnet:
            self.base_url = 'https://testnet.binancefuture.com'
        else:
            self.base_url = 'https://fapi.binance.com'

    def _generate_signature(self, data):
        """Generate signature for authenticated requests."""
        return hmac.new(self.api_secret.encode('utf-8'), urlencode(data).encode('utf-8'), hashlib.sha256).hexdigest()

    def _send_request(self, method, endpoint, params=None, signed=False):
        """Send HTTP request to Binance Futures API."""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

        if signed:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)

        try:
            response = requests.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return None

    def get_exchange_info(self):
        """Get exchange information."""
        return self._send_request('GET', '/fapi/v1/exchangeInfo')

    def get_order_book(self, symbol, limit=100):
        params = {'symbol': symbol, 'limit': limit}
        return self._send_request('GET', '/fapi/v1/depth', params)

    def get_recent_trades(self, symbol, limit=500):
        params = {'symbol': symbol, 'limit': limit}
        return self._send_request('GET', '/fapi/v1/trades', params)

    def get_aggregate_trades(self, symbol, **kwargs):
        params = {'symbol': symbol, **kwargs}
        return self._send_request('GET', '/fapi/v1/aggTrades', params)
    
    def get_continuous_klines(self, pair, contractType, interval, **kwargs):
        params = {'pair': pair, 'contractType': contractType, 'interval': interval, **kwargs}
        return self._send_request('GET', '/fapi/v1/continuousKlines', params)

    def get_index_price_klines(self, pair, interval, **kwargs):
        params = {'pair': pair, 'interval': interval, **kwargs}
        return self._send_request('GET', '/fapi/v1/indexPriceKlines', params)

    def get_mark_price_klines(self, symbol, interval, **kwargs):
        params = {'symbol': symbol, 'interval': interval, **kwargs}
        return self._send_request('GET', '/fapi/v1/markPriceKlines', params)

    def get_mark_price(self, symbol=None):
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/premiumIndex', params)

    def get_funding_rate(self, symbol=None, **kwargs):
        params = {**kwargs}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/fundingRate', params)

    def get_ticker_24hr(self, symbol=None):
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/ticker/24hr', params)

    def get_ticker_price(self, symbol=None):
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/ticker/price', params)

    def get_ticker_book_ticker(self, symbol=None):
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/ticker/bookTicker', params)

    def get_open_interest(self, symbol):
        params = {'symbol': symbol}
        return self._send_request('GET', '/fapi/v1/openInterest', params)

    def get_open_interest_hist(self, symbol, period, **kwargs):
        params = {'symbol': symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/openInterestHist', params)

    def get_top_long_short_account_ratio(self, symbol, period, **kwargs):
        params = {'symbol': symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/topLongShortAccountRatio', params)

    def get_top_long_short_position_ratio(self, symbol, period, **kwargs):
        params = {'symbol': symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/topLongShortPositionRatio', params)

    def get_global_long_short_account_ratio(self, symbol, period, **kwargs):
        params = {'symbol': symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/globalLongShortAccountRatio', params)

    def get_taker_long_short_ratio(self, symbol, period, **kwargs):
        params = {'symbol': symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/takerlongshortRatio', params)

    def get_klines(self, symbol, interval, limit=500):
        """
        Get kline/candlestick data.
        
        :param symbol: String, the trading pair
        :param interval: String, the interval of kline, e.g., '1m', '5m', '1h', '1d'
        :param limit: Integer, the number of klines to retrieve (max 1000)
        """
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        return self._send_request('GET', '/fapi/v1/klines', params)

    def create_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC"):
        """
        Create a new order.
        
        :param symbol: String, the trading pair
        :param side: String, 'BUY' or 'SELL'
        :param order_type: String, 'LIMIT', 'MARKET', 'STOP', 'TAKE_PROFIT', etc.
        :param quantity: Float, the amount to buy or sell
        :param price: Float, the price for limit orders (optional)
        :param time_in_force: String, 'GTC', 'IOC', 'FOK' (optional, default 'GTC')
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timeInForce': time_in_force
        }
        if price:
            params['price'] = price

        return self._send_request('POST', '/fapi/v1/order', params, signed=True)

    def get_open_orders(self, symbol=None):
        """
        Get all open orders on a symbol.
        
        :param symbol: String, the trading pair (optional)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/openOrders', params, signed=True)

    def cancel_order(self, symbol, order_id=None, orig_client_order_id=None):
        """
        Cancel an active order.
        
        :param symbol: String, the trading pair
        :param order_id: Integer, the order id to cancel (optional)
        :param orig_client_order_id: String, the client order id to cancel (optional)
        """
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")

        return self._send_request('DELETE', '/fapi/v1/order', params, signed=True)

    def get_account_balance(self):
        """Get current account balance."""
        return self._send_request('GET', '/fapi/v2/balance', signed=True)

    def get_position_risk(self, symbol=None):
        """
        Get position risk.
        
        :param symbol: String, the trading pair (optional)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v2/positionRisk', params, signed=True)

class TradingStrategies:
    def __init__(self, client: BinanceFuturesClient):
        """
        Initialize TradingStrategies with a BinanceFuturesClient.
        
        :param client: BinanceFuturesClient instance
        """
        self.client = client

    def get_historical_data(self, symbol, interval, limit=500):
        """
        Fetch historical kline data and convert to DataFrame.
        
        :param symbol: String, the trading pair
        :param interval: String, the interval of kline, e.g., '1m', '5m', '1h', '1d'
        :param limit: Integer, the number of klines to retrieve (max 1000)
        """
        klines = self.client.get_klines(symbol, interval, limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def sma_crossover(self, df, short_window=10, long_window=50):
        """
        Simple Moving Average Crossover strategy.
        
        :param df: DataFrame with historical price data
        :param short_window: Integer, the short-term moving average window
        :param long_window: Integer, the long-term moving average window
        """
        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()
        df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
        df['position'] = df['signal'].diff()
        return df

    def rsi(self, df, period=14, oversold=30, overbought=70):
        """
        Relative Strength Index (RSI) strategy.
        
        :param df: DataFrame with historical price data
        :param period: Integer, the RSI period
        :param oversold: Integer, the oversold level
        :param overbought: Integer, the overbought level
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['signal'] = np.where(df['rsi'] < oversold, 1, np.where(df['rsi'] > overbought, -1, 0))
        return df

    # Add more trading strategies as needed

    def backtest(self, df, strategy_name, **strategy_params):
        """
        Perform backtesting on a given strategy.
        
        :param df: DataFrame with historical price data
        :param strategy_name: String, name of the strategy to backtest
        :param strategy_params: Additional parameters for the strategy
        """
        if strategy_name == 'sma_crossover':
            df = self.sma_crossover(df, **strategy_params)
        elif strategy_name == 'rsi':
            df = self.rsi(df, **strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()

        total_return = (df['cumulative_strategy_returns'].iloc[-1] - 1) * 100
        sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
        max_drawdown = (df['cumulative_strategy_returns'] / df['cumulative_strategy_returns'].cummax() - 1).min()

        print(f"Backtesting results for {strategy_name}:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")

        return df

