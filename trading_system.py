import pandas as pd
import numpy as np
import requests
import time
import hmac
import hashlib
import logging
from urllib.parse import urlencode
import websocket
import threading
import json
from datetime import datetime
from config.api_keys import API_KEY, SECRET_KEY

logger = logging.getLogger()

class ExchangeClient:
    def __init__(self, testnet=True, api_key=None, api_secret=None, symbol='BTCUSDT', position_type=None, size=None, entry_price=None, open_time=None):
        """
        Initialize a TradingPosition.
        
        :param position_id: String, unique identifier for the position
        :param symbol: String, the trading pair symbol
        :param position_type: String, 'long' or 'short'
        :param open_time: Datetime, time when the position was opened
        :param entry_price: Float, price at which the position was entered
        :param quantity: Float, size of the position
        :param order_type: String, type of the entry order (e.g., 'market', 'limit')
        :param entry_order_id: String, ID of the entry order
        """
        self._position_id = None
        self._symbol = symbol
        self._position_type = position_type
        self._size = size
        self._entry_price = entry_price
        self._open_time = open_time
        self._stop_loss = None
        self._take_profit = None
        self._current_price = entry_price
        self._pnl = 0
        self._status = 'open'
        self._prices = dict()
        self.testnet = testnet
        self.api_key = API_KEY
        self.api_secret = SECRET_KEY
        
        if self.testnet:
            self.base_url = 'https://testnet.binancefuture.com'
            self.wss_url = 'wss://stream.binancefuture.com/ws'
        else:
            self.base_url = 'https://fapi.binance.com'
            self.base_url = 'wss://fstream.binancefuture.com/ws'
        
        # Initialize other attributes
        self._nominal_value = None
        self._margin_used = None
        self._leverage = None
        self._stop_loss = None
        self._take_profit = None
        self._status = 'open'
        self._unrealized_pnl = 0.0
        self._pnl_percentage = 0.0
        self._close_time = None
        self._close_price = None
        self._realized_pnl = None
        self._close_order_id = None
        self._strategy = None
        self._entry_reason = None
        self._notes = None
        self._market_volatility = None
        self._market_volume = None
        self._commission = 0.0
        self._funding_fee = 0.0
        self._account_id = None
        self._trading_session_id = None
        self._highest_price = entry_price
        self._lowest_price = entry_price
        self._position_duration = None

    # Getters and setters for each attribute
    def get_position_id(self):
        return self._position_id

    # Getters and Setters for symbol
    def get_symbol(self):
        return self._symbol

    # Getters and Setters for position_type
    def get_position_type(self):
        return self._position_type

    def set_position_type(self, position_type):
        if position_type not in ['long', 'short']:
            raise ValueError("Position type must be 'long' or 'short'")
        self._position_type = position_type

    # Getters and Setters for size
    def get_size(self):
        return self._size

    def set_size(self, size):
        if size <= 0:
            raise ValueError("Size must be positive")
        self._size = size

    # Getters and Setters for entry_price
    def get_entry_price(self):
        return self._entry_price

    # Getters and Setters for open_time
    def get_open_time(self):
        return self._open_time

    # Getters and Setters for stop_loss
    def get_stop_loss(self):
        return self._stop_loss

    def set_stop_loss(self, stop_loss):
        if stop_loss is not None and stop_loss <= 0:
            raise ValueError("Stop loss must be positive or None")
        self._stop_loss = stop_loss

    # Getters and Setters for take_profit
    def get_take_profit(self):
        return self._take_profit

    def set_take_profit(self, take_profit):
        if take_profit is not None and take_profit <= 0:
            raise ValueError("Take profit must be positive or None")
        self._take_profit = take_profit

    # Getters and Setters for current_price
    def get_current_price(self):
        return self._current_price

    def set_current_price(self, current_price):
        if current_price <= 0:
            raise ValueError("Current price must be positive")
        self._current_price = current_price
        self._update_pnl()

    # Getters for pnl (no setter as it's calculated)
    def get_pnl(self):
        return self._pnl

    # Getters and Setters for status
    def get_status(self):
        return self._status

    def set_status(self, status):
        if status not in ['open', 'closed']:
            raise ValueError("Status must be 'open' or 'closed'")
        self._status = status

    # Helper method to update PNL
    def _update_pnl(self):
        if self._position_type == 'long':
            self._pnl = (self._current_price - self._entry_price) * self._size
        else:
            self._pnl = (self._entry_price - self._current_price) * self._size

    # Method to close the position
    def close_position(self, close_price, close_time):
        self.set_current_price(close_price)
        self.set_status('closed')
        return {
            'position_id': self._position_id,
            'symbol': self._symbol,
            'type': self._position_type,
            'size': self._size,
            'entry_price': self._entry_price,
            'open_time': self._open_time,
            'close_price': close_price,
            'close_time': close_time,
            'pnl': self._pnl
        }

    def _generate_signature(self, data):
        """Generate signature for authenticated requests."""
        return hmac.new(self.api_secret.encode('utf-8'), urlencode(data).encode('utf-8'), hashlib.sha256).hexdigest()

    def _send_request(self, method, endpoint, params=None, signed=False):
        """Send HTTP request to Binance API."""
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
            logger.error("Error while making %s request to %s: %s (error code %s)", method, endpoint, response.json(), response.status_code)
            return None

    def get_historical_candles(self, symbol, interval):
        """
        Get kline/candlestick data.
        
        :param symbol: String, the trading pair
        :param interval: String, the interval of kline, e.g., '1m', '5m', '1h', '1d'
        """
        data = dict ()
        data['symbol'] = symbol
        data['interval'] = interval
        data['limit'] = 1000
        raw_candles = self._send_request('GET', '/fapi/v1/klines', data)
        candles = []
        if raw_candles is not None:
            for c in raw_candles:
                candles.append([c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])])
        return candles

    def get_contracts(self):
        """Get exchange information."""
        exchange_info = self._send_request('GET', '/fapi/v1/exchangeInfo', None)
        contracts= dict()
        if exchange_info is not None:
            for contract_data  in exchange_info['symbols']:
                contracts[contract_data['pair']] = contract_data
        return contracts
    
    def get_bid_ask(self, symbol):
        data = dict()
        data['symbol'] = symbol
        ob_data = self._send_request('GET', '/fapi/v1/ticker/bookTicker', data)
        if ob_data is not None:
            if symbol not in self._prices:
                self._prices[symbol] = {'bid': float(ob_data['bidPrice']), 'ask': float(ob_data['askPrice'])}
            else:
                self._prices[symbol]['bid']: float(ob_data['bidPrice'])
                self._prices[symbol]['ask']: float(ob_data['askPrice'])
        return self._prices[symbol]
    
    def get_balances(self):
        data = dict()
        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)
        balances = dict()
        account_data = self._send_request ('GET', '/fapi/v1/account', data)    
        if account_data is not None:
            for a in account_data['assets']:
                balances[a['asset']] = a
        return balances


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

    def get_open_interest(self, symbol):
        params = {'symbol': symbol}
        return self._send_request('GET', '/fapi/v1/openInterest', params)

    def get_open_interest_hist(self, symbol, period, **kwargs):
        params = {'symbol': symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/openInterestHist', params)
    
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
    
    def get_open_orders(self, symbol=None):
        """
        Get all open orders on a symbol.
        
        :param symbol: String, the trading pair (optional)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._send_request('GET', '/fapi/v1/openOrders', params, signed=True)

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

class TradingStrategies:
    def __init__(self, client: ExchangeClient):
        """
        Initialize TradingStrategies with a ExchangeClient.
        
        :param client: ExchangeClient instance
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

    def simple_moving_average_crossover(self, symbol, interval, short_window=10, long_window=50):
        df = self.get_historical_klines(symbol, interval)
        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()

        df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
        df['position'] = df['signal'].diff()

        return df['position'].iloc[-1]

    def relative_strength_index(self, symbol, interval, period=14, oversold=30, overbought=70):
        df = self.get_historical_klines(symbol, interval)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        if df['rsi'].iloc[-1] < oversold:
            return 1  # Señal de compra
        elif df['rsi'].iloc[-1] > overbought:
            return -1  # Señal de venta
        else:
            return 0  # Sin señal

    def bollinger_bands(self, symbol, interval, period=20, num_std=2):
        df = self.get_historical_klines(symbol, interval)
        df['sma'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        df['upper_band'] = df['sma'] + (df['std'] * num_std)
        df['lower_band'] = df['sma'] - (df['std'] * num_std)

        if df['close'].iloc[-1] < df['lower_band'].iloc[-1]:
            return 1  # Señal de compra
        elif df['close'].iloc[-1] > df['upper_band'].iloc[-1]:
            return -1  # Señal de venta
        else:
            return 0  # Sin señal

    def macd(self, symbol, interval, fast_period=12, slow_period=26, signal_period=9):
        df = self.get_historical_klines(symbol, interval)
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal_line']

        if df['macd'].iloc[-1] > df['signal_line'].iloc[-1] and df['macd'].iloc[-2] <= df['signal_line'].iloc[-2]:
            return 1  # Señal de compra
        elif df['macd'].iloc[-1] < df['signal_line'].iloc[-1] and df['macd'].iloc[-2] >= df['signal_line'].iloc[-2]:
            return -1  # Señal de venta
        else:
            return 0  # Sin señal

    def fibonacci_retracement(self, symbol, interval, period=100):
        df = self.get_historical_klines(symbol, interval, limit=period)
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low

        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        fib_levels = [high - (diff * level) for level in levels]

        current_price = df['close'].iloc[-1]
        for i in range(len(fib_levels) - 1):
            if fib_levels[i] >= current_price >= fib_levels[i+1]:
                if i < 3:  # Si el precio está en los niveles inferiores de Fibonacci
                    return 1  # Señal de compra
                elif i > 3:  # Si el precio está en los niveles superiores de Fibonacci
                    return -1  # Señal de venta
        return 0  # Sin señal

    def implement_strategy(self, symbol, interval, strategy='sma_crossover', **kwargs):
        if strategy == 'sma_crossover':
            return self.simple_moving_average_crossover(symbol, interval, **kwargs)
        elif strategy == 'rsi':
            return self.relative_strength_index(symbol, interval, **kwargs)
        elif strategy == 'bollinger_bands':
            return self.bollinger_bands(symbol, interval, **kwargs)
        elif strategy == 'macd':
            return self.macd(symbol, interval, **kwargs)
        elif strategy == 'fibonacci':
            return self.fibonacci_retracement(symbol, interval, **kwargs)
        else:
            raise ValueError(f"Estrategia '{strategy}' no reconocida")

    def backtest_strategy(self, symbol, interval, strategy='sma_crossover', start_date=None, end_date=None, initial_balance=10000, **kwargs):
        df = self.get_historical_klines(symbol, interval)
        
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]

        df['signal'] = df.apply(lambda row: self.implement_strategy(symbol, interval, strategy, **kwargs), axis=1)
        df['position'] = df['signal'].shift(1)
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()

        total_return = (df['cumulative_strategy_returns'].iloc[-1] - 1) * 100
        sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
        max_drawdown = (df['cumulative_strategy_returns'] / df['cumulative_strategy_returns'].cummax() - 1).min()

        print(f"Resultados del backtesting para {symbol} usando la estrategia {strategy}:")
        print(f"Retorno total: {total_return:.2f}%")
        print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        print(f"Máximo drawdown: {max_drawdown:.2f}%")

        return df

class TradingBot:
    def __init__(self, exchange_client: ExchangeClient):
        self.client = exchange_client
        self.strategies = TradingStrategies(self.client)
        self._symbol = self.client._symbol
        t = threading.Thread(target=self.start_ws)
        t.start()
        logger.info('Binance Futures Client successfully initialized')
        self._ws_id = 1
        self.ws = None

    def start_ws(self):
        self.ws = websocket.WebSocketApp(self.client.wss_url, on_open=self.on_open, on_close=self.on_close, on_error=self.on_error, on_message=self.on_message)
        self.ws.run_forever()

    def on_open(self, ws):
        logger.info('Binance Connection opened')
        self.subscribe_channel(self._symbol)

    def on_close(self, ws):
        logger.warning('Binance Websocket Connection Closed')

    def on_error(self, ws, msg):
        logger.error('Binance Connection error: %s', msg)

    def on_message(self, ws, msg):
        print(msg)
        data = json.loads(msg)
        if 'e' in data:
            if data['e'] == 'bookTicker':
                symbol = data['s']
                if symbol not in client._prices:
                    client._prices[symbol] = {'bid': float(data['b']), 'ask': float(data['a'])}
                else:
                    client._prices[symbol]['bid']: float(data['b'])
                    client._prices[symbol]['ask']: float(data['a'])
                print(client.prices[symbol])
    

    def subscribe_channel(self, symbol):
        data = dict()
        data['method'] = 'SUBSCRIBE'
        data['params'] = []
        print(self._symbol.lower())
        data['params'].append(self._symbol.lower() + '@bookTicker')
        data['id'] = self._ws_id
        try:
            self.ws.send(json.dumps(data))
        except Exception as e:
            logger.error('Websocket error while subscribing to %s: %s', self._symbol.lower(), e)
        self._ws_id += 1


    def run_strategy(self, symbol: str, interval: str, strategy: str, quantity: float, **kwargs):
        df = self.strategies.implement_strategy(symbol, interval, strategy, **kwargs)
        
        if df is None:
            return

        last_signal = df['signal'].iloc[-1]
        
        if last_signal == 1:
            logger.info(f"Buying {quantity} of {symbol}")
            order = self.client.create_order(symbol, 'BUY', 'MARKET', quantity)
            logger.info(f"Buy order placed: {order}")
        elif last_signal == -1:
            logger.info(f"Selling {quantity} of {symbol}")
            order = self.client.create_order(symbol, 'SELL', 'MARKET', quantity)
            logger.info(f"Sell order placed: {order}")
        else:
            logger.info("No trading signal")

    def backtest(self, symbol: str, interval: str, strategy: str, start_date: str, end_date: str, **kwargs):
        return self.strategies.backtest_strategy(symbol, interval, strategy, start_date, end_date, **kwargs)

    # def get_account_balance(self):
    #     balance = self.client.get_account_balance()
    #     if balance:
    #         logger.info(f"Account balance: {balance}")
    #     else:
    #         logger.error("Failed to fetch account balance")
    #     return balance

    # def get_open_orders(self, symbol: str):
    #     orders = self.client.get_open_orders(symbol)
    #     if orders:
    #         logger.info(f"Open orders for {symbol}: {orders}")
    #     else:
    #         logger.error(f"Failed to fetch open orders for {symbol}")
    #     return orders

    # def cancel_order(self, symbol: str, order_id: str):
    #     result = self.client.cancel_order(symbol, order_id)
    #     if result:
    #         logger.info(f"Order cancelled: {result}")
    #     else:
    #         logger.error(f"Failed to cancel order {order_id} for {symbol}")
    #     return result

    def get_position_risk(self, symbol: str):
        risk = self.client.get_position_risk(symbol)
        if risk:
            logger.info(f"Position risk for {symbol}: {risk}")
        else:
            logger.error(f"Failed to fetch position risk for {symbol}")
        return risk