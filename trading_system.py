
# Importación de bibliotecas necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para cálculos numéricos
import requests  # Para realizar solicitudes HTTP
import time  # Para operaciones relacionadas con el tiempo
import hmac  # Para generar firmas HMAC
import hashlib  # Para funciones hash criptográficas
import logging  # Para registrar eventos y errores
from urllib.parse import urlencode  # Para codificar parámetros de URL
import websocket  # Para conexiones WebSocket
import threading  # Para manejar hilos
import json  # Para trabajar con datos JSON
from datetime import datetime  # Para operaciones con fechas y horas
from config.api_keys import API_KEY, SECRET_KEY  # Importar claves API (asegúrate de tener este archivo)

# Configuración del logger
logger = logging.getLogger()

# Clase para representar el balance de una cuenta
class Balance:
    def __init__(self, info):
        self.initial_margin = float(info['initialMargin'])
        self.maintenance_margin = float(info['maintenanceMargin'])
        self.margin_balance = float(info['marginBalance'])
        self.wallet_balance = float(info['walletBalance'])
        self.unrealized_pnl = float(info['unrealizedPnl'])

# Clase para representar una vela (candle) en un gráfico de trading
class Candle:
    def __init__(self, candle_info):
        self.timestamp = candle_info[0]
        self.open = float(candle_info[1])
        self.high = float(candle_info[2])
        self.low = float(candle_info[3])
        self.close = float(candle_info[4])  
        self.volume = float(candle_info[5])

# Clase para representar un contrato de trading
class Contract:
    def __init__(self, contract_info):
        self.symbol = contract_info['symbol']
        self.base_asset = contract_info['baseAsset']
        self.quote_asset = contract_info['quoteAsset']
        self.price_decimals = contract_info['pricePrecision']
        self.quantity_decimals = contract_info['quantityPrecision']

# Clase para representar el estado de una orden
class OrderStatus:
    def __init__(self, order_info):
        self.order_id = order_info['orderId']
        self.status = order_info['status']
        self.avg_price = float(order_info['avgPrice'])

# Clase principal para interactuar con el exchange (Binance en este caso)
class ExchangeClient:
    def __init__(self, testnet=True, api_key=None, api_secret=None, symbol='BTCUSDT'):
        # Inicialización de atributos
        self._position_id = None
        self._symbol = symbol
        self._prices = dict()
        self.testnet = testnet
        self.api_key = API_KEY
        self.api_secret = SECRET_KEY
        self.headers = {"X-MBX-APIKEY": self.api_key} 
        self.contracts = self.get_contracts()
        self.balances = self.get_balances()
        self.id = 1
        self.ws = None

        # Configuración de URLs según el modo (testnet o producción)
        if self.testnet:
            self.base_url = 'https://testnet.binancefuture.com'
            self.wss_url = 'wss://stream.binancefuture.com/ws'
        else:
            self.base_url = 'https://fapi.binance.com'
            self.wss_url = 'wss://fstream.binancefuture.com/ws'
        
    # Métodos getter y setter para los atributos de la clase
    def get_position_id(self):
        return self._position_id

    # Getters and Setters for symbol
    def get_symbol(self):
        return self._symbol

    # Getters and Setters for position_type
    def get_position_type(self):
        return self._position_type

    # Getters and Setters for size
    def get_size(self):
        return self._size
    
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

    # Getters and Setters for current_price
    def get_current_price(self):
        return self._current_price

    # Getters for pnl (no setter as it's calculated)
    def get_pnl(self):
        return self._pnl

    # Getters and Setters for status
    def get_status(self):
        return self._status

    # Helper method to update PNL
    def _update_pnl(self):
        if self._position_type == 'long':
            self._pnl = (self._current_price - self._entry_price) * self._siz

    # Método para generar una firma para solicitudes autenticadas
    def _generate_signature(self, data):
        return hmac.new(self.api_secret.encode('utf-8'), urlencode(data).encode('utf-8'), hashlib.sha256).hexdigest()

    # Método para enviar solicitudes HTTP a la API de Binance
    def _send_request(self, method, endpoint, params=None, signed=False):
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

    # Método para obtener datos históricos de velas
    def get_historical_candles(self, interval="1h"):
        data = dict()
        data['symbol'] = self._symbol
        data['interval'] = interval
        data['limit'] = 1000
        raw_candles = self._send_request('GET', '/fapi/v1/klines', data)
        candles = []
        if raw_candles is not None:
            for c in raw_candles:
                candles.append(Candle(c))
        return candles

    # Método para obtener información sobre los contratos disponibles
    def get_contracts(self):
        data = dict()
        data['symbol'] = self._symbol
        exchange_info = self._send_request('GET', '/fapi/v1/exchangeInfo', data)
        contracts = dict()
        if exchange_info is not None:
            for contract_data in exchange_info['symbols']:
                contracts[contract_data['pair']] = Contract(contract_data)
        return contracts
    
    # Método para obtener los precios de oferta y demanda actuales
    def get_bid_ask(self):
        data = dict()
        data['symbol'] = self._symbol
        ob_data = self._send_request('GET', '/fapi/v1/ticker/bookTicker', data)
        if ob_data is not None:
            if self._symbol not in self._prices:
                self._prices[self._symbol] = {'bid': float(ob_data['bidPrice']), 'ask': float(ob_data['askPrice'])}
            else:
                self._prices[self._symbol]['bid'] = float(ob_data['bidPrice'])
                self._prices[self._symbol]['ask'] = float(ob_data['askPrice'])
        return self._prices[self._symbol]
    
    # Método para obtener los balances de la cuenta
    def get_balances(self):
        data = dict()
        data['symbol'] = self._symbol
        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)
        balances = dict()
        account_data = self._send_request('GET', '/fapi/v1/account', data)    
        if account_data is not None:
            for a in account_data['assets']:
                balances[a['asset']] = Balance(a)
        return balances

    # Métodos adicionales para obtener información del mercado
    def get_ticker_24hr(self):
        params = {}
        if self._symbol:
            params['symbol'] = self._symbol
        return self._send_request('GET', '/fapi/v1/ticker/24hr', params)

    def get_ticker_price(self):
        params = {}
        if self._symbol:
            params['symbol'] = self._symbol
        return self._send_request('GET', '/fapi/v1/ticker/price', params)

    def get_open_interest(self):
        params = {'symbol': self._symbol}
        return self._send_request('GET', '/fapi/v1/openInterest', params)

    def get_open_interest_hist(self, period, **kwargs):
        params = {'symbol': self._symbol, 'period': period, **kwargs}
        return self._send_request('GET', '/futures/data/openInterestHist', params)
    
    def get_account_balance(self):
        return self._send_request('GET', '/fapi/v2/balance', signed=True)

    def get_position_risk(self):
        params = {}
        if self._symbol:
            params['symbol'] = self._symbol
        return self._send_request('GET', '/fapi/v2/positionRisk', params, signed=True)
    
    def get_open_orders(self):
        params = {}
        if self._symbol:
            params['symbol'] = self._symbol
        return self._send_request('GET', '/fapi/v1/openOrders', params, signed=True)

    # Método para colocar una orden
    def place_order(self, side, quantity, order_type, price=None, time_in_force=None):
        data = {
            'symbol': self._symbol,
            'side': side,
            'quantity': quantity,
            'type': order_type,
            'timeInForce': time_in_force
        }
        if price:
            data['price'] = price
        if time_in_force:
            data['timeInForce'] = time_in_force
        order_status = self._send_request('POST', '/fapi/v1/order', data, signed=True)
        if order_status:
            order_status = OrderStatus(order_status)
        return order_status
    
    # Método para cancelar una orden
    def cancel_order(self, order_id):
        data = dict()
        data['orderId'] = order_id
        data['symbol'] = self._symbol
        order_status = self._send_request('DELETE', '/fapi/v1/order', data, signed=True)
        if order_status:
            order_status = OrderStatus(order_status)
        return order_status

# Clase para implementar estrategias de trading
class TradingStrategies:
    def __init__(self, client: ExchangeClient):
        self._client = client

    # Método para obtener datos históricos y convertirlos en un DataFrame
    def get_historical_data(self, interval='1h', limit=100):
        klines = self._client.get_historical_candles()
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    # Estrategia de cruce de medias móviles simples
    def simple_moving_average_crossover(self, symbol, interval, short_window=10, long_window=50):
        df = self.get_historical_data(interval)
        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()

        df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
        df['position'] = df['signal'].diff()

        return df['position'].iloc[-1]

    # Estrategia del Índice de Fuerza Relativa (RSI)
    def relative_strength_index(self, symbol, interval, period=14, oversold=30, overbought=70):
        df = self.get_historical_data(interval)
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

    # Estrategia de Bandas de Bollinger
    def bollinger_bands(self, symbol, interval, period=20, num_std=2):
        df = self.get_historical_data(interval)
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

    # Estrategia MACD (Moving Average Convergence Divergence)
    def macd(self, symbol, interval, fast_period=12, slow_period=26, signal_period=9):
        df = self.get_historical_data(interval)
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

    # Estrategia de Retroceso de Fibonacci
    def fibonacci_retracement(self, symbol, interval, period=100):
        df = self.get_historical_data(interval)
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

    # Método para implementar una estrategia específica
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

    # Método para realizar backtesting de una estrategia
    def backtest_strategy(self, symbol, interval, strategy='sma_crossover', start_date=None, end_date=None, initial_balance=10000, **kwargs):
        df = self.get_historical_data(interval)
        
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

# Clase principal para el bot de trading
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

    # Método para iniciar la conexión WebSocket
    def start_ws(self):
        self.ws = websocket.WebSocketApp(self.client.wss_url, on_open=self.on_open, on_close=self.on_close, on_error=self.on_error, on_message=self.on_message)
        self.ws.run_forever()

    # Métodos de callback para el WebSocket
    def on_open(self, ws):
        logger.info('Binance Connection opened')
        self.subscribe_channel()

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
                if symbol not in self.client._prices:
                    self.client._prices[symbol] = {'bid': float(data['b']), 'ask': float(data['a'])}
                else:
                    self.client._prices[symbol]['bid'] = float(data['b'])
                    self.client._prices[symbol]['ask'] = float(data['a'])
                print(self.client._prices[symbol])

    # Método para suscribirse a un canal WebSocket
    def subscribe_channel(self):
        data = dict()
        data['method'] = 'SUBSCRIBE'
        data['params'] = []
        data['params'].append(self._symbol.lower() + '@trade')
        data['id'] = self._ws_id
        try:
            self.ws.send(json.dumps(data))
        except Exception as e:
            logger.error('Websocket error while subscribing to %s: %s', self._symbol.lower(), e)
        self._ws_id += 1

    # Método para ejecutar una estrategia de trading
    def run_strategy(self, symbol: str, interval: str, strategy: str, quantity: float, **kwargs):
        df = self.strategies.implement_strategy(symbol, interval, strategy, **kwargs)
        
        if df is None:
            return

        last_signal = df['signal'].iloc[-1]
        
        if last_signal == 1:
            logger.info(f"Buying {quantity} of {symbol}")
            order = self.client.place_order(symbol, 'BUY', 'MARKET', quantity)
            logger.info(f"Buy order placed: {order}")
        elif last_signal == -1:
            logger.info(f"Selling {quantity} of {symbol}")
            order = self.client.place_order(symbol, 'SELL', 'MARKET', quantity)
            logger.info(f"Sell order placed: {order}")
        else:
            logger.info("No trading signal")

    # Método para realizar backtesting
    def backtest(self, symbol: str, interval: str, strategy: str, start_date: str, end_date: str, **kwargs):
        return self.strategies.backtest_strategy(symbol, interval, strategy, start_date, end_date, **kwargs)

    # Método para obtener información sobre el riesgo de la posición
    def get_position_risk(self, symbol: str):
        risk = self.client.get_position_risk(symbol)
        if risk:
            logger.info(f"Position risk for {symbol}: {risk}")
        else:
            logger.error(f"Failed to fetch position risk for {symbol}")
        return risk
