import logging
from config.api_keys import API_KEY, SECRET_KEY
import trading_system 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = 60
    strategy = "SMA"  # o "RSI
    testnet = True
    
    cliente = trading_system.ExchangeClient(True)
    print(cliente.get_recent_trades(symbol='DOGSUSDT'))