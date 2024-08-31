import logging
from trading_system import ExchangeClient, TradingStrategies, TradingBot

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s :: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('info.log')
logger.addHandler(file_handler)

if __name__ == '__main__':
    cliente = ExchangeClient(True,symbol='BTCUSDT')
    
