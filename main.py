import logging
import binance_strateigies

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

""" logger.debug('This message is important only when debugging the program')
logger.info('This message just shows basic information')
logger.warning('This message is about somnething you should pay attention to')
logger.error('This message helps to debug an error that ocurred in the program') """

if __name__ == '__main__':
    logger.info('This is logged only if we execut the main.py file')
    bncftrs = binance_futures.BinanceFuturesClient(testmode=False)
    # print(bncftrs.get_contracts())
    print(bncftrs.get_contract('BNBBTC'))
# Ejemplo de uso:
# client = BinanceFuturesClient(testmode=True)
# strategies = TradingStrategies(client)
# result = strategies.implement_strategy('BTCUSDT', '1h', strategy='sma_crossover')
# print(f"Señal de trading: {result}")
# 
# backtest_df = strategies.backtest_strategy('BTCUSDT', '1h', strategy='sma_crossover', start_date='2023-01-01', end_date='2023-12-31')    