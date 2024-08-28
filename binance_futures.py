import json
import requests
import urllib.parse
import re
import logging
from config.api_keys import API_KEY, SECRET_KEY

logger = logging.getLogger()

class BinanceFuturesClient:
    def __init__(self, testmode):
        if testmode:
            self.endpoints = ['https://api.binance.com', 'https://api-gcp.binance.com', 'https://api1.binance.com', 'https://api2.binance.com', 'https://api3.binance.com', 'https://api4.binance.com']
            logger.warning('Connected to PROD Environment')
        else:
            self.endpoints = ['https://testnet.binance.vision']
            logger.info('Connected to TEST Environment')

    input_data = { 'api_token': '{API_KEY}'}

    def get_objects(self, object):
        try:
            url =  self.endpoints[0]  + '/api/v3/' + str(object)
            headers = {"Content-Type" :"application/json"}     
            req = requests.get(url, headers=headers )
            if req.status_code // 100 == 2:
                logger.debug("La llamada a la API fue exitosa")
                response_dict = json.loads(req.text)
                return  response_dict
            else:
                logger.error(f"Error en la llamada a la API. Código de estado: {req.status_code}")
                logger.debug(f"Mensaje de error: {req.text}")
                return None  
        except requests.RequestException as e:
            logger.error(f"Ocurrió un error al hacer la solicitud: {e}")
            return None

        
    def get_object(self, object, element, id):
        try:
            url = self.endpoints[0]  + '/api/v3/' + str(object) +  '?' + str(element) +  '=' + str(id) 
            headers = {"Content-Type" :"application/json"}     
            req = requests.get(url, headers=headers )
            if req.status_code // 100 == 2:
                logger.debug("La llamada a la API fue exitosa")
                response_dict = json.loads(req.text)
                return  response_dict
            else:
                logger.error(f"Error en la llamada a la API. Código de estado: {req.status_code}")
                logger.debug(f"Mensaje de error: {req.text}")
                return None  
        except requests.RequestException as e:
            logger.error(f"Ocurrió un error al hacer la solicitud: {e}")
            return None
  
# (HMAC SHA256)
# [linux]$ curl -H ": " -X POST '

#     url = 'https://api.binance.com/api/v3/order?symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559&signature=c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71'
# ' + str(api_token)
#     headers = {"X-MBX-APIKEY" :"vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A"}     
#     req = requests.get(url, headers=headers ).json()

    def get_contracts(self):
        return self.get_objects("exchangeInfo")
   
    def get_contract(self, id):
        return self.get_object("exchangeInfo", "symbol", id)
    
    def get_bid_ask(self): 
        return self.get_objects("ticker/bookTicker")

    def get_historical_candles(self):
        return self.get_objects("exchangeInfo", start)






