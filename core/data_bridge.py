import logging
import time
import ccxt
import yfinance as yf
import pandas as pd

logger = logging.getLogger("DataBridge")

class UniversalDataBridge:
    """
    The Universal Data Bridge for Project Digital Evolution.
    Seamlessly routes data requests and order execution between Crypto Markets (Binance)
    and the Indian Stock Market (NSE/BSE).
    """

    def __init__(self, exchange=None):
        # Initialize Binance for crypto by default if no exchange is provided
        self.exchange = exchange or ccxt.binance({
            'enableRateLimit': True,
            # 'apiKey': config.BINANCE_API_KEY,  # To be injected by main.py
            # 'secret': config.BINANCE_SECRET,
        })

    def _identify_market(self, symbol: str) -> str:
        """
        Identifies the target market based on the symbol's naming convention.
        """
        if '/USDT' in symbol:
            return 'CRYPTO'
        elif '.NS' in symbol or '.BO' in symbol:
            return 'STOCKS'
        else:
            raise ValueError(f"Unknown market for symbol: {symbol}. Must contain '/USDT', '.NS', or '.BO'")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV data and normalizes the output to a standard Pandas DataFrame
        regardless of the underlying market or API.
        """
        market = self._identify_market(symbol)
        
        if market == 'CRYPTO':
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except ccxt.NetworkError as e:
                logger.error(f"CCXT Network Error fetching {symbol}: {e}")
            except Exception as e:
                logger.error(f"CCXT Error fetching {symbol}: {e}")
            return pd.DataFrame()
            
        elif market == 'STOCKS':
            try:
                # yfinance timeframe mapping (mostly matches ccxt, e.g., '15m', '1h', '1d')
                yf_interval = timeframe
                
                # yfinance requires a 'period'. We fetch a safe buffer to ensure we get 'limit' rows.
                period = "1mo" if timeframe in ['1m', '5m', '15m', '1h'] else "2y"
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=yf_interval)
                
                if df.empty:
                    logger.warning(f"No data returned from yfinance for {symbol}")
                    return pd.DataFrame()
                    
                # Format to exactly match the CCXT Pandas structure
                df.reset_index(inplace=True)
                
                # yfinance index column might be 'Date' or 'Datetime' depending on the interval
                date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
                
                df.rename(columns={
                    date_col: 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
                
                # Keep only standard columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Standardize timezone to UTC (yfinance often returns timezone-aware datetimes)
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                    
                # Return exactly the requested number of rows
                return df.tail(limit).reset_index(drop=True)
                
            except Exception as e:
                logger.error(f"yfinance Connection/Fetch Error for {symbol}: {e}")
            return pd.DataFrame()

    def place_order(self, symbol: str, side: str, quantity: float, price: float = None) -> dict:
        """
        Routes the order execution to the appropriate exchange/broker API.
        """
        market = self._identify_market(symbol)
        order_type = 'limit' if price else 'market'
        
        if market == 'CRYPTO':
            try:
                logger.info(f"Routing {market} {order_type} {side} order for {quantity} {symbol} at {price} via CCXT")
                # In production, uncomment the line below to execute real trades
                # return self.exchange.create_order(symbol, order_type, side, quantity, price)
                
                # Returning a mock successful response for safety during development
                return {
                    "status": "success",
                    "market": market,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price
                }
            except ccxt.BaseError as e:
                logger.error(f"CCXT Order Error for {symbol}: {e}")
                return {"status": "failed", "error": str(e)}
                
        elif market == 'STOCKS':
            return self._place_stock_order(symbol, side, quantity, price)

    def _place_stock_order(self, symbol: str, side: str, quantity: float, price: float = None) -> dict:
        """
        Placeholder for Indian Stock Market broker API (e.g., Upstox, Zerodha Kite, AngelOne).
        Currently logs the intended action and simulates a successful response.
        """
        order_type = 'LIMIT' if price else 'MARKET'
        
        # Log the exact action that would be sent to the broker
        logger.info(
            f"[INDIAN STOCKS API PLACEHOLDER] "
            f"Action: {side.upper()} | Symbol: {symbol} | Qty: {quantity} | Type: {order_type} | Price: {price}"
        )
        
        # TODO: Integrate Indian Broker API here
        # Example (Zerodha Kite):
        # try:
        #     order_id = kite.place_order(
        #         tradingsymbol=symbol.replace('.NS', ''),
        #         exchange=kite.EXCHANGE_NSE,
        #         transaction_type=kite.TRANSACTION_TYPE_BUY if side.lower() == 'buy' else kite.TRANSACTION_TYPE_SELL,
        #         quantity=int(quantity),
        #         order_type=kite.ORDER_TYPE_LIMIT if price else kite.ORDER_TYPE_MARKET,
        #         price=price
        #     )
        #     return {"status": "success", "order_id": order_id}
        # except Exception as e:
        #     logger.error(f"Broker API Error: {e}")
        #     return {"status": "failed", "error": str(e)}
        
        return {
            "status": "simulated_success",
            "market": "STOCKS",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price
        }
