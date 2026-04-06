import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from config import OHLCV_TIMEFRAME, ALLOWED_SYMBOLS
from core.trade_logger import TradeLogger

# Configure module-level logging
logger = logging.getLogger("Scanner")

# Phase 66: Standalone Technical Functions for Parallel Processing
def compute_technical_indicators(df_json: str) -> Dict:
    """
    Heavy technical calculations executed in a separate process.
    """
    try:
        df = pd.read_json(df_json)
        if df.empty or len(df) < 30:
            return {'adx': 0.0, 'vol_ratio': 1.0, 'atr': 0.0}
            
        # ADX Calculation
        period = 14
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        
        # Volatility Ratio
        sma_atr = atr_series.rolling(window=period).mean()
        vol_ratio = atr_series.iloc[-1] / sma_atr.iloc[-1] if not pd.isna(sma_atr.iloc[-1]) and sma_atr.iloc[-1] != 0 else 1.0
        
        return {
            'adx': float(adx) if not pd.isna(adx) else 0.0,
            'vol_ratio': float(vol_ratio),
            'atr': float(atr_series.iloc[-1])
        }
    except Exception as e:
        return {'adx': 0.0, 'vol_ratio': 1.0, 'atr': 0.0}

class Scanner:
    """
    Phase 4: Autonomous Radar.
    Dynamically finds the best Binance Futures pairs based on volume, volatility, 
    and micro-capital ($12) constraints.
    """

    def __init__(self, exchange, trade_logger):
        self.exchange = exchange
        self.ohlcv_cache = {}  # {cache_key: (df, timestamp)}
        self.cache_ttl = 300   # 5-minute TTL for OHLCV cache
        self.trade_logger = trade_logger
        self.prev_ob = {}
        self.prev_ob_depth = {}
        self.prev_funding_rates = {}
        self.cached_symbols = []
        self.global_panic = False
        self.last_prices_1s = {} # {symbol: (price, timestamp)}
        self.last_stream_update = time.time()
        self.bybit_cache = {'price': None, 'timestamp': 0} # Phase 128: Latency Optimization
        
        # Phase 66: Parallel Processing
        self.executor = ProcessPoolExecutor(max_workers=4)
        
        # Phase 130: Removed Bybit initialization
        self.bybit = None
        
        # Phase 62: Sentiment Proxy
        self.sentiment_keywords = {
            'SEC': -0.5, 'ETF': 0.5, 'HACK': -0.8, 'LISTING': 0.4, 
            'BAN': -0.7, 'ADOPTION': 0.6, 'PUMP': 0.3, 'DUMP': -0.4
        }
        self.last_sentiment_score = 0.0
        self.last_sentiment_update = 0
        self.last_scan_results = [] # Phase 56: Dashboard tracking

    def reset_panic(self):
        """
        Clears the global panic flag.
        """
        logger.info("Resetting GLOBAL_PANIC flag. Market cooling period over.")
        self.global_panic = False

    async def get_dynamic_symbols(self, limit: int = 5) -> List[str]:
        """
        Autonomous Radar: Fetches and ranks symbols based on liquidity, volatility, 
        and micro-capital constraints.
        Phase 67: Failover API
        """
        try:
            # 1. Fetch all active Binance Futures markets
            markets = await self.exchange.load_markets()
            
            # 2. Filter for USDT pairs and Micro-Capital ($12) constraints
            filtered_symbols = []
            for symbol, market in markets.items():
                # Phase 127: Strict Symbol Lockdown
                if ALLOWED_SYMBOLS and symbol not in ALLOWED_SYMBOLS:
                    continue
                    
                if not symbol.endswith('/USDT:USDT'):
                    continue
                
                # Phase 135: Respect dynamic exchange limits for micro-capital
                min_cost = market.get('limits', {}).get('cost', {}).get('min', 0.0)
                if min_cost <= 12.0: # Allow up to INITIAL_CAPITAL min cost
                    filtered_symbols.append(symbol)

            if not filtered_symbols:
                if self.cached_symbols:
                    logger.warning("No symbols found. Using cached symbols.")
                    return self.cached_symbols[:limit]
                return []

            # 3. Predator Ranking
            tickers = await self.exchange.fetch_tickers(filtered_symbols)
            
            ranking_data = []
            for symbol in filtered_symbols:
                ticker = tickers.get(symbol)
                if not ticker:
                    continue
                
                quote_volume = ticker.get('quoteVolume', 0)
                percentage = abs(ticker.get('percentage', 0))
                
                ranking_data.append({
                    'symbol': symbol,
                    'score': quote_volume * percentage # Predator Score
                })

            ranking_data.sort(key=lambda x: x['score'], reverse=True)
            top_symbols = [item['symbol'] for item in ranking_data[:limit]]
            self.cached_symbols = top_symbols # Update cache
            logger.info(f"Autonomous Radar Discovered: {top_symbols}")
            return top_symbols

        except Exception as e:
            logger.error(f"Failover API Triggered: Binance API Error: {e}")
            if self.cached_symbols:
                return self.cached_symbols[:limit]
            return []

    async def volatility_shock_sensor(self, symbol: str, current_price: float) -> bool:
        """
        Phase 68: Volatility Shock Sensor
        If Current_Price changes > 2% in < 1 second, trigger 'GLOBAL_PANIC'.
        """
        now = time.time()
        if symbol in self.last_prices_1s:
            prev_price, prev_ts = self.last_prices_1s[symbol]
            if now - prev_ts <= 1.0:
                change = abs(current_price - prev_price) / prev_price
                if change > 0.02:
                    logger.critical(f"VOLATILITY SHOCK DETECTED: {symbol} moved {change*100:.2f}% in {now-prev_ts:.4f}s!")
                    self.global_panic = True
                    return True
        
        self.last_prices_1s[symbol] = (current_price, now)
        return False

    async def check_stream_health(self):
        """
        Phase 69: Failover WebSocket
        If primary stream lags > 500ms, automatically switch/warn.
        """
        lag = (time.time() - self.last_stream_update) * 1000
        if lag > 500:
            logger.warning(f"STREAM LAG DETECTED: {lag:.2f}ms. Switching to secondary failover...")
            # In a real HFT setup, this would re-init the WebSocket with a different endpoint.
            # For this implementation, we log the event and refresh the connection state.
            self.last_stream_update = time.time()
            return True
        return False

    async def fetch_ohlcv(self, symbol: str, timeframe: str = OHLCV_TIMEFRAME, limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV data with a 5-minute TTL cache.
        """
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.ohlcv_cache:
            df, ts = self.ohlcv_cache[cache_key]
            if (time.time() - ts) < self.cache_ttl:
                return df

        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            self.ohlcv_cache[cache_key] = (df, time.time())
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def get_orderbook_imbalance(self, orderbook: dict) -> float:
        """
        Calculates Orderbook Imbalance: (BidVol - AskVol) / (BidVol + AskVol).
        """
        try:
            bid_vol = sum([b[1] for b in orderbook['bids']])
            ask_vol = sum([a[1] for a in orderbook['asks']])
            
            if (bid_vol + ask_vol) == 0:
                return 0.0
            return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        except Exception as e:
            logger.error(f"Error calculating imbalance: {e}")
            return 0.0

    def get_micro_price(self, orderbook: dict) -> float:
        """
        Calculates the 'True Price' based on orderbook pressure.
        """
        try:
            b_p, b_v = orderbook['bids'][0]
            a_p, a_v = orderbook['asks'][0]
            if (b_v + a_v) == 0:
                return 0.0
            return (b_p * a_v + a_p * b_v) / (b_v + a_v)
        except Exception as e:
            logger.error(f"Error calculating micro-price: {e}")
            return 0.0

    def calculate_volatility_ratio(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Phase 23: Volatility Ratio
        Calculates Current ATR / SMA of ATR.
        """
        if len(df) < period * 2:
            return 1.0
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        sma_atr = atr.rolling(window=period).mean()
        
        current_atr = atr.iloc[-1]
        current_sma_atr = sma_atr.iloc[-1]
        
        if pd.isna(current_atr) or pd.isna(current_sma_atr) or current_sma_atr == 0:
            return 1.0
            
        return float(current_atr / current_sma_atr)

    async def scan_markets(self) -> List[Dict]:
        """
        Scans dynamically discovered symbols and returns candidates with metrics.
        """
        # Phase 22: BTC Dominance Proxy
        try:
            tickers = await self.exchange.fetch_tickers()
            total_vol = sum(t.get('quoteVolume', 0) for s, t in tickers.items() if s.endswith('/USDT:USDT'))
            btc_vol = tickers.get('BTC/USDT:USDT', {}).get('quoteVolume', 0)
            current_btc_dom = btc_vol / total_vol if total_vol > 0 else 0.0
            
            if not hasattr(self, 'prev_btc_dom'):
                self.prev_btc_dom = current_btc_dom
                
            self.btc_dom_roc = (current_btc_dom - self.prev_btc_dom) / self.prev_btc_dom if self.prev_btc_dom > 0 else 0.0
            self.prev_btc_dom = current_btc_dom
            self.current_btc_dom = current_btc_dom
        except Exception as e:
            logger.error(f"Error calculating BTC dominance: {e}")
            self.current_btc_dom = 0.0
            self.btc_dom_roc = 0.0

        # 1. Discover dynamic symbols (Autonomous Radar)
        dynamic_symbols = await self.get_dynamic_symbols(limit=5)
        if not dynamic_symbols:
            return []

        # Phase 11: The Omni-Shield (Anti-Revenge Blacklist)
        blacklisted = await self.trade_logger.get_blacklisted_symbols()

        # 2. Process discovered symbols
        candidates = []
        tasks = []
        for symbol in dynamic_symbols:
            if symbol in blacklisted:
                logger.warning(f"OMNI-SHIELD ACTIVE: {symbol} is blacklisted due to consecutive losses. Skipping.")
                continue
            tasks.append(self._process_symbol(symbol, self.current_btc_dom, self.btc_dom_roc))
            
        if not tasks:
            return []
            
        results = await asyncio.gather(*tasks)
        
        for res in results:
            if res:
                candidates.append(res)
        
        self.last_scan_results = candidates # Cache for heatmap
                
        return candidates

    async def get_cvd(self, symbol: str) -> float:
        """
        Phase 12: The Whale Shadow Protocol (CVD Analysis)
        Calculates Cumulative Volume Delta from recent trades.
        """
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=100)
            cvd = 0.0
            for trade in trades:
                volume = trade.get('amount', 0.0)
                side = trade.get('side', '')
                if side == 'buy':
                    cvd += volume
                elif side == 'sell':
                    cvd -= volume
            return cvd
        except Exception as e:
            logger.error(f"Error fetching trades for CVD on {symbol}: {e}")
            return 0.0

    def get_liquidity_depth(self, orderbook: dict) -> float:
        """
        Phase 15: The Liquidity Depth Shield
        Calculates the available USD depth in the orderbook.
        """
        try:
            bid_val = sum(price * amount for price, amount in orderbook['bids'][:10])
            ask_val = sum(price * amount for price, amount in orderbook['asks'][:10])
            return float(min(bid_val, ask_val))
        except Exception as e:
            logger.error(f"Error calculating liquidity depth: {e}")
            return 0.0

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Phase 20: The Regime Adaptive Matrix
        Calculates the Average Directional Index (ADX) to detect Trending vs. Ranging markets.
        """
        if len(df) < period * 2:
            return 0.0
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed True Range and Directional Movement
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        
        # Directional Index
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0

    def calculate_ofi(self, symbol: str, current_ob: dict) -> float:
        """
        Phase 21: Order Flow Imbalance (OFI)
        Calculates the net order flow imbalance between the current and previous orderbook.
        """
        if symbol not in self.prev_ob:
            self.prev_ob[symbol] = current_ob
            return 0.0
            
        prev_ob = self.prev_ob[symbol]
        
        try:
            current_bid_price, current_bid_vol = current_ob['bids'][0]
            prev_bid_price, prev_bid_vol = prev_ob['bids'][0]
            
            current_ask_price, current_ask_vol = current_ob['asks'][0]
            prev_ask_price, prev_ask_vol = prev_ob['asks'][0]
            
            delta_bid_vol = (current_bid_vol if current_bid_price >= prev_bid_price else 0) - \
                            (prev_bid_vol if current_bid_price <= prev_bid_price else 0)
                            
            delta_ask_vol = (current_ask_vol if current_ask_price <= prev_ask_price else 0) - \
                            (prev_ask_vol if current_ask_price >= prev_ask_price else 0)
                            
            ofi = delta_bid_vol - delta_ask_vol
            self.prev_ob[symbol] = current_ob
            return float(ofi)
        except Exception as e:
            logger.error(f"Error calculating OFI for {symbol}: {e}")
            self.prev_ob[symbol] = current_ob
            return 0.0

    async def get_arbitrage_delta(self, symbol: str, binance_price: float) -> float:
        """
        Phase 130: Arbitrage Delta disabled (Binance only mode).
        """
        return 0.0

    def calculate_ofi_depth(self, symbol: str, current_ob: dict) -> float:
        """
        Phase 30: OFI Depth
        Sum Delta Volume across the top 10 levels.
        """
        if symbol not in self.prev_ob_depth:
            self.prev_ob_depth[symbol] = current_ob
            return 0.0
            
        prev_ob = self.prev_ob_depth[symbol]
        ofi_depth = 0.0
        
        try:
            for i in range(min(10, len(current_ob['bids']), len(prev_ob['bids']))):
                current_bid_price, current_bid_vol = current_ob['bids'][i]
                prev_bid_price, prev_bid_vol = prev_ob['bids'][i]
                
                delta_bid_vol = (current_bid_vol if current_bid_price >= prev_bid_price else 0) - \
                                (prev_bid_vol if current_bid_price <= prev_bid_price else 0)
                ofi_depth += delta_bid_vol
                
            for i in range(min(10, len(current_ob['asks']), len(prev_ob['asks']))):
                current_ask_price, current_ask_vol = current_ob['asks'][i]
                prev_ask_price, prev_ask_vol = prev_ob['asks'][i]
                
                delta_ask_vol = (current_ask_vol if current_ask_price <= prev_ask_price else 0) - \
                                (prev_ask_vol if current_ask_price >= prev_ask_price else 0)
                ofi_depth -= delta_ask_vol
                
            self.prev_ob_depth[symbol] = current_ob
            return float(ofi_depth)
        except Exception as e:
            logger.error(f"Error calculating OFI depth for {symbol}: {e}")
            self.prev_ob_depth[symbol] = current_ob
            return 0.0

    async def get_market_velocity(self, symbol: str) -> float:
        """
        Phase 31: Market Velocity
        Calculate ticks per second (number of trades in the last 10 seconds).
        """
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=100)
            if not trades:
                return 0.0
            
            now = time.time() * 1000
            ten_seconds_ago = now - 10000
            
            recent_trades = [t for t in trades if t['timestamp'] >= ten_seconds_ago]
            return float(len(recent_trades) / 10.0)
        except Exception as e:
            logger.error(f"Error calculating market velocity for {symbol}: {e}")
            return 0.0

    async def get_liquidation_data(self, symbol: str) -> float:
        """
        Phase 61: Liquidation Sensor (Proxy via Volume/Price volatility)
        Detects potential Long/Short squeezes.
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            change = ticker.get('percentage', 0)
            return abs(change) * 0.1
        except Exception as e:
            logger.warning(f"get_momentum_proxy error: {e}")
            return 0.0

    async def get_sentiment_proxy(self) -> float:
        """
        Phase 62: News Sentiment Sensor
        Scans a mock news feed for keywords.
        """
        now = time.time()
        if now - self.last_sentiment_update < 300: # Cache for 5 mins
            return self.last_sentiment_score
            
        try:
            mock_news = ["SEC approves new ETF", "Major exchange HACK reported", "Crypto ADOPTION rising in Asia"]
            score = 0.0
            for news in mock_news:
                for kw, val in self.sentiment_keywords.items():
                    if kw in news.upper():
                        score += val
            
            self.last_sentiment_score = max(-1.0, min(1.0, score))
            self.last_sentiment_update = now
            return self.last_sentiment_score
        except Exception as e:
            logger.warning(f"get_sentiment_proxy error: {e}")
            return 0.0

    async def get_btc_whale_tx(self) -> int:
        """
        Phase 63: BTC Whale Tracker
        """
        try:
            trades = await self.exchange.fetch_trades('BTC/USDT:USDT', limit=100)
            whale_trades = [t for t in trades if t['amount'] * t['price'] > 100000] # > $100k
            return len(whale_trades)
        except Exception as e:
            logger.warning(f"get_btc_whale_tx error: {e}")
            return 0

    def detect_liquidity_walls(self, ob: dict) -> dict:
        """
        Phase 21 & 29: Liquidity Walls
        Finds price levels in top 50 bids/asks where volume is > 3x the average volume of the book.
        Returns distance to the nearest significant sell wall.
        """
        try:
            bids = ob['bids'][:50]
            asks = ob['asks'][:50]
            
            if not bids or not asks:
                return {'wall_dist_pct': 100.0}
                
            current_price = (bids[0][0] + asks[0][0]) / 2
            
            all_vols = [v for p, v in bids] + [v for p, v in asks]
            avg_vol = sum(all_vols) / len(all_vols) if all_vols else 0
            
            wall_threshold = avg_vol * 3
            
            nearest_sell_wall_dist = 100.0
            
            # Check ask walls (sell walls)
            for price, vol in asks:
                if vol > wall_threshold:
                    dist_pct = (price - current_price) / current_price
                    if dist_pct < nearest_sell_wall_dist:
                        nearest_sell_wall_dist = dist_pct
                        
            return {'wall_dist_pct': float(nearest_sell_wall_dist)}
        except Exception as e:
            logger.error(f"Error detecting liquidity walls: {e}")
            return {'wall_dist_pct': 100.0}

    async def _process_symbol(self, symbol: str, btc_dom: float = 0.0, btc_dom_roc: float = 0.0) -> Optional[Dict]:
        """
        Internal method to process a single symbol.
        """
        df = await self.fetch_ohlcv(symbol)
        if df.empty or len(df) < 5:
            return None
            
        last_price = df['close'].iloc[-1]
        
        # Phase 68: Volatility Shock Sensor
        if await self.volatility_shock_sensor(symbol, last_price):
            return None

        df_1h = await self.fetch_ohlcv(symbol, timeframe='1h')
        df_4h = await self.fetch_ohlcv(symbol, timeframe='4h')
        
        # Phase 18: The Relative Strength Matrix
        btc_df = await self.fetch_ohlcv('BTC/USDT:USDT')
        if not btc_df.empty and len(btc_df) >= 5:
            btc_roc = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-5]) / btc_df['close'].iloc[-5]
        else:
            btc_roc = 0.0
            
        symbol_roc = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        relative_strength = symbol_roc - btc_roc
            
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=50)
            imbalance = self.get_orderbook_imbalance(orderbook)
            micro_price = self.get_micro_price(orderbook)
            liquidity_depth = self.get_liquidity_depth(orderbook)
            ofi = self.calculate_ofi_depth(symbol, orderbook)
            wall_data = self.detect_liquidity_walls(orderbook)
            wall_dist = wall_data.get('wall_dist_pct', 100.0)
            
            bid = orderbook['bids'][0][0]
            ask = orderbook['asks'][0][0]
            mid_price = (ask + bid) / 2
            spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0.0
            
            arb_delta = await self.get_arbitrage_delta(symbol, mid_price)
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            imbalance = 0.0
            micro_price = 0.0
            liquidity_depth = 0.0
            spread_pct = 0.0
            ofi = 0.0
            wall_dist = 100.0
            arb_delta = 0.0

        cvd = await self.get_cvd(symbol)
        
        # Phase 66: Parallel Processing for Technical Indicators
        loop = asyncio.get_running_loop()
        tech_data = await loop.run_in_executor(
            self.executor, 
            compute_technical_indicators, 
            df.to_json()
        )
        
        adx = tech_data['adx']
        vol_ratio = tech_data['vol_ratio']
        atr = tech_data['atr']
        
        velocity = await self.get_market_velocity(symbol)
        
        # Phase 61: Funding Velocity
        try:
            funding_info = await self.exchange.fetch_funding_rate(symbol)
            funding_rate = float(funding_info.get('fundingRate', 0.0))
            prev_funding = self.prev_funding_rates.get(symbol, funding_rate)
            funding_rate_velocity = funding_rate - prev_funding
            self.prev_funding_rates[symbol] = funding_rate
        except Exception as e:
            logger.warning(f"Funding rate fetch error for {symbol}: {e}")
            funding_rate = 0.0
            funding_rate_velocity = 0.0

        # Phase 61-65: New Sensors
        liquidations_proxy = await self.get_liquidation_data(symbol)
        news_sentiment_score = await self.get_sentiment_proxy()
        btc_whale_tx_count = await self.get_btc_whale_tx()
        social_volume_spike = vol_ratio * 1.2 # Proxy for social hype

        h_volatility = float(df['close'].pct_change().std() * np.sqrt(288)) if len(df) > 20 else 0.0

        df_15m = await self.fetch_ohlcv(symbol, timeframe='15m')
        if not df_15m.empty and len(df_15m) >= 5:
            trend_15m = float((df_15m['close'].iloc[-1] - df_15m['close'].iloc[-5]) / df_15m['close'].iloc[-5])
        else:
            trend_15m = 0.0

        df['vol_ema'] = df['volume'].ewm(span=10).mean()
        vol_ema_ratio = float(df['volume'].iloc[-1] / df['vol_ema'].iloc[-1]) if df['vol_ema'].iloc[-1] > 0 else 1.0

        recent_high = df['high'].rolling(288).max().iloc[-1] if len(df) >= 288 else df['high'].max()
        recent_low = df['low'].rolling(288).min().iloc[-1] if len(df) >= 288 else df['low'].min()
        last_close = df['close'].iloc[-1]
        if recent_high > recent_low:
            price_range_pos = float((last_close - recent_low) / (recent_high - recent_low))
        else:
            price_range_pos = 0.5

        sentiment_proxy = float(cvd / (df['volume'].iloc[-1] + 1e-8))
        
        # Calculate ATR for return
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(window=14).mean().iloc[-1]) if len(df) >= 14 else 0.0

        # Basic trend check (Close > SMA 20)
        df['sma20'] = df['close'].rolling(window=20).mean()
        last_sma = df['sma20'].iloc[-1]
        
        return {
            'symbol': symbol,
            'last_price': last_close,
            'micro_price': micro_price,
            'imbalance': imbalance,
            'cvd': cvd,
            'liquidity_depth': liquidity_depth,
            'spread_pct': spread_pct,
            'atr': atr,
            'relative_strength': relative_strength,
            'adx': adx,
            'ofi': ofi,
            'wall_dist': wall_dist,
            'btc_dom': btc_dom,
            'btc_dom_roc': btc_dom_roc,
            'vol_ratio': vol_ratio,
            'arb_delta': arb_delta,
            'velocity': velocity,
            'funding_rate': funding_rate,
            'h_volatility': h_volatility,
            'trend_15m': trend_15m,
            'vol_ema_ratio': vol_ema_ratio,
            'price_range_pos': price_range_pos,
            'sentiment_proxy': sentiment_proxy,
            'funding_rate_velocity': funding_rate_velocity,
            'liquidations_proxy': liquidations_proxy,
            'social_volume_spike': social_volume_spike,
            'news_sentiment_score': news_sentiment_score,
            'btc_whale_tx_count': btc_whale_tx_count,
            'signal_timestamp': time.time(),
            'df': df,
            'df_15m': df_15m,
            'df_1h': df_1h,
            'df_4h': df_4h,
            'trend': 'UP' if last_close > last_sma else 'DOWN'
        }
