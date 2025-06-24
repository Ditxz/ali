import os
import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from collections import deque
import sqlite3
import math

# FIXED: Menggunakan ccxt.async_support untuk konsistensi async/await
from ccxt.async_support import binance as async_binance
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import requests
import numpy as np

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("trading_bot.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Global Configuration from Environment Variables ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
MODE = os.getenv('MODE', 'paper').lower() # 'live' or 'paper'
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100')) # For stats tracking
LEVERAGE = int(os.getenv('LEVERAGE', '25'))
FIXED_SL_PERCENT = float(os.getenv('FIXED_SL_PERCENT', '0.003')) # 0.3%
DYNAMIC_TP_MIN_PERCENT = float(os.getenv('DYNAMIC_TP_MIN_PERCENT', '0.005')) # 0.5%
DYNAMIC_TP_MAX_PERCENT = float(os.getenv('DYNAMIC_TP_MAX_PERCENT', '0.02')) # 2%
TRAIL_TP_PERCENT = float(os.getenv('TRAIL_TP_PERCENT', '0.001')) # 0.1% for trailing activation
TRADE_INTERVAL_SECONDS = int(os.getenv('TRADE_INTERVAL_SECONDS', '30'))
MAX_POSITIONS_PER_PAIR = int(os.getenv('MAX_POSITIONS_PER_PAIR', '1')) # Anti-overtrade logic
MIN_NOTIONAL_USDT = float(os.getenv('MIN_NOTIONAL_USDT', '5.0')) # Minimum trade size in USDT notional value

# --- Supported Pairs and Timeframe ---
SUPPORTED_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT',
    'OP/USDT', 'LINK/USDT', 'APT/USDT', 'RNDR/USDT'
]
TIMEFRAME = '1m' # Very aggressive, high frequency

# --- Global State Variables ---
binance_client = None
scheduler = None
db_connection = None
db_cursor = None
open_positions = {} # {symbol: {entry_price, amount, side, sl_price, tp_price, trailing_active, profit_usdt, created_at}}
trade_history = deque(maxlen=1000) # Store recent trades for /stats endpoint

# --- SQLite Database Initialization ---
DATABASE_NAME = 'trading_bot.db'

def init_db():
    """Initializes the SQLite database and creates the trades table."""
    global db_connection, db_cursor
    try:
        db_connection = sqlite3.connect(DATABASE_NAME)
        db_cursor = db_connection.cursor()
        db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                amount REAL,
                leverage INTEGER,
                pnl_usdt REAL,
                pnl_percent REAL,
                status TEXT, -- 'OPEN', 'CLOSED_SL', 'CLOSED_TP', 'CLOSED_MANUAL'
                initial_capital REAL,
                current_capital REAL
            )
        """)
        db_connection.commit()
        logger.info(f"Database '{DATABASE_NAME}' initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")

def log_trade(data):
    """Logs a trade event to the database and in-memory history."""
    try:
        db_cursor.execute("""
            INSERT INTO trades (timestamp, symbol, direction, entry_price, exit_price, amount, leverage, pnl_usdt, pnl_percent, status, initial_capital, current_capital)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            data.get('symbol'),
            data.get('direction'),
            data.get('entry_price'),
            data.get('exit_price'),
            data.get('amount'),
            data.get('leverage'),
            data.get('pnl_usdt'),
            data.get('pnl_percent'),
            data.get('status'),
            data.get('initial_capital'),
            data.get('current_capital')
        ))
        db_connection.commit()
        trade_history.append(data) # Add to in-memory history
        logger.info(f"Trade logged: {data.get('symbol')} {data.get('status')} PnL: {data.get('pnl_usdt'):.2f} USDT")
    except sqlite3.Error as e:
        logger.error(f"Error logging trade: {e}")

def get_trade_history_from_db():
    """Retrieves all trade history from the database."""
    try:
        db_cursor.execute("SELECT * FROM trades ORDER BY timestamp ASC")
        rows = db_cursor.fetchall()
        # Convert rows to dicts for easier consumption if needed, or just return raw rows
        return rows
    except sqlite3.Error as e:
        logger.error(f"Error fetching trade history from DB: {e}")
        return []

# --- Telegram Notifier ---
class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    async def send_message(self, message):
        """Sends a text message to the Telegram chat."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not configured. Skipping notification.")
            return

        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Telegram notification sent: {message[:50]}...")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Telegram message: {e}")

    async def send_signal(self, signal_data):
        """Formats and sends a trading signal to Telegram."""
        message = (
            f"🔔 *AI SIGNAL*\n"
            f"Pair: `{signal_data['symbol']}`\n"
            f"Direction: `{signal_data['direction'].upper()}`\n"
            f"Entry: `{signal_data['entry_price']:,}`\n"
            f"SL: `{signal_data['sl_price']:,}`\n"
            f"TP: `{signal_data['tp_price']:,}`\n"
            f"Confidence: `{signal_data['confidence']}%`"
        )
        await self.send_message(message)

    async def send_trade_update(self, trade_data):
        """Formats and sends a trade update to Telegram."""
        status_emoji = "✅" if trade_data['status'].startswith('CLOSED_TP') else ("❌" if trade_data['status'].startswith('CLOSED_SL') else "🔄")
        message = (
            f"{status_emoji} *TRADE UPDATE*\n"
            f"Pair: `{trade_data['symbol']}`\n"
            f"Direction: `{trade_data['direction'].upper()}`\n"
            f"Status: `{trade_data['status'].replace('_', ' ')}`\n"
            f"Entry: `{trade_data['entry_price']:,}`\n"
            f"Exit: `{trade_data['exit_price']:,}`\n"
            f"PnL: `{trade_data['pnl_usdt']:.2f} USDT` (`{trade_data['pnl_percent']:.2f}%`)\n"
            f"Current Capital: `{trade_data['current_capital']:.2f} USDT`"
        )
        await self.send_message(message)

# --- Risk Management ---
class RiskManager:
    def __init__(self, leverage, fixed_sl_percent, dynamic_tp_min_percent, dynamic_tp_max_percent, trail_tp_percent):
        self.leverage = leverage
        self.fixed_sl_percent = fixed_sl_percent
        self.dynamic_tp_min_percent = dynamic_tp_min_percent
        self.dynamic_tp_max_percent = dynamic_tp_max_percent
        self.trail_tp_percent = trail_tp_percent

    def calculate_sl_tp(self, entry_price, direction, symbol):
        """Calculates Stop Loss and Take Profit prices."""
        # FIXED: Tambahkan pengecekan symbol di markets
        if symbol not in binance_client.markets:
            logger.error(f"Symbol {symbol} not found in markets during SL/TP calculation. Cannot calculate SL/TP.")
            return None, None

        # Using a fixed percentage based on entry price
        if direction == 'long':
            sl_price = entry_price * (1 - self.fixed_sl_percent)
            tp_price = entry_price * (1 + np.random.uniform(self.dynamic_tp_min_percent, self.dynamic_tp_max_percent))
        elif direction == 'short':
            sl_price = entry_price * (1 + self.fixed_sl_percent)
            tp_price = entry_price * (1 - np.random.uniform(self.dynamic_tp_min_percent, self.dynamic_tp_max_percent))
        else:
            raise ValueError("Direction must be 'long' or 'short'")

        # Ensure prices are rounded to appropriate precision for the symbol
        market = binance_client.markets[symbol] # FIXED: Akses via .markets
        price_precision = market['precision']['price']
        
        # FIXED: Menggunakan round() sebagai pengganti decimal_to_precision
        sl_price = round(sl_price, price_precision)
        tp_price = round(tp_price, price_precision)

        return float(sl_price), float(tp_price)

    def should_activate_trailing_tp(self, current_price, position_data):
        """Determines if trailing TP should be activated."""
        if position_data.get('trailing_active'):
            return False # Already active

        entry_price = position_data['entry_price']
        direction = position_data['side']
        profit_percent = 0

        if direction == 'long':
            profit_percent = (current_price - entry_price) / entry_price
        elif direction == 'short':
            profit_percent = (entry_price - current_price) / entry_price

        return profit_percent >= self.trail_tp_percent

    def update_trailing_stop(self, current_price, position_data):
        """Updates the trailing stop loss price."""
        direction = position_data['side']
        current_sl = position_data['sl_price']
        new_sl = current_sl

        # FIXED: Tambahkan pengecekan symbol di markets
        symbol = position_data['symbol']
        if symbol not in binance_client.markets:
            logger.error(f"Symbol {symbol} not found in markets during trailing SL update. Cannot update SL.")
            return current_sl

        # Fetching market info for precision
        market = binance_client.markets[symbol] # FIXED: Akses via .markets
        price_precision = market['precision']['price']

        if direction == 'long':
            # Trailing stop moves up with price, never down
            potential_sl = current_price * (1 - self.fixed_sl_percent) # Use fixed_sl_percent as the trailing offset
            new_sl = max(current_sl, potential_sl)
        elif direction == 'short':
            # Trailing stop moves down with price, never up
            potential_sl = current_price * (1 + self.fixed_sl_percent) # Use fixed_sl_percent as the trailing offset
            new_sl = min(current_sl, potential_sl)

        # FIXED: Menggunakan round() sebagai pengganti decimal_to_precision
        new_sl = round(new_sl, price_precision)
        return new_sl

# --- Signal Engine (AI Layer 1 & 2) ---
class SignalEngine:
    def __init__(self, binance_client):
        self.binance_client = binance_client
        self.ohlcv_cache = {} # Cache for OHLCV data to avoid redundant fetches

    async def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Fetches OHLCV data for a given symbol and timeframe."""
        cache_key = f"{symbol}_{timeframe}"
        # Simple cache invalidation: always fetch fresh data for real-time
        # In a real high-freq system, this might be event-driven or streamed
        try:
            ohlcv = await self.binance_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f"No OHLCV data fetched for {symbol} {timeframe}")
                return []
            self.ohlcv_cache[cache_key] = ohlcv
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return []

    # --- Indicator Calculations ---
    def calculate_ema(self, prices, period):
        """Calculates Exponential Moving Average (EMA)."""
        if len(prices) < period:
            return None
        ema = [0.0] * len(prices)
        smoothing_factor = 2 / (period + 1)
        ema[period - 1] = sum(prices[0:period]) / period # Simple MA for first value
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * smoothing_factor) + (ema[i-1] * (1 - smoothing_factor))
        return ema[-1] # Return the latest EMA value

    def calculate_cci(self, high_prices, low_prices, close_prices, period):
        """Calculates Commodity Channel Index (CCI)."""
        if len(close_prices) < period:
            return None
        tp_values = [(high_prices[i] + low_prices[i] + close_prices[i]) / 3 for i in range(len(close_prices))]
        
        typical_prices = np.array(tp_values[-period:])
        ma_typical_prices = np.mean(typical_prices)
        
        # Mean Deviation
        mean_deviation = np.mean(np.abs(typical_prices - ma_typical_prices))
        
        if mean_deviation == 0:
            return 0 # Avoid division by zero
        
        cci = (tp_values[-1] - ma_typical_prices) / (0.015 * mean_deviation)
        return cci

    def calculate_atr(self, high_prices, low_prices, close_prices, period):
        """Calculates Average True Range (ATR)."""
        if len(close_prices) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(close_prices)):
            tr = max(high_prices[i] - low_prices[i],
                     abs(high_prices[i] - close_prices[i-1]),
                     abs(low_prices[i] - close_prices[i-1]))
            true_ranges.append(tr)
        
        # Calculate EMA of True Ranges for ATR
        if len(true_ranges) < period:
            return None # Not enough data for initial EMA
        
        # Initial SMA for the first ATR value
        atr_values = [sum(true_ranges[0:period]) / period]
        
        smoothing_factor = 2 / (period + 1)
        for i in range(period, len(true_ranges)):
            atr_val = (true_ranges[i] * smoothing_factor) + (atr_values[-1] * (1 - smoothing_factor))
            atr_values.append(atr_val)
            
        return atr_values[-1]

    def calculate_bollinger_bands(self, prices, period, std_dev):
        """Calculates Bollinger Bands (Middle Band, Upper Band, Lower Band)."""
        if len(prices) < period:
            return None, None, None
        
        current_prices = np.array(prices[-period:])
        middle_band = np.mean(current_prices)
        std = np.std(current_prices)
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return middle_band, upper_band, lower_band

    # --- Layer 1: Entry AI (Rule-based + Heuristics) ---
    def detect_price_action(self, ohlcv_data):
        """
        Detects basic price action patterns: breakout, order block, FVG, volume surge, market structure.
        Returns a dictionary of detected patterns.
        """
        patterns = {
            'breakout_long': False,
            'breakout_short': False,
            'order_block_long': False,
            'order_block_short': False,
            'fvg_long': False,
            'fvg_short': False,
            'volume_surge': False,
            'bos_choch_long': False, # Break of Structure / Change of Character
            'bos_choch_short': False
        }

        if len(ohlcv_data) < 5: # Need at least a few candles
            return patterns

        closes = np.array([c[4] for c in ohlcv_data])
        highs = np.array([c[2] for c in ohlcv_data])
        lows = np.array([c[3] for c in ohlcv_data])
        volumes = np.array([c[5] for c in ohlcv_data])

        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_open = ohlcv_data[-1][1]
        current_volume = volumes[-1]

        # Breakout (simple: current candle breaks N-period high/low)
        lookback_period_breakout = 5
        if len(closes) >= lookback_period_breakout + 1:
            prev_highs = highs[-lookback_period_breakout-1:-1]
            prev_lows = lows[-lookback_period_breakout-1:-1]
            
            if current_close > np.max(prev_highs) and current_close > current_open:
                patterns['breakout_long'] = True
            elif current_close < np.min(prev_lows) and current_close < current_open:
                patterns['breakout_short'] = True

        # Volume Surge: Current volume significantly higher than average
        lookback_period_volume = 10
        if len(volumes) >= lookback_period_volume + 1:
            avg_volume = np.mean(volumes[-lookback_period_volume-1:-1])
            if current_volume > avg_volume * 1.5: # 50% higher than average
                patterns['volume_surge'] = True

        # Fair Value Gap (FVG): Simplified as a gap between candle 1 high/low and candle 3 low/high
        # e.g., for bullish FVG, C1.low > C3.high (or C1.low > C2.high & C2.high > C3.high)
        # For bullish FVG, look for (low of candle 1) > (high of candle 3) in a 3-candle sequence (1,2,3 from oldest to newest)
        if len(ohlcv_data) >= 3:
            c1_h, c1_l = ohlcv_data[-3][2], ohlcv_data[-3][3]
            c2_h, c2_l = ohlcv_data[-2][2], ohlcv_data[-2][3]
            c3_h, c3_l = ohlcv_data[-1][2], ohlcv_data[-1][3]

            # Bullish FVG: Current candle (c3) closes strong bullish, previous (c2) might be small, c1's low is above c3's high
            if c3_h > c3_l and (c3_h - c3_l) / c3_l > 0.001: # Check if c3 is a significant bullish candle
                if c1_l > c3_h and c2_h > c3_h: # Simple gap check: C1 low is above C3 high, C2 high is above C3 high
                    patterns['fvg_long'] = True
            
            # Bearish FVG: Current candle (c3) closes strong bearish, previous (c2) might be small, c1's high is below c3's low
            if c3_l < c3_h and (c3_h - c3_l) / c3_h > 0.001: # Check if c3 is a significant bearish candle
                if c1_h < c3_l and c2_l < c3_l: # Simple gap check: C1 high is below C3 low, C2 low is below C3 low
                    patterns['fvg_short'] = True

        # Order Block Zone (simplified): Look for a large bearish candle followed by a bullish one for long
        # or a large bullish candle followed by a bearish one for short, implying institutional interest.
        if len(ohlcv_data) >= 2:
            prev_open, prev_close = ohlcv_data[-2][1], ohlcv_data[-2][4]
            curr_open, curr_close = ohlcv_data[-1][1], ohlcv_data[-1][4]

            # Bullish Order Block: Significant bearish candle followed by a bullish reversal
            if prev_close < prev_open and abs(prev_close - prev_open) / prev_open > 0.002: # Significant bearish candle
                if curr_close > curr_open and curr_close > prev_close: # Bullish reversal
                    patterns['order_block_long'] = True
            
            # Bearish Order Block: Significant bullish candle followed by a bearish reversal
            if prev_close > prev_open and abs(prev_close - prev_open) / prev_open > 0.002: # Significant bullish candle
                if curr_close < curr_open and curr_close < prev_close: # Bearish reversal
                    patterns['order_block_short'] = True

        # Market Structure BOS / CHoCH (simplified): Looking for a clear higher high/lower low break
        # This is very basic. For proper BOS/CHoCH, a more robust swing point detection is needed.
        lookback_ms = 10
        if len(closes) >= lookback_ms:
            recent_highs = highs[-lookback_ms:]
            recent_lows = lows[-lookback_ms:]

            # Long CHoCH/BOS: If current close breaks above a previous swing high after a downtrend
            # Very simplified: Current close breaks the highest high of the last few candles (after a dip)
            if current_close > np.max(recent_highs[:-1]) and current_close > closes[-2]:
                # Check for a prior lower low or lower high to confirm CHoCH aspect
                if lows[-2] < np.min(lows[-lookback_ms-1:-2]): # Simple prior dip
                    patterns['bos_choch_long'] = True
            
            # Short CHoCH/BOS: If current close breaks below a previous swing low after an uptrend
            # Very simplified: Current close breaks the lowest low of the last few candles (after a rise)
            if current_close < np.min(recent_lows[:-1]) and current_close < closes[-2]:
                # Check for a prior higher high or higher low to confirm CHoCH aspect
                if highs[-2] > np.max(highs[-lookback_ms-1:-2]): # Simple prior rise
                    patterns['bos_choch_short'] = True

        return patterns

    # --- Layer 2: Confirmation Entry ---
    def confirm_entry(self, ohlcv_data, current_price, detected_patterns):
        """
        Confirms entry based on indicator combinations: EMA, CCI, SSL Hybrid (simplified), Volatility Filter, Volume Delta (simplified).
        Returns 'long', 'short', or None and a confidence score.
        """
        if len(ohlcv_data) < 50: # Need enough data for 50-period indicators
            return None, 0

        closes = [c[4] for c in ohlcv_data]
        highs = [c[2] for c in ohlcv_data]
        lows = [c[3] for c in ohlcv_data]
        volumes = [c[5] for c in ohlcv_data]

        ema5 = self.calculate_ema(closes, 5)
        ema20 = self.calculate_ema(closes, 20)
        ema50 = self.calculate_ema(closes, 50)

        cci50 = self.calculate_cci(highs, lows, closes, 50)
        cci100 = self.calculate_cci(highs, lows, closes, 100)

        atr = self.calculate_atr(highs, lows, closes, 14) # Standard ATR period
        bb_mid, bb_upper, bb_lower = self.calculate_bollinger_bands(closes, 20, 2) # Standard BB

        # Volume Delta Divergence (simplified): Check if current volume confirms direction of price movement
        # Or if volume is increasing on pullbacks for reversals
        volume_confirm_long = False
        volume_confirm_short = False
        if len(volumes) >= 2 and len(closes) >= 2:
            if closes[-1] > closes[-2] and volumes[-1] > volumes[-2]: # Price up, Volume up
                volume_confirm_long = True
            if closes[-1] < closes[-2] and volumes[-1] > volumes[-2]: # Price down, Volume up
                volume_confirm_short = True

        # --- Aggressive LONG Confirmation ---
        long_conditions_met = 0
        if ema5 and ema20 and ema50:
            if ema5 > ema20 > ema50: # EMA alignment
                long_conditions_met += 1
        
        if cci50 and cci100:
            if cci50 > -100 and cci100 > -100: # Not oversold for long
                long_conditions_met += 1
            if cci50 > cci100: # CCI cross up (positive momentum)
                long_conditions_met += 1

        if atr and atr > (current_price * 0.0005): # Volatility filter: ATR > 0.05% of price
            long_conditions_met += 1

        if bb_mid and bb_upper and bb_lower:
            if current_price > bb_mid: # Price above middle band
                long_conditions_met += 1

        if volume_confirm_long:
            long_conditions_met += 1

        # --- Aggressive SHORT Confirmation ---
        short_conditions_met = 0
        if ema5 and ema20 and ema50:
            if ema5 < ema20 < ema50: # EMA alignment
                short_conditions_met += 1

        if cci50 and cci100:
            if cci50 < 100 and cci100 < 100: # Not overbought for short
                short_conditions_met += 1
            if cci50 < cci100: # CCI cross down (negative momentum)
                short_conditions_met += 1
        
        if atr and atr > (current_price * 0.0005): # Volatility filter: ATR > 0.05% of price
            short_conditions_met += 1
        
        if bb_mid and bb_upper and bb_lower:
            if current_price < bb_mid: # Price below middle band
                short_conditions_met += 1

        if volume_confirm_short:
            short_conditions_met += 1

        total_possible_confirmations = 6 # (EMA, 2xCCI, ATR, BB, Volume)

        # Combine with Layer 1 patterns
        long_signal = False
        if detected_patterns['breakout_long'] or detected_patterns['order_block_long'] or \
           detected_patterns['fvg_long'] or detected_patterns['bos_choch_long']:
            if long_conditions_met >= 3 and detected_patterns['volume_surge']: # At least 3 confirmations + volume surge
                long_signal = True
        
        short_signal = False
        if detected_patterns['breakout_short'] or detected_patterns['order_block_short'] or \
           detected_patterns['fvg_short'] or detected_patterns['bos_choch_short']:
            if short_conditions_met >= 3 and detected_patterns['volume_surge']: # At least 3 confirmations + volume surge
                short_signal = True

        confidence_long = int((long_conditions_met / total_possible_confirmations) * 100) if long_conditions_met > 0 else 0
        confidence_short = int((short_conditions_met / total_possible_confirmations) * 100) if short_conditions_met > 0 else 0

        # Prioritize stronger signal if both are present (unlikely with this setup)
        if long_signal and not short_signal:
            return 'long', confidence_long
        elif short_signal and not long_signal:
            return 'short', confidence_short
        
        return None, 0

    async def generate_signal(self, symbol, timeframe=TIMEFRAME):
        """Generates a trading signal for a given symbol."""
        ohlcv_data = await self.fetch_ohlcv(symbol, timeframe, limit=100) # Fetch enough data for all indicators
        if not ohlcv_data or len(ohlcv_data) < 50:
            logger.warning(f"Insufficient OHLCV data for {symbol} to generate signal.")
            return None

        current_price = ohlcv_data[-1][4] # Last candle's close price

        detected_patterns = self.detect_price_action(ohlcv_data)
        direction, confidence = self.confirm_entry(ohlcv_data, current_price, detected_patterns)

        if direction:
            logger.info(f"Signal generated for {symbol}: {direction.upper()} with confidence {confidence}%")
            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'confidence': confidence
            }
        return None

# --- Core Trading Bot Logic ---
class TradingBot:
    def __init__(self, binance_client, signal_engine, risk_manager, telegram_notifier):
        self.binance_client = binance_client
        self.signal_engine = signal_engine
        self.risk_manager = risk_manager
        self.telegram_notifier = telegram_notifier
        self.current_capital = INITIAL_CAPITAL # Track capital for stats
        self.last_trade_time = {} # To track last signal for anti-overtrade

    async def get_balance(self):
        """Fetches available USDT balance."""
        try:
            balance = await self.binance_client.fetch_balance({'type': 'future'})
            return balance['USDT']['free']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0

    async def get_open_positions(self):
        """Fetches and updates internal open_positions state."""
        global open_positions
        try:
            positions = await self.binance_client.fetch_positions([s.replace('/USDT', '') for s in SUPPORTED_PAIRS])
            
            # Clear positions that are no longer open (closed by exchange, etc.)
            symbols_in_ccxt_positions = {p['symbol'] for p in positions if float(p['notional']) != 0}
            symbols_to_remove = [sym for sym in open_positions if sym not in symbols_in_ccxt_positions]
            for sym in symbols_to_remove:
                logger.info(f"Position for {sym} not found via CCXT, assuming closed. Removing from internal tracking.")
                # If a position disappears without explicit closure, it's considered closed.
                # Here we might need a more robust way to differentiate between 'closed by bot' and 'closed by exchange/manually'.
                # For simplicity, if it's not in CCXT's open positions, we remove it.
                del open_positions[sym]

            for position in positions:
                symbol = position['symbol']
                current_amount = float(position['notional']) / float(position['markPrice']) if float(position['markPrice']) != 0 else 0
                
                # Check if position is actually open (notional value is not zero)
                if abs(float(position['notional'])) > 0:
                    side = 'long' if float(position['positionAmt']) > 0 else 'short'
                    
                    if symbol not in open_positions:
                        # This should ideally not happen if bot places orders, but for robustness
                        # It means a position was opened externally or bot restarted.
                        # We need to initialize SL/TP for it.
                        entry_price = float(position['entryPrice'])
                        sl_price, tp_price = self.risk_manager.calculate_sl_tp(entry_price, side, symbol)
                        open_positions[symbol] = {
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'amount': current_amount,
                            'side': side,
                            'sl_price': sl_price,
                            'tp_price': tp_price,
                            'trailing_active': False,
                            'profit_usdt': 0, # To be calculated
                            'created_at': datetime.now().isoformat()
                        }
                        logger.warning(f"Detected un-tracked open position for {symbol}. Initialized SL/TP internally.")
                    else:
                        # Update existing position details if necessary
                        open_positions[symbol]['amount'] = current_amount
                        open_positions[symbol]['side'] = side
                        open_positions[symbol]['entry_price'] = float(position['entryPrice'])
                        # PnL calculation
                        if float(position['unrealizedPnl']) is not None:
                            open_positions[symbol]['profit_usdt'] = float(position['unrealizedPnl'])
                            
            logger.debug(f"Current open positions tracked: {list(open_positions.keys())}")
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")

    async def place_order(self, symbol, type, side, amount_usdt, price=None):
        """Places an order on Binance Futures."""
        try:
            # Set leverage first for the symbol
            if MODE == 'live':
                await self.binance_client.set_leverage(LEVERAGE, symbol)
                logger.info(f"Set {LEVERAGE}x leverage for {symbol}")

            # FIXED: Tambahkan pengecekan symbol di markets
            if symbol not in self.binance_client.markets:
                logger.error(f"Symbol {symbol} not found in markets during order placement. Skipping.")
                await self.telegram_notifier.send_message(f"🚨 *ERROR:* Symbol `{symbol}` not found in markets. Cannot place order.")
                return None
            market = self.binance_client.markets[symbol] # FIXED: Akses via .markets
            
            # Use current mark price for calculation of amount
            ticker = await self.binance_client.fetch_ticker(symbol)
            current_mark_price = ticker['markPrice']

            available_balance = await self.get_balance()
            if available_balance < MIN_NOTIONAL_USDT / LEVERAGE:
                logger.error(f"Insufficient balance to open position for {symbol}. Available: {available_balance:.2f} USDT, Required: {MIN_NOTIONAL_USDT / LEVERAGE:.2f} USDT")
                await self.telegram_notifier.send_message(f"🚨 *ERROR:* Insufficient balance to open position for {symbol}. Available: `{available_balance:.2f} USDT`, Required: `{MIN_NOTIONAL_USDT / LEVERAGE:.2f} USDT`")
                return None

            amount = MIN_NOTIONAL_USDT / current_mark_price # This is the quantity of the base asset
            
            # Ensure amount is rounded to appropriate precision for the symbol
            amount_precision = market['precision']['amount']
            # FIXED: Menggunakan math.trunc untuk presisi jumlah (truncate, bukan round)
            amount = math.trunc(amount * (10**amount_precision)) / (10**amount_precision) 
            
            if amount * current_mark_price < MIN_NOTIONAL_USDT:
                logger.warning(f"Calculated amount {amount} for {symbol} results in notional less than {MIN_NOTIONAL_USDT}. Adjusting to minimum.")
                amount = MIN_NOTIONAL_USDT / current_mark_price # Re-calculate to ensure it meets notional min
                amount = math.trunc(amount * (10**amount_precision)) / (10**amount_precision) # FIXED: Truncate again

            if amount == 0:
                logger.error(f"Calculated amount for {symbol} is zero. Skipping order.")
                return None

            params = {'positionSide': 'BOTH'} # For one-way mode, or specific for hedge mode

            if MODE == 'paper':
                logger.info(f"[PAPER TRADE] Placing {side.upper()} {amount} {symbol} at {current_mark_price} (Notional: {amount * current_mark_price:.2f} USDT, Margin: {(amount * current_mark_price) / LEVERAGE:.2f} USDT)")
                # Simulate order response
                return {
                    'info': {'status': 'FILLED'},
                    'id': 'sim_order_' + str(int(time.time())),
                    'datetime': datetime.now().isoformat(),
                    'symbol': symbol,
                    'type': type,
                    'side': side,
                    'price': current_mark_price,
                    'amount': amount,
                    'cost': amount * current_mark_price, # Notional cost
                    'filled': amount,
                    'remaining': 0
                }
            else:
                order = await self.binance_client.create_order(
                    symbol=symbol,
                    type=type, # 'MARKET' or 'LIMIT'
                    side=side, # 'buy' or 'sell'
                    amount=amount, # Base asset amount
                    price=price, # Not required for MARKET orders
                    params=params
                )
                logger.info(f"Placed {side.upper()} order for {amount} {symbol}: {order['id']}")
                return order
        except ccxt.NetworkError as e:
            logger.error(f"Network error placing order for {symbol}: {e}. Retrying...")
            await asyncio.sleep(5) # Wait before retry
            return await self.place_order(symbol, type, side, amount_usdt, price) # Recursive retry
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error placing order for {symbol}: {e}")
            await self.telegram_notifier.send_message(f"🚨 *ERROR:* Exchange error placing order for `{symbol}`: `{e}`")
            return None
        except Exception as e:
            logger.error(f"Unexpected error placing order for {symbol}: {e}")
            await self.telegram_notifier.send_message(f"🚨 *ERROR:* Unexpected error placing order for `{symbol}`: `{e}`")
            return None

    async def close_position(self, symbol, side, amount, entry_price, status_reason):
        """Closes an open position."""
        global open_positions
        try:
            # Determine opposite side to close
            close_side = 'sell' if side == 'long' else 'buy'
            
            if MODE == 'paper':
                logger.info(f"[PAPER TRADE] Closing {symbol} {side.upper()} position of {amount} at current price for {status_reason}")
                current_price = (await self.binance_client.fetch_ticker(symbol))['last']
                pnl_usdt = 0
                if side == 'long':
                    pnl_usdt = (current_price - entry_price) * amount
                elif side == 'short':
                    pnl_usdt = (entry_price - current_price) * amount
                
                pnl_percent = (pnl_usdt / ((entry_price * amount) / LEVERAGE)) * 100 if ((entry_price * amount) / LEVERAGE) != 0 else 0
                self.current_capital += pnl_usdt
                
                log_trade({
                    'symbol': symbol,
                    'direction': side,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'amount': amount,
                    'leverage': LEVERAGE,
                    'pnl_usdt': pnl_usdt,
                    'pnl_percent': pnl_percent,
                    'status': status_reason,
                    'initial_capital': INITIAL_CAPITAL,
                    'current_capital': self.current_capital
                })
                await self.telegram_notifier.send_trade_update({
                    'symbol': symbol,
                    'direction': side,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_usdt': pnl_usdt,
                    'pnl_percent': pnl_percent,
                    'status': status_reason,
                    'current_capital': self.current_capital
                })
                if symbol in open_positions:
                    del open_positions[symbol] # Remove from tracking
                logger.info(f"Closed {symbol} {side.upper()} position. PnL: {pnl_usdt:.2f} USDT. Current Capital: {self.current_capital:.2f}")
                return True
            else:
                # Use create_order with reduceOnly=True to close a position
                order = await self.binance_client.create_order(
                    symbol=symbol,
                    type='MARKET',
                    side=close_side,
                    amount=amount,
                    params={'reduceOnly': True, 'positionSide': 'BOTH'}
                )
                logger.info(f"Closing {symbol} {side.upper()} position with order ID: {order['id']}")
                # After closing, fetch updated positions to update PnL and current capital
                await asyncio.sleep(1) # Give exchange time to process
                await self.get_open_positions() # Refresh internal state

                # Calculate PnL for logging and notification
                # Need to fetch closed order details for exact PnL, but for simplicity, estimate from ticker
                # A more robust solution would be to track orders and listen for fills or fetch individual orders.
                
                # For now, let's assume successful closure and log.
                pnl_usdt = open_positions[symbol]['profit_usdt'] if symbol in open_positions else 0
                if symbol in open_positions: # If position is still there, something went wrong, or partially closed
                    logger.warning(f"Position for {symbol} still detected after close attempt.")
                    return False
                
                # Get current price for exit_price in log
                current_price = (await self.binance_client.fetch_ticker(symbol))['last']

                # Update current capital based on PnL from the actual exchange (if available) or estimated
                # For now, let's update capital based on the previous PnL recorded if not explicitly given by the close order.
                self.current_capital += pnl_usdt # This is an estimate if PnL not from `order`

                log_trade({
                    'symbol': symbol,
                    'direction': side,
                    'entry_price': entry_price,
                    'exit_price': current_price, # Use current_price as exit for logging
                    'amount': amount,
                    'leverage': LEVERAGE,
                    'pnl_usdt': pnl_usdt,
                    'pnl_percent': (pnl_usdt / ((entry_price * amount) / LEVERAGE)) * 100 if ((entry_price * amount) / LEVERAGE) != 0 else 0,
                    'status': status_reason,
                    'initial_capital': INITIAL_CAPITAL,
                    'current_capital': self.current_capital
                })
                await self.telegram_notifier.send_trade_update({
                    'symbol': symbol,
                    'direction': side,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_usdt': pnl_usdt,
                    'pnl_percent': (pnl_usdt / ((entry_price * amount) / LEVERAGE)) * 100 if ((entry_price * amount) / LEVERAGE) != 0 else 0,
                    'status': status_reason,
                    'current_capital': self.current_capital
                })
                if symbol in open_positions: # Final check
                    del open_positions[symbol]
                logger.info(f"Closed {symbol} {side.upper()} position. PnL: {pnl_usdt:.2f} USDT. Current Capital: {self.current_capital:.2f}")
                return True

        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error closing position for {symbol}: {e}")
            await self.telegram_notifier.send_message(f"🚨 *ERROR:* Exchange error closing position for `{symbol}`: `{e}`")
            return False
        except Exception as e:
            logger.error(f"Unexpected error closing position for {symbol}: {e}")
            await self.telegram_notifier.send_message(f"🚨 *ERROR:* Unexpected error closing position for `{symbol}`: `{e}`")
            return False


    async def monitor_positions(self):
        """Monitors all open positions for SL/TP and applies trailing TP."""
        global open_positions
        if not open_positions:
            return

        symbols_to_check = list(open_positions.keys())
        for symbol in symbols_to_check:
            position = open_positions.get(symbol)
            if not position:
                continue

            try:
                ticker = await self.binance_client.fetch_ticker(symbol)
                current_price = ticker['last']

                # Calculate PnL for logging and display, update in `open_positions`
                pnl_usdt = 0
                if position['side'] == 'long':
                    pnl_usdt = (current_price - position['entry_price']) * position['amount']
                elif position['side'] == 'short':
                    pnl_usdt = (position['entry_price'] - current_price) * position['amount']
                open_positions[symbol]['profit_usdt'] = pnl_usdt
                
                # Check Stop Loss
                if position['side'] == 'long' and current_price <= position['sl_price']:
                    logger.warning(f"SL hit for {symbol} LONG position. Closing.")
                    await self.close_position(symbol, 'long', position['amount'], position['entry_price'], 'CLOSED_SL')
                    continue # Move to next position after closing

                if position['side'] == 'short' and current_price >= position['sl_price']:
                    logger.warning(f"SL hit for {symbol} SHORT position. Closing.")
                    await self.close_position(symbol, 'short', position['amount'], position['entry_price'], 'CLOSED_SL')
                    continue # Move to next position after closing

                # Check Take Profit
                if position['side'] == 'long' and current_price >= position['tp_price']:
                    logger.info(f"TP hit for {symbol} LONG position. Closing.")
                    await self.close_position(symbol, 'long', position['amount'], position['entry_price'], 'CLOSED_TP')
                    continue

                if position['side'] == 'short' and current_price <= position['tp_price']:
                    logger.info(f"TP hit for {symbol} SHORT position. Closing.")
                    await self.close_position(symbol, 'short', position['amount'], position['entry_price'], 'CLOSED_TP')
                    continue

                # Apply Trailing TP
                if self.risk_manager.should_activate_trailing_tp(current_price, position):
                    if not position['trailing_active']:
                        logger.info(f"Activating trailing TP for {symbol} position.")
                        open_positions[symbol]['trailing_active'] = True
                    
                    new_sl_price = self.risk_manager.update_trailing_stop(current_price, position)
                    if new_sl_price != position['sl_price']:
                        logger.info(f"Updating trailing SL for {symbol}: {position['sl_price']:.4f} -> {new_sl_price:.4f}")
                        open_positions[symbol]['sl_price'] = new_sl_price
                        # In a real system, you might update SL order on exchange if supported
                        # For now, we only update internally and close when price hits this internal SL.

            except Exception as e:
                logger.error(f"Error monitoring position for {symbol}: {e}")


    async def execute_trade(self, signal):
        """Executes a trade based on the generated signal."""
        global open_positions

        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal['entry_price']

        # Anti-overtrade logic: Max 1 position per pair per signal
        if symbol in open_positions and MAX_POSITIONS_PER_PAIR == 1:
            logger.info(f"Already an open position for {symbol}. Skipping new trade.")
            return

        # Check last trade time for this symbol to prevent too frequent trades on the same signal
        if symbol in self.last_trade_time and (datetime.now() - self.last_trade_time[symbol]).total_seconds() < 300: # 5 minutes cooldown
            logger.info(f"Cooldown active for {symbol}. Skipping new trade.")
            return

        # Calculate initial SL and TP
        sl_price, tp_price = self.risk_manager.calculate_sl_tp(entry_price, direction, symbol)
        
        # FIXED: Periksa jika SL/TP tidak berhasil dihitung (misal karena symbol tidak ditemukan)
        if sl_price is None or tp_price is None:
            logger.error(f"Could not calculate SL/TP for {symbol}. Skipping trade execution.")
            await self.telegram_notifier.send_message(f"🚨 *ERROR:* Could not calculate SL/TP for `{symbol}`. Skipping trade.")
            return

        logger.info(f"Executing {direction.upper()} trade for {symbol} at {entry_price:.4f}. SL: {sl_price:.4f}, TP: {tp_price:.4f}")
        await self.telegram_notifier.send_signal({
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'confidence': signal['confidence']
        })

        order_side = 'buy' if direction == 'long' else 'sell'
        
        # Place the market order to open position
        # We need to pass the USDT amount for the trade.
        # This bot is aggressive, so it will attempt to use a fixed notional value per trade
        # which will be derived from MIN_NOTIONAL_USDT.
        order = await self.place_order(symbol, 'MARKET', order_side, MIN_NOTIONAL_USDT)
        
        if order and order['info']['status'] == 'FILLED':
            filled_price = float(order['price']) # Or order['avgPrice'] for filled orders
            filled_amount = float(order['filled'])
            
            # Recalculate SL/TP based on actual filled price if it differs
            sl_price_actual, tp_price_actual = self.risk_manager.calculate_sl_tp(filled_price, direction, symbol)
            
            # FIXED: Periksa kembali jika SL/TP tidak berhasil dihitung setelah pengisian (misal karena symbol tidak ditemukan)
            if sl_price_actual is None or tp_price_actual is None:
                logger.error(f"Could not recalculate actual SL/TP for {symbol}. Using initial calculated values.")
                sl_price_actual = sl_price
                tp_price_actual = tp_price

            # Store the open position
            open_positions[symbol] = {
                'symbol': symbol,
                'entry_price': filled_price,
                'amount': filled_amount,
                'side': direction,
                'sl_price': sl_price_actual,
                'tp_price': tp_price_actual,
                'trailing_active': False,
                'profit_usdt': 0,
                'created_at': datetime.now().isoformat()
            }
            self.last_trade_time[symbol] = datetime.now()
            logger.info(f"Position opened for {symbol}: {direction.upper()} {filled_amount} at {filled_price:.4f}.")
        else:
            logger.error(f"Failed to open position for {symbol}. Order response: {order}")

    async def trade_loop(self):
        """Main trading loop that runs periodically."""
        logger.info("Running trade loop...")
        await self.get_open_positions() # Always update positions from exchange first

        # Monitor and manage existing positions
        await self.monitor_positions()

        # Iterate through pairs to find new trading opportunities
        for pair in SUPPORTED_PAIRS:
            if pair in open_positions and MAX_POSITIONS_PER_PAIR == 1:
                logger.info(f"Skipping new signal generation for {pair} as position is already open.")
                continue

            try:
                signal = await self.signal_engine.generate_signal(pair)
                if signal:
                    await self.execute_trade(signal)
                else:
                    logger.debug(f"No signal for {pair}")
            except Exception as e:
                logger.error(f"Error processing {pair} in trade loop: {e}")
                await self.telegram_notifier.send_message(f"🚨 *ERROR:* Error in trade loop for `{pair}`: `{e}`")
            
            await asyncio.sleep(0.5) # Small delay between pairs to avoid rate limits

# --- FastAPI App Initialization ---
app = FastAPI(title="AI Crypto Futures Trading Bot",
              description="High-frequency, aggressive crypto futures trading bot powered by AI heuristics.")

telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
risk_manager = RiskManager(LEVERAGE, FIXED_SL_PERCENT, DYNAMIC_TP_MIN_PERCENT, DYNAMIC_TP_MAX_PERCENT, TRAIL_TP_PERCENT)

@app.on_event("startup")
async def startup_event():
    """Initializes client, scheduler, and bot components on startup."""
    global binance_client, scheduler, trading_bot

    init_db() # Initialize SQLite database

    # Initialize Binance client
    try:
        # FIXED: Menggunakan async_binance dari ccxt.async_support
        binance_client = async_binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'options': {
                'defaultType': 'future',
                'createMarketBuyOrderRequiresPrice': False, # For Binance
                'adjustForTimeDifference': True, # Synchronize time with Binance
                'warnOnFetchOpenOrdersWithoutSymbol': False,
                'recvWindow': 60000, # Max receive window
            },
            'enableRateLimit': True, # Enable ccxt's built-in rate limiter
            'urls': {
                'api': {
                    'public': 'https://fapi.binance.com/fapi/v1',
                    'private': 'https://fapi.binance.com/fapi/v1',
                },
                'test': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                },
            },
        })
        # Use testnet if MODE is 'paper'
        if MODE == 'paper':
            binance_client.set_sandbox_mode(True)
            logger.info("Running in PAPER mode (Binance Testnet)")
            await telegram_notifier.send_message("🚀 *Bot Started:* Running in `PAPER MODE` (Binance Testnet)")
        else:
            logger.info("Running in LIVE mode (Binance Futures)")
            await telegram_notifier.send_message("🚀 *Bot Started:* Running in `LIVE MODE` (Binance Futures)")
        
        # FIXED: await binance_client.load_markets() karena sekarang menggunakan async_support
        await binance_client.load_markets()
        logger.info("Binance client initialized and markets loaded.")

    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {e}")
        await telegram_notifier.send_message(f"🚨 *FATAL ERROR:* Failed to initialize Binance client: `{e}`. Bot will not run.")
        return # Stop startup if client fails

    signal_engine = SignalEngine(binance_client)
    trading_bot = TradingBot(binance_client, signal_engine, risk_manager, telegram_notifier)

    # Initialize and start scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(trading_bot.trade_loop, 'interval', seconds=TRADE_INTERVAL_SECONDS)
    scheduler.start()
    logger.info(f"Scheduler started with trade loop interval: {TRADE_INTERVAL_SECONDS} seconds.")
    await telegram_notifier.send_message(f"🔄 *Scheduler Activated:* Trading loop running every `{TRADE_INTERVAL_SECONDS}` seconds.")

@app.on_event("shutdown")
async def shutdown_event():
    """Stops the scheduler and closes database connection on shutdown."""
    global scheduler, db_connection
    if scheduler:
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
    if db_connection:
        db_connection.close()
        logger.info("Database connection closed.")
    await telegram_notifier.send_message("🛑 *Bot Stopped:* Trading bot is shutting down.")

@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {"status": "ok", "message": "AI Crypto Trading Bot is running!"}

@app.get("/stats")
async def get_stats():
    """Returns trading statistics."""
    try:
        trades = get_trade_history_from_db()
        
        total_trades = len(trades)
        wins = sum(1 for t in trades if t[9].startswith('CLOSED_TP')) # t[9] is status
        losses = sum(1 for t in trades if t[9].startswith('CLOSED_SL'))
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        total_pnl_usdt = sum(t[8] for t in trades) # t[8] is pnl_usdt
        
        current_capital_from_db = 0
        if trades:
            # Get the latest current_capital from the last trade entry
            current_capital_from_db = trades[-1][11] # t[11] is current_capital
        else:
            current_capital_from_db = INITIAL_CAPITAL
        
        # Calculate daily growth
        daily_growth = {}
        trade_by_day = {}
        for trade in trades:
            trade_date = datetime.fromisoformat(trade[1]).strftime('%Y-%m-%d') # t[1] is timestamp
            if trade_date not in trade_by_day:
                trade_by_day[trade_date] = {'pnl': 0, 'wins': 0, 'losses': 0}
            
            trade_by_day[trade_date]['pnl'] += trade[8] # Add PnL
            if trade[9].startswith('CLOSED_TP'):
                trade_by_day[trade_date]['wins'] += 1
            elif trade[9].startswith('CLOSED_SL'):
                trade_by_day[trade_date]['losses'] += 1
        
        sorted_days = sorted(trade_by_day.keys())
        prev_day_capital = INITIAL_CAPITAL
        for day in sorted_days:
            day_pnl = trade_by_day[day]['pnl']
            # Capital at end of day = Capital at start of day + PnL of day
            end_of_day_capital = prev_day_capital + day_pnl
            
            growth_percent = (end_of_day_capital - prev_day_capital) / prev_day_capital * 100 if prev_day_capital != 0 else 0
            
            daily_growth[day] = {
                'pnl_usdt': day_pnl,
                'wins': trade_by_day[day]['wins'],
                'losses': trade_by_day[day]['losses'],
                'win_rate': (trade_by_day[day]['wins'] / (trade_by_day[day]['wins'] + trade_by_day[day]['losses']) * 100) if (trade_by_day[day]['wins'] + trade_by_day[day]['losses']) > 0 else 0,
                'capital_start_day': prev_day_capital,
                'capital_end_day': end_of_day_capital,
                'growth_percent': growth_percent
            }
            prev_day_capital = end_of_day_capital # Update for next day's calculation

        summary = {
            "total_trades": total_trades,
            "total_wins": wins,
            "total_losses": losses,
            "win_rate": f"{win_rate:.2f}%",
            "total_pnl_usdt": f"{total_pnl_usdt:.2f}",
            "initial_capital": f"{INITIAL_CAPITAL:.2f}",
            "current_capital": f"{current_capital_from_db:.2f}",
            "overall_growth_percent": f"{(current_capital_from_db - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:.2f}%" if INITIAL_CAPITAL != 0 else "0.00%",
            "daily_summary": daily_growth
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating statistics: {e}")

# This block allows you to run the FastAPI application directly
if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    except Exception as e:
        logger.error(f"Uvicorn server failed to start: {e}")

