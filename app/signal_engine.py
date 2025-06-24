import ccxt.pro as ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
from app.ai_filter import AIFilter
from app.telegram_bot import TelegramBot # Tetap diimpor karena AI membutuhkan ini untuk inisialisasi

# Konfigurasi Database (SQLite)
Base = declarative_base()

class SignalLog(Base):
    """Model database untuk mencatat sinyal yang dikirim."""
    __tablename__ = 'signal_logs'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String)
    timeframe = Column(String)
    signal_type = Column(String)
    entry = Column(Float)
    tp1 = Column(Float)
    tp2 = Column(Float)
    sl = Column(Float)
    ai_score = Column(Float)
    confidence_emoji = Column(String)

    def __repr__(self):
        return (f"<SignalLog(pair='{self.pair}', tf='{self.timeframe}', "
                f"signal='{self.signal_type}', entry={self.entry})>")

class SignalEngine:
    def __init__(self):
        """
        Menginisialisasi SignalEngine dengan pengaturan, AI filter,
        dan koneksi bursa. TelegramBot tidak lagi diinisialisasi di sini
        karena pengiriman sinyal akan dikelola oleh main.py.
        """
        self.settings = settings
        self.ai_filter = AIFilter() # AIFilter masih butuh, jadi tetap di sini

        # Inisialisasi koneksi Binance Futures
        self.exchange = ccxt.binance({
            'apiKey': self.settings.get("BINANCE_API_KEY"),
            'secret': self.settings.get("BINANCE_SECRET_KEY"),
            'options': {
                'defaultType': 'future',
                'warnOnFetchOHLCVPayloadEmpty': False # Suppress warning for empty payload
            },
            'enableRateLimit': True, # Untuk menghindari pembatasan rate
        })

        self.last_scan_times = {} # Untuk melacak kapan terakhir kali setiap pair di-scan

    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """
        Mengambil data OHLCV historis dari Binance.
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                # print(f"Peringatan: Tidak ada data OHLCV yang diambil untuk {symbol} ({timeframe}).")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt.NetworkError as e:
            print(f"Kesalahan jaringan saat mengambil OHLCV untuk {symbol} ({timeframe}): {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            print(f"Kesalahan bursa saat mengambil OHLCV untuk {symbol} ({timeframe}): {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Kesalahan tak terduga saat mengambil OHLCV untuk {symbol} ({timeframe}): {e}")
            return pd.DataFrame()

    def _calculate_ema(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Menghitung Exponential Moving Averages (EMA) untuk beberapa periode."""
        for period in periods:
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Menghitung Relative Strength Index (RSI)."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
        """Menghitung Moving Average Convergence Divergence (MACD)."""
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        return df

    def _calculate_cci(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Menghitung Commodity Channel Index (CCI)."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).sum() / len(x), raw=True)
        df['cci'] = (tp - ma) / (0.015 * md)
        return df

    def _calculate_ssl_hybrid(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Menghitung SSL Hybrid Indicator.
        SSL Hybrid seringkali merupakan kombinasi EMA (HL2) dengan kriteria warna.
        Sini kita akan menghitung dua EMA dari HL2 dan membandingkannya.
        """
        hl2 = (df['high'] + df['low']) / 2
        ema_high = hl2.ewm(span=period, adjust=False).mean()
        ema_low = hl2.ewm(span=period, adjust=False).mean() # Bisa juga menggunakan periode berbeda atau offset

        df['ssl_up'] = np.where(df['close'] > ema_high, ema_high, np.nan)
        df['ssl_down'] = np.where(df['close'] < ema_low, ema_low, np.nan)
        
        # Untuk kesederhanaan, kita bisa menggunakan crossover dua EMA dari HL2
        df['ssl_ma'] = hl2.ewm(span=period, adjust=False).mean()
        df['ssl_signal'] = hl2.ewm(span=period*2, adjust=False).mean() # Contoh signal line

        df['ssl_direction'] = 0 # 1: Up, -1: Down, 0: Sideways/Neutral
        df.loc[df['ssl_ma'] > df['ssl_signal'], 'ssl_direction'] = 1
        df.loc[df['ssl_ma'] < df['ssl_signal'], 'ssl_direction'] = -1

        return df

    def _calculate_vwap_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghitung VWAP (Volume Weighted Average Price) dan Delta Volume (sederhana).
        Delta Volume akan disederhanakan karena data tick tidak tersedia dari ccxt fetch_ohlcv.
        Kita akan mengestimasi volume delta berdasarkan arah pergerakan harga.
        """
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Estimasi Delta Volume: volume positif jika close > open, negatif jika close < open
        df['delta_volume'] = np.where(df['close'] > df['open'], df['volume'],
                                      np.where(df['close'] < df['open'], -df['volume'], 0))
        
        # Contoh indikator Imbalance (cumulative delta)
        df['cumulative_delta'] = df['delta_volume'].cumsum()

        return df

    def _detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mendeteksi Fair Value Gap (FVG) zones.
        FVG terjadi ketika ada gap antara high candle 1 dan low candle 3 (untuk bearish FVG)
        atau low candle 1 dan high candle 3 (untuk bullish FVG).
        """
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False

        if len(df) >= 3:
            # Bearish FVG: High candle[0] > Low candle[2] (gap ke bawah)
            # Candle 0, 1, 2 = prev_prev, prev, current
            # Kita lihat dari sudut pandang candle saat ini (indeks terakhir)
            for i in range(2, len(df)):
                # Bullish FVG (gap ke atas): Low[i-2] > High[i] -> price moved down, then gap up leaving unfilled space
                # Example: current candle's low is higher than two candles before's high
                if df['low'].iloc[i] > df['high'].iloc[i-2]:
                    # Also need to make sure the middle candle (i-1) is part of the gap
                    # Check if the body of middle candle is not filling the gap
                    if df['high'].iloc[i-1] < df['low'].iloc[i] and df['low'].iloc[i-1] > df['high'].iloc[i-2]:
                         df.loc[df.index[i], 'fvg_bullish'] = True
                
                # Bearish FVG (gap ke bawah): High[i-2] < Low[i] -> price moved up, then gap down leaving unfilled space
                # Example: current candle's high is lower than two candles before's low
                if df['high'].iloc[i] < df['low'].iloc[i-2]:
                    # Check if the body of middle candle is not filling the gap
                    if df['low'].iloc[i-1] > df['high'].iloc[i] and df['high'].iloc[i-1] < df['low'].iloc[i-2]:
                        df.loc[df.index[i], 'fvg_bearish'] = True
        return df

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung semua indikator teknis yang diminta."""
        df = self._calculate_ema(df.copy(), self.settings.get('EMA_PERIODS'))
        df = self._calculate_rsi(df.copy(), self.settings.get('RSI_PERIOD'))
        df = self._calculate_macd(df.copy(), self.settings.get('MACD_FAST_PERIOD'),
                                  self.settings.get('MACD_SLOW_PERIOD'), self.settings.get('MACD_SIGNAL_PERIOD'))
        df = self._calculate_cci(df.copy(), self.settings.get('CCI_PERIOD'))
        df = self._calculate_ssl_hybrid(df.copy(), self.settings.get('SSL_HYBRID_PERIOD'))
        df = self._calculate_vwap_delta(df.copy())
        df = self._detect_fvg(df.copy())
        return df.dropna() # Hapus baris dengan nilai NaN yang dihasilkan oleh perhitungan indikator

    def _apply_trading_strategies(self, df: pd.DataFrame, timeframe_1h_df: pd.DataFrame) -> tuple[str | None, str | None]:
        """
        Menerapkan kombinasi strategi trading dan mengidentifikasi sinyal BUY/SELL.
        Mengembalikan ('BUY', 'STRONG') atau ('SELL', 'STRONG') atau (None, None).
        Ini adalah logika inti yang disederhanakan.
        """
        if df.empty:
            return None, None

        latest = df.iloc[-1]
        
        # Multi-timeframe confirmation (TF 1h untuk trend filter)
        is_uptrend_1h = False
        is_downtrend_1h = False
        if not timeframe_1h_df.empty:
            latest_1h = timeframe_1h_df.iloc[-1]
            if latest_1h['close'] > latest_1h['ema200'] and latest_1h['ema13'] > latest_1h['ema34']:
                is_uptrend_1h = True
            elif latest_1h['close'] < latest_1h['ema200'] and latest_1h['ema13'] < latest_1h['ema34']:
                is_downtrend_1h = True

        # === BUY Signal Logic ===
        buy_conditions = []

        # EMA Crossover for immediate trend (e.g., 13 EMA crosses above 34 EMA)
        if latest['ema13'] > latest['ema34'] and df['ema13'].iloc[-2] < df['ema34'].iloc[-2]:
            buy_conditions.append("EMA_CROSS_UP")

        # SSL Hybrid - Bullish confirmation
        if latest['ssl_direction'] == 1:
            buy_conditions.append("SSL_BULLISH")

        # RSI Divergence (simplified: RSI oversold and then moves up while price consolidates/moves slightly down)
        # Atau hanya RSI di area oversold
        if latest['rsi'] < 30 and df['rsi'].iloc[-2] < latest['rsi']: # RSI naik dari oversold
            buy_conditions.append("RSI_OVERSOLD_REVERSE")

        # MACD Histogram Reversal (from negative to positive territory or turning up from low)
        if latest['macd_histogram'] > 0 and df['macd_histogram'].iloc[-2] < 0:
            buy_conditions.append("MACD_HIST_CROSS_UP_ZERO")
        elif latest['macd_histogram'] > df['macd_histogram'].iloc[-2] and latest['macd_histogram'] < 0: # MACD hist turning up from below zero
            buy_conditions.append("MACD_HIST_TURN_UP")

        # CCI from oversold
        if latest['cci'] < -100 and latest['cci'] > df['cci'].iloc[-2]:
            buy_conditions.append("CCI_OVERSOLD_REVERSE")

        # VWAP / Delta Volume Analyzer (positive cumulative delta and price above VWAP)
        if latest['close'] > latest['vwap'] and latest['delta_volume'] > 0:
            buy_conditions.append("VWAP_POSITIVE_FLOW")

        # FVG Zone (price enters bullish FVG and reverses)
        if latest['fvg_bullish'] and latest['close'] > latest['open']: # Bullish FVG detected and current candle is bullish
            buy_conditions.append("FVG_BULLISH_REACTION")

        # Wyckoff / Smart Liquidity Grab / Order Block (simplified detection based on price action + volume)
        # Ini sangat disederhanakan dan harus dikembangkan lebih lanjut untuk akurasi.
        # Misal: Volume tinggi pada candle reversal bullish setelah penurunan
        if latest['volume'] > df['volume'].rolling(window=10).mean().iloc[-1] * 1.5 and \
           latest['close'] > latest['open'] and latest['low'] < df['low'].iloc[-2] and \
           latest['close'] > df['close'].iloc[-2]:
            buy_conditions.append("VOLUME_REVERSAL_CANDLE") # Sinyal pembalikan volume tinggi

        # Kombinasi kriteria untuk BUY
        if is_uptrend_1h and \
           ("EMA_CROSS_UP" in buy_conditions or "SSL_BULLISH" in buy_conditions) and \
           ("RSI_OVERSOLD_REVERSE" in buy_conditions or "MACD_HIST_TURN_UP" in buy_conditions) and \
           ("VWAP_POSITIVE_FLOW" in buy_conditions or "VOLUME_REVERSAL_CANDLE" in buy_conditions):
            return "BUY", "STRONG"

        # === SELL Signal Logic ===
        sell_conditions = []

        # EMA Crossover for immediate trend (e.g., 13 EMA crosses below 34 EMA)
        if latest['ema13'] < latest['ema34'] and df['ema13'].iloc[-2] > df['ema34'].iloc[-2]:
            sell_conditions.append("EMA_CROSS_DOWN")

        # SSL Hybrid - Bearish confirmation
        if latest['ssl_direction'] == -1:
            sell_conditions.append("SSL_BEARISH")

        # RSI Divergence (simplified: RSI overbought and then moves down while price consolidates/moves slightly up)
        # Atau hanya RSI di area overbought
        if latest['rsi'] > 70 and df['rsi'].iloc[-2] > latest['rsi']: # RSI turun dari overbought
            sell_conditions.append("RSI_OVERBOUGHT_REVERSE")

        # MACD Histogram Reversal (from positive to negative territory or turning down from high)
        if latest['macd_histogram'] < 0 and df['macd_histogram'].iloc[-2] > 0:
            sell_conditions.append("MACD_HIST_CROSS_DOWN_ZERO")
        elif latest['macd_histogram'] < df['macd_histogram'].iloc[-2] and latest['macd_histogram'] > 0: # MACD hist turning down from above zero
            sell_conditions.append("MACD_HIST_TURN_DOWN")

        # CCI from overbought
        if latest['cci'] > 100 and latest['cci'] < df['cci'].iloc[-2]:
            sell_conditions.append("CCI_OVERBOUGHT_REVERSE")

        # VWAP / Delta Volume Analyzer (negative cumulative delta and price below VWAP)
        if latest['close'] < latest['vwap'] and latest['delta_volume'] < 0:
            sell_conditions.append("VWAP_NEGATIVE_FLOW")

        # FVG Zone (price enters bearish FVG and reverses)
        if latest['fvg_bearish'] and latest['close'] < latest['open']: # Bearish FVG detected and current candle is bearish
            sell_conditions.append("FVG_BEARISH_REACTION")

        # Wyckoff / Smart Liquidity Grab / Order Block (simplified detection)
        # Misal: Volume tinggi pada candle reversal bearish setelah kenaikan
        if latest['volume'] > df['volume'].rolling(window=10).mean().iloc[-1] * 1.5 and \
           latest['close'] < latest['open'] and latest['high'] > df['high'].iloc[-2] and \
           latest['close'] < df['close'].iloc[-2]:
            sell_conditions.append("VOLUME_REVERSAL_CANDLE") # Sinyal pembalikan volume tinggi

        # Kombinasi kriteria untuk SELL
        if is_downtrend_1h and \
           ("EMA_CROSS_DOWN" in sell_conditions or "SSL_BEARISH" in sell_conditions) and \
           ("RSI_OVERBOUGHT_REVERSE" in sell_conditions or "MACD_HIST_TURN_DOWN" in sell_conditions) and \
           ("VWAP_NEGATIVE_FLOW" in sell_conditions or "VOLUME_REVERSAL_CANDLE" in sell_conditions):
            return "SELL", "STRONG"

        return None, None # Tidak ada sinyal

    def _calculate_tp_sl(self, entry_price: float, signal_type: str) -> tuple[float, float, float]:
        """
        Menghitung Take Profit (TP) dan Stop Loss (SL) berdasarkan entry price,
        TP/SL ratio dari konfigurasi, dan tipe sinyal.
        """
        tp_ratio = self.settings.get('TP_RATIO', 1.5)
        sl_ratio = self.settings.get('SL_RATIO', 0.5) # Ini adalah persentase dari entry price, bukan rasio SL.
                                                       # Jika ingin rasio R:R, ini perlu dihitung ulang.
                                                       # Misal, SL = 0.5% dari harga entry.
        
        # Misalnya, SL_PERCENT adalah persentase dari harga entry (0.5 berarti 0.5%)
        # Untuk R:R, kita bisa asumsikan SL = X%, maka TP = X% * TP_RATIO
        sl_percent = sl_ratio / 100.0 # Ubah 0.5 menjadi 0.005
        
        if signal_type == "BUY":
            sl_price = entry_price * (1 - sl_percent)
            tp_1_price = entry_price + (entry_price - sl_price) * tp_ratio
            tp_2_price = entry_price + (entry_price - sl_price) * (tp_ratio * 2) # TP2 adalah 2x dari TP1 target
        elif signal_type == "SELL":
            sl_price = entry_price * (1 + sl_percent)
            tp_1_price = entry_price - (sl_price - entry_price) * tp_ratio
            tp_2_price = entry_price - (sl_price - entry_price) * (tp_ratio * 2) # TP2 adalah 2x dari TP1 target
        else:
            return 0.0, 0.0, 0.0 # Default jika tidak ada sinyal

        return tp_1_price, tp_2_price, sl_price

    async def scan_pairs(self) -> list[dict]:
        """
        Memindai semua pair yang dikonfigurasi pada timeframe yang ditentukan,
        menghitung indikator, menerapkan strategi, dan MENGEMBALIKAN sinyal yang ditemukan.
        """
        symbols = self.settings.get('SYMBOLS')
        timeframes = self.settings.get('TIMEFRAMES')
        found_signals = []
        
        # Ambil data 1h untuk trend filter terlebih dahulu
        timeframe_1h_data = {}
        if '1h' in timeframes:
            for symbol in symbols:
                df_1h = await self._fetch_ohlcv(symbol, '1h', limit=50) # Hanya butuh beberapa candle untuk trend
                if not df_1h.empty:
                    df_1h = self._calculate_ema(df_1h, [13, 34, 200]) # Hanya butuh EMA untuk trend 1h
                    timeframe_1h_data[symbol] = df_1h
                # else:
                #     print(f"Tidak dapat mengambil data 1h untuk {symbol}. Trend filter mungkin tidak akurat.")

        for symbol in symbols:
            current_time = datetime.now()
            # Cek apakah sudah waktunya scan lagi untuk pair ini
            if symbol in self.last_scan_times and \
               (current_time - self.last_scan_times[symbol]).total_seconds() < self.settings.get("SCAN_INTERVAL_SECONDS"):
                # print(f"Melewati {symbol}: terlalu cepat untuk scan lagi.")
                continue

            self.last_scan_times[symbol] = current_time # Perbarui waktu scan terakhir

            print(f"Memindai {symbol}...")
            
            # Ambil data 1h yang sudah dihitung untuk symbol ini
            df_1h_for_trend = timeframe_1h_data.get(symbol, pd.DataFrame())

            for tf in [t for t in timeframes if t != '1h']: # Scan hanya TF 5m dan 15m untuk sinyal
                df = await self._fetch_ohlcv(symbol, tf, limit=200) # Batasi limit untuk performa
                if df.empty:
                    continue

                df_with_indicators = self._calculate_all_indicators(df)
                if df_with_indicators.empty:
                    # print(f"Tidak cukup data setelah menghitung indikator untuk {symbol} ({tf}).")
                    continue

                # Pastikan ada cukup data untuk indikator (misalnya, setelah dropna)
                if len(df_with_indicators) < 2: # Setidaknya 2 candle untuk perbandingan
                    # print(f"Tidak cukup data candle untuk {symbol} ({tf}) setelah kalkulasi indikator.")
                    continue

                signal_type, signal_strength = self._apply_trading_strategies(df_with_indicators, df_1h_for_trend)

                if signal_type:
                    # Ambil candle terbaru untuk harga entry
                    latest_close = df_with_indicators['close'].iloc[-1]
                    entry_price = latest_close # Menggunakan harga close candle terakhir sebagai entry

                    tp1, tp2, sl = self._calculate_tp_sl(entry_price, signal_type)

                    # Dapatkan fitur untuk AI Filter
                    ai_features = self.ai_filter.get_features_from_df(df_with_indicators, df_1h_for_trend)
                    ai_score = self.ai_filter.predict_signal_score(ai_features)

                    confidence_emoji = ""
                    formatted_signal_type = ""

                    if signal_type == "BUY":
                        if ai_score >= self.settings.get("AI_THRESHOLD_STRONG_BUY"):
                            confidence_emoji = "✅"
                            formatted_signal_type = "🟢 BUY STRONG"
                        else:
                            confidence_emoji = "🟡" # Cukup yakin, tapi tidak kuat
                            formatted_signal_type = "🟢 BUY"
                    elif signal_type == "SELL":
                        if ai_score <= self.settings.get("AI_THRESHOLD_STRONG_SELL"):
                            confidence_emoji = "✅"
                            formatted_signal_type = "🔴 SELL STRONG"
                        else:
                            confidence_emoji = "🟡" # Cukup yakin, tapi tidak kuat
                            formatted_signal_type = "🔴 SELL"
                    else:
                        continue # Should not happen if signal_type is set

                    signal_data = {
                        'pair': symbol,
                        'timeframe': tf.upper(),
                        'signal': formatted_signal_type,
                        'entry': entry_price,
                        'tp1': tp1,
                        'tp2': tp2,
                        'sl': sl,
                        'ai_score': ai_score,
                        'confidence_emoji': confidence_emoji
                    }
                    found_signals.append(signal_data)
                # else:
                #     print(f"Tidak ada sinyal yang teridentifikasi untuk {symbol} ({tf}).")
        
        print(f"Pemindaian semua pair selesai pada {datetime.now().strftime('%H:%M:%S')}")
        return found_signals


    async def close(self):
        """Menutup koneksi bursa."""
        await self.exchange.close()
        print("SignalEngine dihentikan. Koneksi bursa ditutup.")

# Database initialization (moved to main.py, but kept here for SignalLog definition)
engine = create_engine(settings.get("DATABASE_URL"))
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
