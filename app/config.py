import yaml
import os
from dotenv import load_dotenv

# Muat variabel lingkungan dari .env jika ada (untuk pengembangan lokal)
load_dotenv()

# Path ke file konfigurasi YAML
CONFIG_FILE_PATH = "config.yaml"

class Config:
    def __init__(self):
        self._config = {}
        self.load_config()

    def load_config(self):
        """Memuat konfigurasi dari config.yaml atau variabel lingkungan."""
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            print(f"Peringatan: File '{CONFIG_FILE_PATH}' tidak ditemukan. Mencoba memuat dari variabel lingkungan.")
            # Fallback ke variabel lingkungan jika config.yaml tidak ada
            self._config = {
                "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
                "BINANCE_SECRET_KEY": os.getenv("BINANCE_SECRET_KEY"),
                "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
                "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
                "SYMBOLS": os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(','),
                "TIMEFRAMES": os.getenv("TIMEFRAMES", "5m,15m,1h").split(','),
                "SCAN_INTERVAL_SECONDS": int(os.getenv("SCAN_INTERVAL_SECONDS", 30)),
                "EMA_PERIODS": list(map(int, os.getenv("EMA_PERIODS", "13,34,50,200").split(','))),
                "RSI_PERIOD": int(os.getenv("RSI_PERIOD", 14)),
                "CCI_PERIOD": int(os.getenv("CCI_PERIOD", 14)),
                "MACD_FAST_PERIOD": int(os.getenv("MACD_FAST_PERIOD", 12)),
                "MACD_SLOW_PERIOD": int(os.getenv("MACD_SLOW_PERIOD", 26)),
                "MACD_SIGNAL_PERIOD": int(os.getenv("MACD_SIGNAL_PERIOD", 9)),
                "SSL_HYBRID_PERIOD": int(os.getenv("SSL_HYBRID_PERIOD", 20)),
                "DEFAULT_LEVERAGE": int(os.getenv("DEFAULT_LEVERAGE", 10)),
                "TP_RATIO": float(os.getenv("TP_RATIO", 1.5)),
                "SL_RATIO": float(os.getenv("SL_RATIO", 0.5)),
                "RISK_PER_TRADE_PERCENT": float(os.getenv("RISK_PER_TRADE_PERCENT", 1.0)),
                "AI_MODEL_PATH": os.getenv("AI_MODEL_PATH", "model/ai_model.pkl"),
                "AI_THRESHOLD_STRONG_BUY": float(os.getenv("AI_THRESHOLD_STRONG_BUY", 0.85)),
                "AI_THRESHOLD_STRONG_SELL": float(os.getenv("AI_THRESHOLD_STRONG_SELL", 0.15)),
                "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///data/logs.db")
            }

        # Pastikan tipe data yang benar
        self._config['EMA_PERIODS'] = [int(p) for p in self._config['EMA_PERIODS']]
        self._config['TIMEFRAMES'] = [str(tf) for tf in self._config['TIMEFRAMES']]
        self._config['SYMBOLS'] = [str(s) for s in self._config['SYMBOLS']]

    def get(self, key, default=None):
        """Mendapatkan nilai konfigurasi."""
        return self._config.get(key, default)

# Inisialisasi objek konfigurasi
settings = Config()

if __name__ == "__main__":
    # Contoh penggunaan
    print("Pengaturan yang dimuat:")
    print(f"Binance API Key: {settings.get('BINANCE_API_KEY')[:5]}...")
    print(f"Telegram Chat ID: {settings.get('TELEGRAM_CHAT_ID')}")
    print(f"Symbols: {settings.get('SYMBOLS')}")
    print(f"Scan Interval: {settings.get('SCAN_INTERVAL_SECONDS')} detik")
    print(f"EMA Periods: {settings.get('EMA_PERIODS')}")
    print(f"AI Model Path: {settings.get('AI_MODEL_PATH')}")
