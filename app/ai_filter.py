import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from app.config import settings

class AIFilter:
    def __init__(self):
        """
        Menginisialisasi AI Filter.
        Mencoba memuat model AI yang sudah ada, atau membuat model dummy jika tidak ada.
        """
        self.model = None
        self.model_path = settings.get("AI_MODEL_PATH")
        self.load_model()

    def load_model(self):
        """Memuat model AI dari file atau membuat model dummy."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"Model AI berhasil dimuat dari {self.model_path}")
            except Exception as e:
                print(f"Gagal memuat model AI dari {self.model_path}: {e}. Membuat model dummy.")
                self._create_dummy_model()
        else:
            print(f"Model AI tidak ditemukan di {self.model_path}. Membuat model dummy.")
            self._create_dummy_model()

    def _create_dummy_model(self):
        """
        Membuat model AI dummy untuk tujuan demonstrasi.
        Model ini akan memprediksi skor berdasarkan aturan sederhana
        dan dapat diganti dengan model yang terlatih secara nyata.
        """
        # Contoh data dummy untuk melatih model Random Forest sederhana
        # Fitur: RSI, Volume (normalisasi), Trend (1=Up, 0=Sideways, -1=Down), CandleType (1=Bullish, 0=Doji, -1=Bearish)
        # Target: Buy (1), Sell (0.1), No Entry (0.5)
        np.random.seed(42)
        data_size = 1000
        dummy_data = {
            'rsi': np.random.uniform(20, 80, data_size),
            'volume_norm': np.random.uniform(0, 1, data_size),
            'trend': np.random.choice([-1, 0, 1], data_size, p=[0.2, 0.3, 0.5]),
            'candle_type': np.random.choice([-1, 0, 1], data_size, p=[0.25, 0.1, 0.65]) # Lebih banyak bullish
        }
        df_dummy = pd.DataFrame(dummy_data)

        # Membuat target dummy berdasarkan beberapa aturan sederhana
        # Misalnya, BUY jika RSI di bawah 40 dan trend naik
        # SELL jika RSI di atas 60 dan trend turun
        # NO ENTRY di antaranya
        def generate_dummy_target(row):
            if row['rsi'] < 40 and row['trend'] == 1 and row['candle_type'] == 1:
                return 1.0 # Strong Buy
            elif row['rsi'] > 60 and row['trend'] == -1 and row['candle_type'] == -1:
                return 0.0 # Strong Sell (skor rendah)
            elif 40 <= row['rsi'] <= 60 and row['trend'] == 0:
                return 0.5 # No Entry (skor tengah)
            elif row['rsi'] < 50 and row['trend'] == 1:
                return 0.9 # Buy
            elif row['rsi'] > 50 and row['trend'] == -1:
                return 0.1 # Sell
            else:
                return np.random.uniform(0.3, 0.7) # Random for mixed cases

        df_dummy['target_score'] = df_dummy.apply(generate_dummy_target, axis=1)

        # Untuk klasifikasi, kita bisa mengategorikan target_score
        # Misalnya, 0.0-0.3 -> SELL (0), 0.3-0.7 -> NO_ENTRY (1), 0.7-1.0 -> BUY (2)
        df_dummy['target_class'] = pd.cut(df_dummy['target_score'], bins=[-0.1, 0.3, 0.7, 1.1], labels=[0, 1, 2])

        X = df_dummy[['rsi', 'volume_norm', 'trend', 'candle_type']]
        y = df_dummy['target_class'] # Menggunakan kelas untuk pelatihan

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Pastikan direktori model ada
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model AI dummy dibuat dan disimpan ke {self.model_path}")

    def get_features_from_df(self, df: pd.DataFrame, timeframe_1h_df: pd.DataFrame) -> dict:
        """
        Mengekstrak fitur dari DataFrame OHLCV untuk AI model.
        Ini adalah contoh fitur, Anda perlu menyesuaikannya dengan kebutuhan model Anda.
        """
        if df.empty:
            return {}

        latest_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] if len(df) > 1 else None

        features = {}

        # RSI
        features['rsi'] = latest_candle['rsi']

        # Volume (normalisasi)
        max_volume = df['volume'].max()
        features['volume_norm'] = latest_candle['volume'] / max_volume if max_volume > 0 else 0

        # Trend (dari TF 1h)
        if not timeframe_1h_df.empty:
            latest_1h_candle = timeframe_1h_df.iloc[-1]
            ema200_1h = latest_1h_candle['ema200']
            if latest_1h_candle['close'] > ema200_1h:
                features['trend'] = 1 # Up
            elif latest_1h_candle['close'] < ema200_1h:
                features['trend'] = -1 # Down
            else:
                features['trend'] = 0 # Sideways
        else:
            features['trend'] = 0 # Default jika data 1h tidak tersedia

        # Candle structure / formation
        if prev_candle is not None:
            if latest_candle['close'] > latest_candle['open'] and \
               latest_candle['close'] > prev_candle['close'] and \
               latest_candle['high'] - latest_candle['close'] < (latest_candle['close'] - latest_candle['open']) * 0.3:
                features['candle_type'] = 1 # Bullish kuat
            elif latest_candle['close'] < latest_candle['open'] and \
                 latest_candle['close'] < prev_candle['close'] and \
                 latest_candle['open'] - latest_candle['close'] < (latest_candle['open'] - latest_candle['close']) * 0.3:
                features['candle_type'] = -1 # Bearish kuat
            elif abs(latest_candle['close'] - latest_candle['open']) < (latest_candle['high'] - latest_candle['low']) * 0.1:
                features['candle_type'] = 0 # Doji/Small body
            else:
                features['candle_type'] = 0.5 if latest_candle['close'] > latest_candle['open'] else -0.5 # Netral
        else:
            features['candle_type'] = 0 # Default jika hanya ada satu candle

        return features

    def predict_signal_score(self, features: dict) -> float:
        """
        Memprediksi skor sinyal (0-1) menggunakan model AI.
        Ini akan mengembalikan probabilitas untuk kelas "BUY" (kelas 2) dari model dummy.
        """
        if not self.model or not features:
            return 0.5 # Skor netral jika model tidak ada atau fitur kosong

        # Pastikan urutan fitur sesuai dengan yang digunakan saat pelatihan
        feature_names = ['rsi', 'volume_norm', 'trend', 'candle_type']
        input_data = [features.get(f, 0) for f in feature_names] # Gunakan 0 sebagai default jika fitur tidak ada

        try:
            # Mengubah input menjadi DataFrame yang sesuai dengan format pelatihan
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Mendapatkan probabilitas untuk setiap kelas
            probabilities = self.model.predict_proba(input_df)[0]
            
            # Asumsi: kelas BUY adalah kelas 2, kelas NO_ENTRY adalah kelas 1, kelas SELL adalah kelas 0
            # Kita akan mengembalikan probabilitas untuk kelas BUY (indeks 2)
            # Atau, jika model memiliki 2 kelas (misal 0=Sell, 1=Buy), sesuaikan indeks
            
            # Jika model dummy memiliki 3 kelas (0, 1, 2)
            if len(self.model.classes_) == 3:
                buy_prob = probabilities[2] # Probabilitas kelas BUY
                sell_prob = probabilities[0] # Probabilitas kelas SELL
                
                # Mengubah probabilitas klasifikasi menjadi skor 0-1
                # Jika BUY prob tinggi, skor mendekati 1. Jika SELL prob tinggi, skor mendekati 0.
                score = (buy_prob - sell_prob + 1) / 2 # Normalisasi ke 0-1
                return float(np.clip(score, 0, 1))
            elif len(self.model.classes_) == 2: # Misalnya, jika model hanya membedakan antara Sell (0) dan Buy (1)
                buy_prob = probabilities[1]
                return float(np.clip(buy_prob, 0, 1))
            else:
                return 0.5 # Fallback

        except Exception as e:
            print(f"Kesalahan saat memprediksi skor AI: {e}. Mengembalikan skor netral.")
            return 0.5

# Contoh penggunaan (untuk pengujian lokal)
if __name__ == "__main__":
    ai_filter = AIFilter()

    # Buat DataFrame dummy untuk pengujian
    dummy_df = pd.DataFrame({
        'open': [100, 105, 110, 115, 120],
        'high': [106, 112, 118, 125, 130],
        'low': [98, 103, 108, 113, 118],
        'close': [105, 110, 115, 120, 128],
        'volume': [1000, 1200, 800, 1500, 2000],
        'rsi': [40, 45, 50, 55, 65], # Contoh nilai RSI
        'ema200': [100, 101, 102, 103, 104] # Contoh nilai EMA200
    })

    dummy_1h_df = pd.DataFrame({
        'open': [1000], 'high': [1050], 'low': [990], 'close': [1040],
        'volume': [5000], 'ema200': [1020]
    })
    
    # Tambahkan kolom yang dibutuhkan untuk 'candle_type'
    dummy_df['ema13'] = dummy_df['close'].ewm(span=13, adjust=False).mean()
    dummy_df['ema34'] = dummy_df['close'].ewm(span=34, adjust=False).mean()
    dummy_df['ema50'] = dummy_df['close'].ewm(span=50, adjust=False).mean()
    dummy_df['ema200'] = dummy_df['close'].ewm(span=200, adjust=False).mean()

    print("\nMenguji prediksi AI:")
    # Pastikan features memiliki kunci yang diharapkan oleh model
    features_to_predict = ai_filter.get_features_from_df(dummy_df, dummy_1h_df)
    print(f"Fitur yang diekstrak: {features_to_predict}")
    score = ai_filter.predict_signal_score(features_to_predict)
    print(f"Skor AI prediksi: {score:.2f}")

    # Contoh skenario SELL
    dummy_df_sell = pd.DataFrame({
        'open': [120, 115, 110, 105, 100],
        'high': [125, 120, 115, 110, 105],
        'low': [118, 113, 108, 103, 98],
        'close': [115, 110, 105, 100, 95],
        'volume': [2000, 1800, 1500, 1200, 1000],
        'rsi': [60, 55, 50, 45, 35], # Contoh nilai RSI
        'ema200': [104, 103, 102, 101, 100] # Contoh nilai EMA200
    })
    dummy_df_sell['ema13'] = dummy_df_sell['close'].ewm(span=13, adjust=False).mean()
    dummy_df_sell['ema34'] = dummy_df_sell['close'].ewm(span=34, adjust=False).mean()
    dummy_df_sell['ema50'] = dummy_df_sell['close'].ewm(span=50, adjust=False).mean()
    dummy_df_sell['ema200'] = dummy_df_sell['close'].ewm(span=200, adjust=False).mean()

    dummy_1h_df_sell = pd.DataFrame({
        'open': [1000], 'high': [1010], 'low': [950], 'close': [960],
        'volume': [5000], 'ema200': [980]
    })

    features_to_predict_sell = ai_filter.get_features_from_df(dummy_df_sell, dummy_1h_df_sell)
    print(f"Fitur yang diekstrak (SELL): {features_to_predict_sell}")
    score_sell = ai_filter.predict_signal_score(features_to_predict_sell)
    print(f"Skor AI prediksi (SELL): {score_sell:.2f}")
