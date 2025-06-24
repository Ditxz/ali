from telegram import Bot
from telegram.error import TelegramError
import asyncio
from app.config import settings

class TelegramBot:
    def __init__(self):
        """Menginisialisasi bot Telegram dengan token dari pengaturan."""
        self.bot = Bot(token=settings.get("TELEGRAM_BOT_TOKEN"))
        self.chat_id = settings.get("TELEGRAM_CHAT_ID")
        if not self.chat_id:
            print("Peringatan: TELEGRAM_CHAT_ID tidak diatur. Sinyal tidak akan dikirim.")

    async def send_signal(self, signal_data: dict):
        """Mengirim sinyal trading yang diformat ke Telegram."""
        if not self.chat_id:
            print("Kesalahan: Tidak dapat mengirim sinyal, TELEGRAM_CHAT_ID tidak diatur.")
            return

        message = self._format_signal_message(signal_data)
        try:
            # Menggunakan parse_mode='MarkdownV2' untuk format yang lebih kaya
            # Perhatikan bahwa karakter khusus di MarkdownV2 perlu di-escape
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='MarkdownV2')
            print(f"Sinyal berhasil dikirim untuk {signal_data['pair']} ({signal_data['timeframe']})")
        except TelegramError as e:
            print(f"Gagal mengirim sinyal ke Telegram: {e}")
        except Exception as e:
            print(f"Terjadi kesalahan tak terduga saat mengirim sinyal: {e}")

    def _format_signal_message(self, signal_data: dict) -> str:
        """Memformat data sinyal menjadi pesan yang profesional untuk Telegram."""
        pair = signal_data.get('pair', 'N/A')
        timeframe = signal_data.get('timeframe', 'N/A')
        signal_type = signal_data.get('signal', 'N/A')
        entry = signal_data.get('entry', 'N/A')
        tp1 = signal_data.get('tp1', 'N/A')
        tp2 = signal_data.get('tp2', 'N/A')
        sl = signal_data.get('sl', 'N/A')
        ai_score = signal_data.get('ai_score', 'N/A')
        confidence_emoji = signal_data.get('confidence_emoji', '')

        # Escape karakter khusus untuk MarkdownV2
        # Referensi: https://core.telegram.org/bots/api#markdownv2-style
        def escape_markdown_v2(text):
            if not isinstance(text, str):
                text = str(text)
            # Karakter yang perlu di-escape: _, *, [, ], (, ), ~, `, >, #, +, -, =, |, {, }, ., !
            # Juga perlu escape backslashes
            return text.replace('\\', '\\\\').replace('_', '\\_').replace('*', '\\*') \
                       .replace('[', '\\[') .replace(']', '\\]').replace('(', '\\(') \
                       .replace(')', '\\)').replace('~', '\\~').replace('`', '\\`') \
                       .replace('>', '\\>').replace('#', '\\#').replace('+', '\\+') \
                       .replace('-', '\\-').replace('=', '\\=').replace('|', '\\|') \
                       .replace('{', '\\{').replace('}', '\\}').replace('.', '\\.') \
                       .replace('!', '\\!')

        # Menerapkan escape ke semua nilai
        pair_escaped = escape_markdown_v2(pair)
        timeframe_escaped = escape_markdown_v2(timeframe)
        signal_type_escaped = escape_markdown_v2(signal_type)
        entry_escaped = escape_markdown_v2(f"{entry:.4f}" if isinstance(entry, (float, int)) else entry)
        tp1_escaped = escape_markdown_v2(f"{tp1:.4f}" if isinstance(tp1, (float, int)) else tp1)
        tp2_escaped = escape_markdown_v2(f"{tp2:.4f}" if isinstance(tp2, (float, int)) else tp2)
        sl_escaped = escape_markdown_v2(f"{sl:.4f}" if isinstance(sl, (float, int)) else sl)
        ai_score_escaped = escape_markdown_v2(f"{ai_score:.2f}")

        message = (
            f"🚨 \\[CHIMERA SIGNAL X\\-99]\n"
            f"📌 Pair: *{pair_escaped}*\n"
            f"⏰ TF: *{timeframe_escaped}*\n"
            f"📈 Signal: *{signal_type_escaped}*\n"
            f"🎯 Entry: `{entry_escaped}`\n"
            f"🎯 TP1: `{tp1_escaped}` \\| TP2: `{tp2_escaped}`\n"
            f"❌ SL: `{sl_escaped}`\n"
            f"🧠 AI Score: `{ai_score_escaped}` {confidence_emoji}\n"
        )
        return message

# Contoh penggunaan (untuk pengujian lokal)
async def main():
    bot = TelegramBot()
    dummy_signal = {
        'pair': 'ETH/USDT',
        'timeframe': '5M',
        'signal': '🟢 BUY STRONG',
        'entry': 3580.0,
        'tp1': 3595.0,
        'tp2': 3615.0,
        'sl': 3568.0,
        'ai_score': 0.94,
        'confidence_emoji': '✅'
    }
    await bot.send_signal(dummy_signal)

if __name__ == "__main__":
    # Untuk menjalankan ini, Anda perlu mengatur TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID
    # di file config.yaml atau sebagai variabel lingkungan.
    # Anda juga perlu menginstal python-telegram-bot: pip install python-telegram-bot==20.6
    asyncio.run(main())
