import logging
import json
import uuid
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tensorflow as tf
import numpy as np
import os
from datetime import datetime, timedelta

try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except ImportError as e:
    logging.error(f"Помилка імпорту transformers: {e}")
    exit()

# =================================================================================
# КОНФІГУРАЦІЯ ТА ГЛОБАЛЬНІ ЗМІННІ
# =================================================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")  # вместо жесткого токена
ADMIN_ID = int(os.getenv("ADMIN_ID", "1481795711"))  # можно тоже вынести

MODEL_PATH = "."
CLASSES = ["Норма", "Порно/інтім", "Реклама/спам"]
DATA_FILE = "bot_data.json"

# ВПЕВНЕНОСТІ ДЛЯ РІЗНИХ ТИПІВ ПЕРЕВІРОК
PROFILE_CONFIDENCE_THRESHOLD = 0.70  # Високий поріг для профілю
TEXT_CONFIDENCE_THRESHOLD = 0.70     # Нижчий поріг для тексту

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# =================================================================================
# ЗАВАНТАЖЕННЯ МОДЕЛІ AI
# =================================================================================

logger.info("Завантаження токенізатора та моделі AI...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    logger.info("✅ Модель AI успішно завантажена.")
except Exception as e:
    logger.error(f"❌ Помилка завантаження моделі AI: {e}")
    exit()

# =================================================================================
# ФУНКЦІЇ ДЛЯ ЗБЕРЕЖЕННЯ ДАНИХ
# =================================================================================

def load_data():
    """Завантажує дані з JSON файлу."""
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            # Конвертуємо строки дат у об'єкти datetime
            for chat_id, chat_data in data.get("active_chats", {}).items():
                if "activated_at" in chat_data:
                    data["active_chats"][chat_id]["activated_at"] = datetime.fromisoformat(chat_data["activated_at"])
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"keys": {}, "active_chats": {}}

def save_data(data):
    """Зберігає дані в JSON файл."""
    # Конвертуємо datetime у строки для зберігання
    data_to_save = data.copy()
    for chat_id, chat_data in data_to_save.get("active_chats", {}).items():
        if "activated_at" in chat_data and isinstance(chat_data["activated_at"], datetime):
            data_to_save["active_chats"][chat_id]["activated_at"] = chat_data["activated_at"].isoformat()
    
    with open(DATA_FILE, "w") as f:
        json.dump(data_to_save, f, indent=4)

def is_chat_active(chat_data):
    """Перевіряє, чи активний ще чат."""
    if not chat_data or "activated_at" not in chat_data:
        return False
    
    activated_at = chat_data["activated_at"]
    key_type = chat_data.get("key_type", "month")
    
    if key_type == "day":
        expiry_time = activated_at + timedelta(days=1)
    else:  # month
        expiry_time = activated_at + timedelta(days=30)
    
    return datetime.now() < expiry_time

# =================================================================================
# ОСНОВНІ ФУНКЦІЇ АНАЛІЗУ
# =================================================================================

def normalize_and_fix_text(text: str) -> str:
    """Нормалізація тексту."""
    char_map = {
        'o': 'о', '0': 'о', 'a': 'а', '@': 'а', 'e': 'е', 'c': 'с', 'k': 'к', 'p': 'р',
        'x': 'х', 'y': 'у', 'i': 'і', 'b': 'б', '6': 'б', 'l': 'л', '1': 'і', '|': 'і',
        'm': 'м', 'h': 'н', 't': 'т'
    }
    text = text.lower()
    fixed_chars = [char_map.get(char, char) for char in text]
    text = "".join(fixed_chars)
    return re.sub(r'\s+', ' ', text).strip()

def predict_text(text: str) -> (str, float):
    """Аналізує текст за допомогою моделі AI."""
    if not text or not text.strip():
        return "Норма", 1.0

    cleaned_text = normalize_and_fix_text(text)
    
    try:
        inputs = tokenizer(cleaned_text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)[0].numpy()
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        category = CLASSES[prediction_idx]
        
        return category, float(confidence)
    except Exception as e:
        logger.error(f"Помилка під час прогнозування: {e}")
        return "Норма", 1.0

def contains_suspicious_links(text: str) -> bool:
    """Перевіряє текст на наявність підозрілих посилань."""
    if not text:
        return False
    
    url_pattern = r'https?://[^\s]+|t\.me/[^\s]+|www\.[^\s]+|\+[^\s]+'
    urls = re.findall(url_pattern, text.lower())
    
    if not urls:
        return False
    
    suspicious_domains = [
        't.me/', 'telegram.me/', 'bit.ly/', 'tinyurl.com', 'shorturl.com',
        'cutt.ly', 'rb.gy', 'goo.gl', 'ow.ly', 'is.gd', 'clck.ru',
        'vk.cc', 'sex', 'porn', 'dating', 'casino', 'bet', 'crypto',
        'onlyfans.com', 'only-fans', 'fansly.com', 'adult', 'xxx',
        'spymer', 'наркоти', 'drugs', 'марихуана', 'weed'
    ]
    
    for url in urls:
        if url.startswith('+') or 't.me/+' in url:
            return True
            
        for domain in suspicious_domains:
            if domain in url:
                return True
    
    return False

async def analyze_profile(user: dict, context: ContextTypes.DEFAULT_TYPE) -> (bool, str, str):
    """
    Аналіз профілю користувача (ТІЛЬКИ bio) - ім'я та username ігноруються.
    Повертає (is_suspicious, reason, details)
    """
    try:
        reasons = []
        details = []
        profile_categories = []

        # Bio (опис профілю)
        try:
            user_chat = await context.bot.get_chat(user.id)
            user_bio = getattr(user_chat, "bio", "") or ""
        except Exception as e:
            logger.warning(f"Не вдалося отримати bio користувача {user.id}: {e}")
            user_bio = ""

        if user_bio.strip():
            bio_category, bio_confidence = predict_text(user_bio)
            if bio_category != "Норма" and bio_confidence >= PROFILE_CONFIDENCE_THRESHOLD:
                reasons.append("підозрілий опис профілю")
                details.append(f"опис: '{user_bio}' → {bio_category} ({bio_confidence:.2f})")
                profile_categories.append(bio_category)

        # Перевірка на бота
        if user.is_bot:
            reasons.append("обліковий запис бота")
            details.append("аккаунт є ботом")

        if reasons:
            if profile_categories:
                categories_info = f" (категорія: {', '.join(set(profile_categories))})"
            else:
                categories_info = ""
            
            return True, ", ".join(reasons) + categories_info, "; ".join(details)
        else:
            return False, "", ""

    except Exception as e:
        logger.warning(f"Не вдалося проаналізувати профіль {user.id}: {e}")
        return False, "", ""

# =================================================================================
# ОБРОБНИКИ КОМАНД ТА ПОВІДОМЛЕНЬ
# =================================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обробник команди /start."""
    user = update.effective_user
    if user.id == ADMIN_ID:
        await update.message.reply_text(
            "Привіт, Адміністраторе!\n\n"
            "Доступні команди:\n"
            "• `/get_key_month НАЗВА_ЧАТУ` - ключ на 30 днів\n"
            "• `/get_key_day НАЗВА_ЧАТУ` - ключ на 1 день\n\n"
            "Ключ активується при першому використанні і діє відповідний термін."
        )
    else:
        await update.message.reply_text("Я бот-модератор. Моє налаштування доступне тільки адміністратору.")

async def generate_key_month(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерує ключ активації на 30 днів."""
    await generate_key(update, context, "month")

async def generate_key_day(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерує ключ активації на 1 день."""
    await generate_key(update, context, "day")

async def generate_key(update: Update, context: ContextTypes.DEFAULT_TYPE, key_type: str):
    """Генерує ключ активації для чату."""
    user = update.effective_user
    if user.id != ADMIN_ID:
        return

    if not context.args:
        if key_type == "month":
            await update.message.reply_text("Будь ласка, вкажіть точну назву чату після команди. Наприклад:\n`/get_key_month Мій Супер Чат`")
        else:
            await update.message.reply_text("Будь ласка, вкажіть точну назву чату після команди. Наприклад:\n`/get_key_day Мій Супер Чат`")
        return

    chat_title = " ".join(context.args)
    key = str(uuid.uuid4())

    data = load_data()
    data["keys"][key] = {
        "chat_title": chat_title,
        "key_type": key_type,
        "used": False
    }
    save_data(data)
    
    duration = "30 днів" if key_type == "month" else "1 день"
    logger.info(f"Згенеровано ключ на {duration} для чату '{chat_title}': {key}")
    
    await update.message.reply_text(
        f"✅ Ключ для чату «{chat_title}» згенеровано!\n"
        f"⏰ Термін дії: {duration}\n"
        f"🔑 Ключ: `{key}`\n\n"
        f"Додайте мене в чат з правами адміністратора та надішліть туди ключ для активації."
    )

async def activate_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Активує бота в чаті, якщо введено правильний ключ."""
    chat_id = update.effective_chat.id
    chat_title = update.effective_chat.title
    text = update.message.text.strip()

    data = load_data()
    
    # Перевіряємо, чи чат вже активований і не прострочений
    if chat_id in data.get("active_chats", {}):
        chat_data = data["active_chats"][chat_id]
        if is_chat_active(chat_data):
            return  # Чат вже активний
        else:
            # Видаляємо прострочений чат
            del data["active_chats"][chat_id]
            save_data(data)

    # Шукаємо ключ
    key_data = data["keys"].get(text)
    if key_data and not key_data["used"]:
        expected_chat_title = key_data["chat_title"]
        key_type = key_data["key_type"]
        
        if chat_title == expected_chat_title:
            # Активуємо чат
            data["active_chats"][chat_id] = {
                "activated_at": datetime.now(),
                "key_type": key_type
            }
            # Позначаємо ключ як використаний
            data["keys"][text]["used"] = True
            save_data(data)
            
            duration = "30 днів" if key_type == "month" else "1 день"
            logger.info(f"БОТ АКТИВОВАНИЙ у чаті '{chat_title}' (ID: {chat_id}) на {duration}")
            
            await update.message.reply_text(
                f"✅ Бот-модератор успішно активований!\n"
                f"⏰ Захист діятиме: {duration}\n"
                f"📅 Активовано: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            )
            await context.bot.delete_message(chat_id=chat_id, message_id=update.message.message_id)
        else:
            await update.message.reply_text("❌ Назва чату не відповідає ключу.")
    elif key_data and key_data["used"]:
        await update.message.reply_text("❌ Цей ключ вже використано.")
    else:
        # Неправильний ключ - ігноруємо
        pass

async def moderate_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ГОЛОВНА ФУНКЦІЯ МОДЕРАЦІЇ"""
    chat_id = update.effective_chat.id
    data = load_data()

    # Перевіряємо, чи чат активний і не прострочений
    chat_data = data.get("active_chats", {}).get(chat_id)
    if not chat_data or not is_chat_active(chat_data):
        return

    user = update.effective_user
    message = update.message
    
    if not user or not message:
        return
        
    if user.id == ADMIN_ID:
        return

    is_suspicious = False
    reason = ""
    details = ""

    # Аналіз профілю
    profile_suspicious, profile_reason, profile_details = await analyze_profile(user, context)
    
    if profile_suspicious:
        is_suspicious = True
        reason = f"Підозрілий профіль: {profile_reason}"
        details = profile_details

    # Аналіз тексту повідомлення (тільки якщо профіль чистий)
    if message.text and not is_suspicious:
        if contains_suspicious_links(message.text):
            is_suspicious = True
            reason = "підозрілі посилання в повідомленні"
            details = "виявлено підозрілі посилання в тексті"
        else:
            text_category, text_confidence = predict_text(message.text)
            if text_category != "Норма" and text_confidence >= TEXT_CONFIDENCE_THRESHOLD:
                is_suspicious = True
                reason = f"підозрілий текст: {text_category}"
                details = f"текст: '{message.text}' (впевненість: {text_confidence:.2f})"

    # Видалення повідомлення
    if is_suspicious:
        try:
            await message.delete()
            logger.info(f"🗑️ Видалено повідомлення від {user.first_name} у чаті {chat_id}. Причина: {reason}")
            
            print(f"\n🔍 ВИЯВЛЕНО ПІДОЗРІЛИЙ КОРИСТУВАЧ:")
            print(f"   👤 Ім'я: {user.first_name} {user.last_name or ''}")
            print(f"   📛 Юзернейм: @{user.username or 'немає'}")
            print(f"   🆔 ID: {user.id}")
            print(f"   📝 Причина: {reason}")
            print(f"   🔍 Деталі: {details}")
            
            try:
                user_chat = await context.bot.get_chat(user.id)
                user_bio = user_chat.bio or ""
                if user_bio:
                    print(f"   📋 Опис профілю: {user_bio}")
            except:
                pass
                
            print("─" * 60)
            
        except Exception as e:
            logger.error(f"Не вдалося видалити повідомлення у чаті {chat_id}: {e}")

# =================================================================================
# ГОЛОВНА ФУНКЦІЯ ЗАПУСКУ БОТА
# =================================================================================

def main():
    """Запускає бота."""
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("get_key_month", generate_key_month, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("get_key_day", generate_key_day, filters=filters.ChatType.PRIVATE))

    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS, activate_bot), group=0)
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS, moderate_message), group=1)
    
    logger.info("🤖 Бот запускається...")
    print("\n" + "="*60)
    print("🚀 БОТ-МОДЕРАТОР ЗАПУЩЕНИЙ")
    print("🔑 Доступні ключі:")
    print("   • /get_key_month - на 30 днів")
    print("   • /get_key_day - на 1 день")
    print("⏰ Термін дії відлічується від моменту активації")
    print("="*60 + "\n")
    
    application.run_polling()

if __name__ == "__main__":
    main()
