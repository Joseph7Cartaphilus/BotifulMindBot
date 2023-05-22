import telebot
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import sqlite3
import bcrypt

# Создаем бота и задаем его токен
bot = telebot.TeleBot("6100020200:AAFhd4ai7QIRnUc1dbgXOE9cDIxFLh0KZH4")

# Загружаем данные для нейронной сети из файлов
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_models.model')

# Подключение к базе данных
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Создание таблицы пользователей
c.execute('''CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT,
                 password TEXT,
                 user_id INTEGER)''')

# Создание таблицы истории сообщений
c.execute('''CREATE TABLE IF NOT EXISTS messages
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 message TEXT,
                 FOREIGN KEY (user_id) REFERENCES users (id))''')

# Закрытие соединения с базой данных
conn.close()


# Функция, которая проверяет, является ли пользователь зарегистрированным
def is_user_registered(user_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    return bool(user)


# Функция, которая получает ответ бота на входящее сообщение
def get_bot_response(message):
    sentence_words = nltk.word_tokenize(message)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag_of_words = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag_of_words[i] = 1
    intent_predictions = model.predict(np.array([bag_of_words]))[0]
    predicted_intent_index = np.argmax(intent_predictions)
    predicted_intent_tag = classes[predicted_intent_index]

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent_tag:
            response = random.choice(intent['responses'])
            break
    return response


# Функция, которая отображает пользователю кнопки выбора
def show_menu(message):
    markup = telebot.types.ReplyKeyboardMarkup(row_width=1)
    item1 = telebot.types.KeyboardButton('Advise a movie/TV series')
    item2 = telebot.types.KeyboardButton('I want to talk')
    item3 = telebot.types.KeyboardButton('Tell us about the movie/series')
    item4 = telebot.types.KeyboardButton('History of appeals')
    markup.add(item1, item2, item3, item4)
    bot.send_message(message.chat.id, "Select one of the options:", reply_markup=markup)


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start_message(message):
    user_id = message.from_user.id

    if is_user_registered(user_id):
        show_menu(message)
    else:
        bot.send_message(message.chat.id, "Hello! You are not registered. Please use the /register command to register.")


# Обработчик команды /register
@bot.message_handler(commands=['register'])
def register_message(message):
    user_id = message.from_user.id

    if is_user_registered(user_id):
        bot.send_message(message.chat.id, "You are already registered.")
        return

    # Хеширование пароля пользователя
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user_id.encode('utf-8'), salt)

    # Регистрация пользователя в базе данных
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, user_id) VALUES (?, ?, ?)", (message.from_user.username, hashed_password, user_id))
    conn.commit()
    conn.close()

    bot.send_message(message.chat.id, "You have been registered successfully.")
    show_menu(message)


# Обработчик входящих сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id

    if not is_user_registered(user_id):
        bot.send_message(message.chat.id, "You are not registered. Please use the /register command to register.")
        return

    # Сохранение сообщения в базу данных
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO messages (user_id, message) VALUES (?, ?)", (user_id, message.text))
    conn.commit()
    conn.close()

    # Обработка сообщения от зарегистрированного пользователя
    if message.text == 'Advise a movie/TV series':
        response = get_bot_response(message.text)
    elif message.text == 'I want to chat':
        response = "What do you want to know about?"
    elif message.text == 'Tell us about the movie/series':
        response = "What movie/series do you want to know about?"
    elif message.text == 'History of appeals':
        show_history(message)
        return
    else:
        response = "Select one of the options:"

    bot.send_message(message.chat.id, response)


# Функция, которая отображает историю обращений пользователя
def show_history(message):
    user_id = message.from_user.id

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT message FROM messages WHERE user_id=?", (user_id,))
    messages = c.fetchall()
    conn.close()

    if messages:
        response = "The history of your appeals:\n"
        for i, msg in enumerate(messages):
            response += f"{i + 1}. {msg[0]}\n"
    else:
        response = "You don't have any requests in your history yet."

    bot.send_message(message.chat.id, response)


# Запуск бота
bot.polling()
