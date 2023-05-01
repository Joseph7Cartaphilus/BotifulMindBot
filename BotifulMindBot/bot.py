import telebot
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Создаем бота и задаем его токен
bot = telebot.TeleBot("6100020200:AAFhd4ai7QIRnUc1dbgXOE9cDIxFLh0KZH4")

# Загружаем данные для нейронной сети из файлов
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_models.model')

# Функция, которая получает ответ бота на входящее сообщение
def get_bot_response(message):
    # Прогнозируем тег интента на основе входящего сообщения
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

    # Выбираем ответ на основе прогнозируемого интента
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent_tag:
            response = random.choice(intent['responses'])
            break
    return response

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Hello! I'm a bot ready to answer your questions.")

# Обработчик входящих сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    response = get_bot_response(message.text)
    bot.send_message(message.chat.id, response)

# Запускаем бота
bot.polling()
