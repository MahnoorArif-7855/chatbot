# from itertools import Predicate
import json
from os import stat
import pickle
import random
import tkinter as tk
from tkinter import *
from tkinter import font
from keras import models
from keras.layers.core.lambda_layer import Lambda
from keras.models import load_model
import nltk
from nltk import probability
from nltk.stem import WordNetLemmatizer
import numpy as np
# from nltk.tag import sequential

import keras
# import np
lemmatizer = WordNetLemmatizer()
#imports file form folders
intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def bow(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return (np.array(bag))
    # return sentence_words

#chatbot response or prediction
def predict_class(sentence):
    sentence_bag = bow(sentence)
    res = model.predict(np.array([sentence_bag]))[0]
    ERROR_THRESHOLBD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLBD]
    # sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

#user msg
def getResponse(ints):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break

    return result


def chatbot_response(msg):
    ints = predict_class(msg)
    res = getResponse(ints)
    return res


#Send Msg Button
def send():
    msg = TextEntryBox.get("1.0", 'end-1c').strip()
    TextEntryBox.delete('1.0', 'end')

    if msg != '':
        chatHistory.config(state=NORMAL)
        chatHistory.insert('end', 'You: 😁' + msg + "\n\n")

        res = chatbot_response(msg)
        chatHistory.insert('end', 'bot: 🤓' + res + "\n\n")
        chatHistory.config(state=DISABLED)
        chatHistory.yview('end')


base = tk.Tk()
base.title("Alexa")
base.geometry("400x500")
base.resizable(width=False, height=False)


# chat history textview
chatHistory = Text(base, bd=0, bg='white', font="Arial")
chatHistory.config(state=DISABLED)

SendButton = Button(base, font=('Arial', 12, 'bold'),
                    text='Send', bg='#dfdfdf', activebackground="#3e3e3e", fg='#ffffff', command=send)
TextEntryBox = Text(base, bd=0, bg='white', font='Arial')
chatHistory.place(x=6, y=6, height=386, width=386)
TextEntryBox.place(x=128, y=400, height=80, width=265)
SendButton.place(x=6, y=400, height=80, width=125)

base.mainloop()
