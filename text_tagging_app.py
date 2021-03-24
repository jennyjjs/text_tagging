import pandas as pd
import streamlit
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, abort
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras import optimizers
from keras.layers.recurrent import LSTM
from keras.layers import Masking
from keras.layers import Dropout
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import re
from collections import Counter
import nltk
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import text_to_word_sequence
from typing import List
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')


# Create application and api
app = Flask(__name__)
api = Api(app)

# Add argument
parser = reqparse.RequestParser()
parser.add_argument("text", type = str, required = True)

def incorrect_format(txt):
    try:
        str(txt)
    except TypeError:
        abort(404, message = "Text is not a string")


# Import data
df_clean = pd.read_csv("/Users/jennyshang/Documents/Forma AI/Mentorship/Sample_Data_clean.csv")

# Add embedding
labels = df_clean["Line item"]
docs = df_clean["stop_removed_keyword"]

encoder = LabelEncoder()
labels = to_categorical(encoder.fit_transform(labels))

tokenizer = Tokenizer(num_words=10000, oov_token = "UNKNOWN_TOKEN")
tokenizer.fit_on_texts(df_clean["stop_removed_keyword"].values)

def get_max_token_length_per_doc(docs):
    return max(list(map(lambda x: len(x.split()), docs)))

max_length = get_max_token_length_per_doc(docs)

def integer_encode_documents(docs, tokenizer):
    return tokenizer.texts_to_sequences(docs)

def get_max_token_length_per_doc(docs):
    return max(list(map(lambda x: len(x.split()), docs)))

max_length = get_max_token_length_per_doc(docs)
# Define the number of time steps the model should back propagate.

MAX_SEQUENCE_LENGTH = 50

## We encode the training data, where the documents are each product's information (name, brand, description
## and details)

encoded_docs = integer_encode_documents(df_clean["stop_removed_keyword"].values, tokenizer)

padded_docs = pad_sequences(encoded_docs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

## Define the training inputs and outputs.

X_train = padded_docs
y_train = labels
VOCAB_SIZE = int(len(tokenizer.word_index) * 1.1)


## GLOVE embeddings

def load_glove_vectors():
    embeddings_index = {}
    with open('/Users/jennyshang/Documents/Forma AI/Mentorship/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index

embeddings_index = load_glove_vectors()

## Create a matrix for the words in the training dataset. rndom

embedding_matrix = zeros((VOCAB_SIZE, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Add model

def make_lstm_classification_model():
    from keras.layers.recurrent import LSTM
    from keras.layers import Masking
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(units=20, input_shape=(1, MAX_SEQUENCE_LENGTH)))
    model.add(Dense(5))
    # model.add(Dropout(0.25))
    model.add(Dense(92, activation='softmax'))

    opt = optimizers.RMSprop(lr=0.03)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = make_lstm_classification_model()
model.fit(X_train, y_train, batch_size = 2, epochs=50, verbose=1)

# Prediction function
def pred_line_item(doc, model):

    stop = list(set(stopwords.words('english')))
    stop.extend(["nan",",","","'",".","@","/",":",";","_"])

    cleaned_txts = []
    for text in doc:
        words = word_tokenize(text)
        new_words = []
        for word in words:
            if word in stop:
                continue
            new_words.append(word.lower())
        cleaned_txt = " ".join(new_words)
        cleaned_txts.append(cleaned_txt)

    my_doc = [" ".join(cleaned_txts[i] for i in range(len(cleaned_txts)))]


    encoded_text = integer_encode_documents(my_doc, tokenizer)
    padded_text = pad_sequences(encoded_text, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    prediction = model.predict_classes(padded_text)
    line_item = encoder.inverse_transform(prediction)

    return line_item

# Define class

class Get_Line_Item(Resource):

    def get(self, text):
        incorrect_format(text)
        result = pred_line_item(text,model)[0]

        return jsonify(result)

# Add endpoint
api.add_resource(Get_Line_Item, '/getlineitem/<string:text>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
