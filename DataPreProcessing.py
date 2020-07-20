import numpy as np
import re
import pickle
import nltk
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import text
from keras import utils
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


def text_cleaner(text):
    """
    Function to pre-process the text required by the model.
    :param text: text to be used.
    :return: pre-processed or cleaned text.
    """
    replace_by_space = re.compile('[/(){}\[\]\|@,;.]')
    bad_symbols = re.compile('[^0-9a-z #+_=*&]')
    stop_words = set(stopwords.words('english'))

    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = replace_by_space.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = bad_symbols.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in stop_words)  # delete stopwors from text
    return text


def save_trained_values(content, name):
    """
    Function to save the trained model components
    :param content: content to be saved.
    :param name: file name
    """
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tokenizer(train_posts, test_posts, max_words):
    """
    Function to tokenize and convert texts_to_matrix(posts).
    :param train_posts: training data(post)
    :param test_posts: testing data(post)
    :param max_words: Number of words for tokenizer.
    :return: tokenized train and test posts.
    """
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(train_posts)  # only fit on train

    # Save tokenizer
    save_trained_values(tokenize, 'tokenizer')

    # tokenize train and text data to matrix.
    x_train = tokenize.texts_to_matrix(train_posts)
    x_test = tokenize.texts_to_matrix(test_posts)

    # Get our training data word index
    word_index = tokenize.word_index

    return x_train, x_test, word_index


def label_encoder(train_tags, test_tags):
    """
    Function to hot-encode the labels. (Text labels are converted to Keras hot-encoded labels.)
    :param train_tags: training data(tags)
    :param test_tags: testing data(tags)
    :return: hot-encoded train and test labels.
    """
    encoder = LabelEncoder()
    encoder.fit(train_tags)

    # Save tokenizer
    #save_trained_values(encoder, 'encoder')

    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    num_classes = np.max(y_train) + 1
    # one-hot encoding labels usng Keras.
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return y_train, y_test, num_classes
