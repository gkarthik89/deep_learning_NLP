import datetime
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

from DataLoader import load_data
from DataPreProcessing import text_cleaner, tokenizer, label_encoder


def create_training_dataset(X, y, test_size=0.3, shuffle=True):
    return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)


def create_model(max_words, num_classes, number_node, dropout, loss_function, optimizer, metrics):
    # Build the model
    model = Sequential()
    model.add(Dense(number_node, input_shape=(max_words,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def save_model(model, file_name):
    # save the trained model.
    tf.keras.models.save_model(model, file_name+".h5")


def train_model(model, x_train, y_train):

    hist = model.fit(x_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=EPOCH,
                     verbose=1,
                     validation_split=VALIDATION_SPLIT)
    save_model(model, "case_notes_classification")
    return hist


def plot_metrics(hist):
    """
    Function to plot.
    :param hist: Training values of model.
    """
    history = hist.history
    # plot train/ test loss-accuracy metrics.
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    """
    Main fucntion to train the model.
    :return: nothing
    """
    dataframe = load_data()
    dataframe['Notes'] = dataframe['Notes'].apply(lambda x: text_cleaner(x))
    train_posts, test_posts, train_tags, test_tags = create_training_dataset(dataframe.Notes, dataframe.LabelName)
    x_train, x_test, word_index = tokenizer(train_posts, test_posts, max_words=MAX_WORDS)
    y_train, y_test, num_classes = label_encoder(train_tags, test_tags)
    model = create_model(MAX_WORDS, num_classes, NUMBER_NODES, DROPOUT, LOSS, OPTIMIZER, METRICS)
    model.summary()
    hist = train_model(model, x_train, y_train)
    plot_metrics(hist)

    return


if __name__ == '__main__':
    CURRENT_DATE = datetime.datetime.now()
    MAX_WORDS = 1000
    NUMBER_NODES = 2500
    DROPOUT = 0.7
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = ['accuracy']
    BATCH_SIZE = 2000
    EPOCH = 1
    VALIDATION_SPLIT = 0.1

    main()
    print("Completed prediction in %s seconds", (datetime.datetime.now()-CURRENT_DATE).total_seconds())
