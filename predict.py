import datetime
import pickle
from keras.models import load_model
from sql_helper import read_from_solarworks
from DataPreProcessing import text_cleaner
from DataLoader import load_data

def model_load():
    model = load_model('model_text_classification_final_N.h5')
    return model


def load_tokenizer():
    with open('tokenizer_N.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    return loaded_tokenizer


def load_labels():
    with open('encoder_N.pickle', 'rb') as handle:
        encoded_labels = pickle.load(handle)
    return encoded_labels


def write_to_table(dataframe):
    write_to_table(dataframe, 'predictions')


def main():
    """
    Function to predict.
    :param dataframe: DataFrame containing notes.
    :return: DataFrame with predicted labels.
    """
    # load the data for prediction.
    dataframe = load_data()
    print("Total number of records for prediction, %s", len(dataframe))
    #dataframe = dataframe[0:3]

    model = model_load()
    tokenizer = load_tokenizer()
    encoder = load_labels()

    # clean the text.
    dataframe['notes'] = dataframe['notes'].apply(lambda x: text_cleaner(x))
    # tokenize the text and predict.
    dataframe['predicted'] = dataframe['notes'].apply(
        lambda x: model.predict_classes(tokenizer.texts_to_matrix([x])).tolist())
    # Convert to labels.
    dataframe['subqueueName'] = dataframe['predicted'].apply(lambda x: encoder.inverse_transform(x)[0])

    return dataframe


if __name__ == '__main__':
    CURRENT_DATE = datetime.datetime.now()
    main()
    print("Completed prediction in %s seconds", (datetime.datetime.now()-CURRENT_DATE).total_seconds())
