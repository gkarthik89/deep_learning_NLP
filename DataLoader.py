import pandas as pd


def filter_dataframe(dataframe):
    """
    Function to filter rows that have count less than specified value for training neural Network.
    :param dataframe:
    :return:  DataFrame
    """
    number_of_rows = 50
    counts = dataframe['LabelID'].value_counts()
    # Select the values where the count is less than 200
    to_remove = counts[counts <= number_of_rows].index

    # Keep rows where the city column is not in to_remove
    dataframe = dataframe[~dataframe.LabelID.isin(to_remove)]
    print("Total number of records after filteration %s",len(dataframe))
    return dataframe


def load_data():
    """
    Fucntion to load the data required for the model.
    :return: dataframe
    """
    df = pd.read_csv('New1.csv',sep=',',
                      names=["ID", "LabelID", "LabelName", "Notes"])
    # Drop rows with NAs.
    df.dropna(subset=['Notes'], inplace=True)
    print("Total number of records got from the csv %s", len(df))
    # filter dataframe.
    df = filter_dataframe(df)
    return df
