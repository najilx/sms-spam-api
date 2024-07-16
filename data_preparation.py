# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Load the dataset
    data = pd.read_csv('spam.csv', encoding='latin-1')

    # Keep only the necessary columns
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']

    # Map 'ham' to 0 and 'spam' to 1
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=1)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("Data preparation completed.")
