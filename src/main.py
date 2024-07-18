import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def get_clean_data():
    """
    The dataset contains an erroneous colmn consisting entirely of 'NaN' which we will drop
    along with the 'id' column. Additionally, we need to change the strings in the 'diagnosis'
    column from 'M' (Malignant) or 'B' (Benign) to '1' or '0'.
    """
    data = pd.read_csv("data/cancer-data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def create_model(data):
    """
    The target variable is the diagnosis. We leave everything else as a predictor.
    The predictors are in various unit types sizes, so we need to normalize them.
    """
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))

    return model, scaler


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('src/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('src/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
