from collections import Counter
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

my_data_file = "creditcard.csv"


def read_data(data_file):
    df = pd.read_csv(data_file)
    df.head()
    df.isnull().values.any()
    df["Amount"].describe()
    return df


def get_data_details(df):
    non_fraud = len(df[df.Class == 0])
    fraud = len(df[df.Class == 1])
    fraud_percent = (fraud / (fraud + non_fraud)) * 100
    print("Number of Genuine transactions: ", non_fraud)
    print("Number of Fraud transactions: ", fraud)
    print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))
    return {'non_fraud': non_fraud, 'fraud': fraud, 'fraud_percent': fraud_percent}


def preprocess(df):
    std = StandardScaler()
    df["NormalizedAmount"] = std.fit_transform(df["Amount"].values.reshape(-1, 1))
    df.drop(["Amount", "Time"], inplace=True, axis=1)
    yi = df["Class"]
    xi = df.drop(["Class"], axis=1)
    return xi, yi


def train_test_partition(x, y, test_size, random_state):
    (train_XI, test_XI, train_YI, test_YI) = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print("Shape of train_X: ", train_XI.shape)
    print("Shape of test_X: ", test_XI.shape)
    print()
    return train_XI, test_XI, train_YI, test_YI


def train_model(model, train_x, train_y):
    model.fit(train_x, train_y)


def predict_model(name, model, test_x, test_y):
    predictions = model.predict(test_x)
    score = model.score(test_x, test_y) * 100
    print(name, "Score:", score)
    return predictions, score


def metrics(actual, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actual, predictions)))
    print("Precision: {:.5f}".format(precision_score(actual, predictions)))
    print("Recall: {:.5f}".format(recall_score(actual, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actual, predictions)))
    print()


def print_confusion_matrix(name, matrix):
    print("Confusion Matrix - " + name)
    print(matrix)


def print_metrics(name, test_y, predictions):
    print("Evaluation of " + name + " Model")
    print()
    metrics(test_y, predictions)


def show_result_stats(name, test_y, predictions):
    confusion_matrix_model = confusion_matrix(test_y, predictions)
    print_confusion_matrix(name, confusion_matrix_model)
    print_metrics(name, test_y, predictions.round())


def resample_dataset(x, y):
    x_resampled, y_resampled = SMOTE().fit_resample(x, y)
    print("Resampled shape of X: ", x_resampled.shape)
    print("Resampled shape of Y: ", y_resampled.shape)
    value_counts = Counter(y_resampled)
    print(value_counts)
    print()
    return x_resampled, y_resampled


def predict_single(model, x_val):
    return model.predict(np.array([x_val]))[0]


dataframe = read_data(my_data_file)
get_data_details(dataframe)

# Preprocessing
(X, Y) = preprocess(dataframe)

# Data Splitting
(train_X, test_X, train_Y, test_Y) = train_test_partition(X, Y, 0.3, 42)


# Logistic regression
def model_lr(train_x, train_y, test_x, test_y):
    logistic_regression = linear_model.LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
    train_model(logistic_regression, train_x, train_y)
    (predictions_lr, linear_regression_score) = predict_model("Logistic Regression", logistic_regression,
                                                              test_x, test_y)
    show_result_stats("Logistic Regression", test_Y, predictions_lr)
    return test_Y, predictions_lr, linear_regression_score


# Decision Tree
def model_dt(train_x, train_y, test_x, test_y):
    decision_tree = DecisionTreeClassifier()
    train_model(decision_tree, train_x, train_y)
    (predictions_dt, decision_tree_score) = predict_model("Decision Tree", decision_tree, test_x, test_y)
    show_result_stats("Decision Tree", test_y, predictions_dt)
    return test_Y, predictions_dt


# Random Forest
def model_rf(train_x, train_y, test_x, test_y):
    random_forest = RandomForestClassifier(n_estimators=100)
    train_model(random_forest, train_x, train_y)
    predictions_rf, random_forest_score = predict_model("Random Forest", random_forest, test_x, test_y)
    show_result_stats("Random Forest", test_Y, predictions_rf)
    return random_forest, test_Y, predictions_rf


# Resampled Random Forest
def model_random_rf(x, y):
    x_resampled, y_resampled = resample_dataset(x, y)

    (train_X2, test_X2, train_Y2, test_Y2) = train_test_partition(x_resampled, y_resampled, 0.3, 42)

    rf_resampled = RandomForestClassifier(n_estimators=100)
    train_model(rf_resampled, train_X2, train_Y2)
    predictions_resampled, random_forest_score_resampled = predict_model("Resampled Random Forest", rf_resampled,
                                                                         test_X2, test_Y2)
    show_result_stats("Resampled Random Forest", test_Y2, predictions_resampled)
    return test_Y2, predictions_resampled
