import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from card_detection import get_data_details, preprocess, train_test_partition, train_model, predict_model, \
    resample_dataset, predict_single, show_result_stats


def make_sample_dataset():
    data_file = "creditcard.csv"
    dataframe = pd.read_csv(data_file)
    return dataframe.sample(frac=0.2)


def test_get_data_details():
    sample_df = make_sample_dataset()
    map_res = get_data_details(sample_df)
    fraud_percent = map_res["fraud_percent"]
    print(fraud_percent)
    assert 0 < fraud_percent < 1


def test_preprocess():
    sample_df = make_sample_dataset()
    xi, yi = preprocess(sample_df)
    x_cols = xi.columns.to_list()
    assert "Class" not in x_cols
    assert "Amount" not in x_cols
    assert "Time" not in x_cols


def test_train_test_partition():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (act_train_X, act_test_X, act_train_Y, act_test_Y) = train_test_partition(xi, yi, 0.3, 42)
    (pred_train_X, pred_test_X, pred_train_Y, pred_test_Y) = train_test_split(xi, yi, test_size=0.3, random_state=42)
    assert act_train_X.shape == pred_train_X.shape
    assert act_test_X.shape == pred_test_X.shape
    assert act_test_X.shape == pred_test_X.shape


def test_resample_dataset():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    xii, yii = resample_dataset(xi, yi)
    (nums_initial, rows_initial) = xi.shape
    (nums, rows) = xii.shape
    print(nums, nums_initial, rows_initial, rows)
    assert nums > nums_initial and rows_initial == rows


# Testing Logistic Regression


def test_predict_model_lr():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    logistic_regression = linear_model.LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
    train_model(logistic_regression, train_X, train_Y)
    (predictions_lr, linear_regression_score) = predict_model("Logistic Regression", logistic_regression, test_X,
                                                              test_Y)
    show_result_stats("Logistic Regression", test_Y, predictions_lr)
    assert linear_regression_score > 90


def test_predict_single_lr():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    logistic_regression = linear_model.LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
    train_model(logistic_regression, train_X, train_Y)
    test_x_numpy = test_X.to_numpy()
    test_y_numpy = test_Y.to_numpy()
    count = 100
    valid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act = test_y_numpy[0]
        test_pred = predict_single(logistic_regression, test_val)
        valid += 1 if test_pred == test_act else 0
    print("valid = ", valid)
    assert valid > 96


def test_predict_single_lr_neg():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    logistic_regression = linear_model.LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
    train_model(logistic_regression, train_X, train_Y)
    test_x_numpy = test_X.to_numpy()
    test_y_numpy = test_Y.to_numpy()
    count = 100
    invalid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act_neg = 1 if test_y_numpy[0] == 0 else 0
        test_pred = predict_single(logistic_regression, test_val)
        invalid += 1 if test_pred == test_act_neg else 0
    print("invalid = ", invalid)
    assert invalid > 4


# Testing Decision Tree


def test_predict_model_dt():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    decision_tree = DecisionTreeClassifier()
    train_model(decision_tree, train_X, train_Y)
    (predictions_lr, decision_tree_score) = predict_model("Decision Tree", decision_tree, test_X,
                                                              test_Y)
    show_result_stats("Decision Tree", test_Y, predictions_lr)
    assert decision_tree_score > 90


def test_predict_single_dt():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    decision_tree = DecisionTreeClassifier()
    train_model(decision_tree, train_X, train_Y)
    test_x_numpy = test_X.to_numpy()
    test_y_numpy = test_Y.to_numpy()
    count = 100
    valid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act = test_y_numpy[0]
        test_pred = predict_single(decision_tree, test_val)
        valid += 1 if test_pred == test_act else 0
    print("valid = ", valid)
    assert valid > 96


def test_predict_single_dt_neg():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    decision_tree = DecisionTreeClassifier()
    train_model(decision_tree, train_X, train_Y)
    test_x_numpy = test_X.to_numpy()
    test_y_numpy = test_Y.to_numpy()
    count = 100
    invalid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act_neg = 1 if test_y_numpy[0] == 0 else 0
        test_pred = predict_single(decision_tree, test_val)
        invalid += 1 if test_pred == test_act_neg else 0
    print("invalid = ", invalid)
    assert invalid > 4


# Testing Random Forest


def test_predict_model_rr():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    random_forest = RandomForestClassifier(n_estimators=100)
    train_model(random_forest, train_X, train_Y)
    (predictions_lr, random_forest_score) = predict_model("Logistic Regression", random_forest, test_X, test_Y)
    show_result_stats("Random Forest", test_Y, predictions_lr)
    assert random_forest_score > 90


def test_predict_single_rr():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    random_forest = RandomForestClassifier(n_estimators=100)
    train_model(random_forest, train_X, train_Y)
    test_x_numpy = test_X.to_numpy()
    test_y_numpy = test_Y.to_numpy()
    count = 100
    valid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act = test_y_numpy[0]
        test_pred = predict_single(random_forest, test_val)
        valid += 1 if test_pred == test_act else 0
    print("valid = ", valid)
    assert valid > 96


def test_predict_single_rr_neg():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    (train_X, test_X, train_Y, test_Y) = train_test_partition(xi, yi, 0.3, 42)
    random_forest = RandomForestClassifier(n_estimators=100)
    train_model(random_forest, train_X, train_Y)
    test_x_numpy = test_X.to_numpy()
    test_y_numpy = test_Y.to_numpy()
    count = 100
    invalid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act_neg = 1 if test_y_numpy[0] == 0 else 0
        test_pred = predict_single(random_forest, test_val)
        invalid += 1 if test_pred == test_act_neg else 0
    print("invalid = ", invalid)
    assert invalid > 4


# Testing Resampled Random Forest


def test_predict_model_rrr():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    x_resampled, y_resampled = resample_dataset(xi, yi)
    (train_X2, test_X2, train_Y2, test_Y2) = train_test_partition(x_resampled, y_resampled, 0.3, 42)
    rf_resampled = RandomForestClassifier(n_estimators=100)
    train_model(rf_resampled, train_X2, train_Y2)
    (predictions_lr, random_forest_score_resampled) = predict_model("Resampled Random Forest", rf_resampled, test_X2, test_Y2)
    show_result_stats("Resampled Random Forest", test_Y2, predictions_lr)
    assert random_forest_score_resampled > 90


def test_predict_single_rrr():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    x_resampled, y_resampled = resample_dataset(xi, yi)
    (train_X2, test_X2, train_Y2, test_Y2) = train_test_partition(x_resampled, y_resampled, 0.3, 42)
    rf_resampled = RandomForestClassifier(n_estimators=100)
    train_model(rf_resampled, train_X2, train_Y2)
    test_x_numpy = test_X2.to_numpy()
    test_y_numpy = test_Y2.to_numpy()
    count = 100
    valid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act = test_y_numpy[0]
        test_pred = predict_single(rf_resampled, test_val)
        valid += 1 if test_pred == test_act else 0
    print("valid = ", valid)
    assert valid > 96


def test_predict_single_rrr_neg():
    df = make_sample_dataset()
    xi, yi = preprocess(df)
    x_resampled, y_resampled = resample_dataset(xi, yi)
    (train_X2, test_X2, train_Y2, test_Y2) = train_test_partition(x_resampled, y_resampled, 0.3, 42)
    rf_resampled = RandomForestClassifier(n_estimators=100)
    train_model(rf_resampled, train_X2, train_Y2)
    test_x_numpy = test_X2.to_numpy()
    test_y_numpy = test_Y2.to_numpy()
    count = 100
    invalid = 0
    for i in range(count):
        test_val = test_x_numpy[0]
        test_act_neg = 1 if test_y_numpy[0] == 0 else 0
        test_pred = predict_single(rf_resampled, test_val)
        invalid += 1 if test_pred == test_act_neg else 0
    print("invalid = ", invalid)
    assert invalid > 4



# data = [[0.00000000e+00, -1.35980713e+00, -7.27811733e-02,
#          2.53634674e+00, 1.37815522e+00, -3.38320770e-01,
#          4.62387778e-01, 2.39598554e-01, 9.86979013e-02,
#          3.63786970e-01, 9.07941720e-02, -5.51599533e-01,
#          -6.17800856e-01, -9.91389847e-01, -3.11169354e-01,
#          1.46817697e+00, -4.70400525e-01, 2.07971242e-01,
#          2.57905802e-02, 4.03992960e-01, 2.51412098e-01,
#          -1.83067779e-02, 2.77837576e-01, -1.10473910e-01,
#          6.69280749e-02, 1.28539358e-01, -1.89114844e-01,
#          1.33558377e-01, -2.10530535e-02, 1.49620000e+02,
#          0],
#         [4.06000000e+02, -2.31222654e+00, 1.95199201e+00,
#          -1.60985073e+00, 3.99790559e+00, -5.22187865e-01,
#          -1.42654532e+00, -2.53738731e+00, 1.39165725e+00,
#          -2.77008928e+00, -2.77227214e+00, 3.20203321e+00,
#          -2.89990739e+00, -5.95221881e-01, -4.28925378e+00,
#          3.89724120e-01, -1.14074718e+00, -2.83005567e+00,
#          -1.68224682e-02, 4.16955705e-01, 1.26910559e-01,
#          5.17232371e-01, -3.50493686e-02, -4.65211076e-01,
#          3.20198199e-01, 4.45191675e-02, 1.77839798e-01,
#          2.61145003e-01, -1.43275875e-01, 0.00000000e+00,
#          1],
#         [4.72000000e+02, -3.04354062e+00, -3.15730712e+00,
#          1.08846278e+00, 2.28864362e+00, 1.35980513e+00,
#          -1.06482252e+00, 3.25574266e-01, -6.77936532e-02,
#          -2.70952836e-01, -8.38586565e-01, -4.14575448e-01,
#          -5.03140860e-01, 6.76501545e-01, -1.69202893e+00,
#          2.00063484e+00, 6.66779696e-01, 5.99717414e-01,
#          1.72532101e+00, 2.83344830e-01, 2.10233879e+00,
#          6.61695925e-01, 4.35477209e-01, 1.37596574e+00,
#          -2.93803153e-01, 2.79798032e-01, -1.45361715e-01,
#          -2.52773123e-01, 3.57642252e-02, 5.29000000e+02,
#          1]]
# columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
#            'Class']
# df = pd.DataFrame(data, columns=columns)
