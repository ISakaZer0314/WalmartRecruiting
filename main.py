import pandas as pd
import train
import feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__":
    t = pd.read_csv("data/train.csv")
    stores = pd.read_csv("data/stores.csv")
    features = pd.read_csv("data/features.csv")

    train_data = feature.create(t, stores, features)

    y = t['Weekly_Sales']

    train_x, test_x, train_y, test_y = train_test_split(train_data, y, test_size=0.2, shuffle=True, random_state=0)


    for n_estimators in [50, 100, 150, 200, 250]:
        params = {'n_estimators': 100, 'max_depth': 3}
        save_model_name = 'model_n_estimators_{}'.format(n_estimators)

        model = train.model_create(train_x, train_y, params, save_model_name)

        predict = model.predict(test_x)

        print("n_estimators: {}".format(n_estimators))
        print("MAE on training set: %.3f" % mean_absolute_error(train_y, model.predict(train_x)))
        print("MAE on test set: %.3f" % mean_absolute_error(test_y, predict))
    
    for max_depth in [3, 5, 7, 9, 11, 13]:
        params = {'n_estimators': 100, 'max_depth': max_depth}
        save_model_name = 'model_max_depth_{}'.format(max_depth)

        model = train.model_create(train_x, train_y, params, save_model_name)

        predict = model.predict(test_x)

        print("max_depth: {}".format(max_depth))
        print("MAE on training set: %.3f" % mean_absolute_error(train_y, model.predict(train_x)))
        print("MAE on test set: %.3f" % mean_absolute_error(test_y, predict))