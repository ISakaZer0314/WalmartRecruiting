from sklearn.ensemble import GradientBoostingRegressor
import pickle


def model_create(x, y, params, save_model_name):

    model = GradientBoostingRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

    model.fit(x, y)

    pickle.dump(model, open("model/{}.pickle".format(save_model_name), "wb"))

    return model