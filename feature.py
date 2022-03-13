import pandas as pd
import config as cfg
from sklearn import preprocessing


def create(t, stores, features):
    joined_train = join(t, stores, features)
    joined_train_with_new_info = add(joined_train)
    complemented = completion(joined_train_with_new_info, ['CPI', 'Unemployment'])
    encoded = label_encoding(complemented)
    return extract(encoded, cfg.X_CLOMNS)

def join(t, stores, features):
    df = pd.merge(t, stores, on='Store', how='left')
    return pd.merge(df, features.drop(['IsHoliday'], axis=1), on=['Store', 'Date'], how='left')

def add(data):
    has_other_info = _add_is_thanksgiving(data)
    return _add_month(has_other_info)

def _add_is_thanksgiving(data):
    is_thanksgiving = [False] * len(data)
    has_is_thanksgiving = data.assign(IsThanksgiving = is_thanksgiving)
    for thanksgiving in cfg.TANKSGIVING:
        has_is_thanksgiving.loc[has_is_thanksgiving['Date'] == thanksgiving, 'IsThanksgiving'] = True
    return has_is_thanksgiving

def _add_month(data):
    data['Month'] = data['Date'].str[5:7]
    return data

def completion(data, target_clomns):
    for tc in target_clomns:
        if (tc == 'CPI'):
            _each_store(data, tc)
        if (tc == 'Unemployment'):
            _all_at_once(data, tc)
    return data

def _each_store(data, tc):
    for store_id in set(data['Store'].values.tolist()):
        if (_is_null_count(data[data['Store'] == store_id][tc]) > 0):
            mean = data[data['Store'] == store_id][tc].mean()
            data[data['Store'] == store_id & data['CPI'].isnull() == True]['CPI'] = mean
    return data

def _all_at_once(data, tc):
    if (_is_null_count(data[tc]) > 0):
        mean = data[tc].mean()
        data[data[tc.isnull() == True]][tc] = mean
    return data

def _is_null_count(d):
    return d.isnull().sum()

def label_encoding(data):
    for ec in cfg.FORENCODINGCLOMNS:
        le = preprocessing.LabelEncoder()
        le.fit(data[ec])
        data[ec] = le.transform(data[ec])
    return data

def extract(data, x):
    return data.loc[:, x]