import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network, preprocessing
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.set_option('max_columns', None)


def design_matrix(df, isTest=False):

    # one-hot encodings

    # label encodings
    le_original_payment_type = preprocessing.LabelEncoder()
    le_original_payment_type.fit(df.original_payment_type)
    df.original_payment_type = le_original_payment_type.transform(df.original_payment_type.values)

    le_original_payment_method = preprocessing.LabelEncoder()
    le_original_payment_method.fit(df.original_payment_method)
    df.original_payment_method = le_original_payment_method.transform(df.original_payment_method.values)

    le_charge_option = preprocessing.LabelEncoder()
    le_charge_option.fit(df.charge_option)
    df.charge_option = le_charge_option.transform(df.charge_option.values)

    le_hotel_brand_code = preprocessing.LabelEncoder()
    le_hotel_brand_code.fit(df.hotel_brand_code)
    df.hotel_brand_code = le_hotel_brand_code.transform(df.hotel_brand_code.values)

    le_hotel_chain_code = preprocessing.LabelEncoder()
    le_hotel_chain_code.fit(df.hotel_chain_code)
    df.hotel_chain_code = le_hotel_chain_code.transform(df.hotel_chain_code.values)

    le_hotel_area_code = preprocessing.LabelEncoder()
    le_hotel_area_code.fit(df.hotel_area_code)
    df.hotel_area_code = le_hotel_area_code.transform(df.hotel_area_code.values)

    le_accommadation_type_name = preprocessing.LabelEncoder()
    le_accommadation_type_name.fit(df.accommadation_type_name)
    df.accommadation_type_name = le_accommadation_type_name.transform(df.accommadation_type_name.values)

    le_original_payment_currency = preprocessing.LabelEncoder()
    le_original_payment_currency.fit(df.original_payment_currency)
    df.original_payment_currency = le_original_payment_currency.transform(df.original_payment_currency.values)

    le_customer_nationality = preprocessing.LabelEncoder()
    le_customer_nationality.fit(df.customer_nationality)
    df.customer_nationality = le_customer_nationality.transform(df.customer_nationality.values)

    le_cancellation_policy_code = preprocessing.LabelEncoder()
    le_cancellation_policy_code.fit(df.cancellation_policy_code)
    df.cancellation_policy_code = le_cancellation_policy_code.transform(df.cancellation_policy_code.values)

    le_origin_country_code = preprocessing.LabelEncoder()
    le_origin_country_code.fit(df.origin_country_code)
    df.origin_country_code = le_origin_country_code.transform(df.origin_country_code.values)

    le_hotel_country_code = preprocessing.LabelEncoder()
    le_hotel_country_code.fit(df.hotel_country_code)
    df.hotel_country_code = le_hotel_country_code.transform(df.hotel_country_code.values)

    # booleans
    df['is_first_booking'] = df['is_first_booking'].astype(int)
    df['is_user_logged_in'] = df['is_user_logged_in'].astype(int)
    df[df.request_nonesmoke.isnull()] = 0
    df[df.request_earlycheckin.isnull()] = 0
    df[df.request_highfloor.isnull()] = 0
    df[df.request_largebed.isnull()] = 0
    df[df.request_twinbeds.isnull()] = 0
    df[df.request_airport.isnull()] = 0
    df[df.request_latecheckin.isnull()] = 0

    # columns dropped
    df.drop(['h_booking_id',
             'hotel_id',
             'hotel_live_date',
             'h_customer_id',
             'language',
             'hotel_city_code',
             'guest_nationality_country_name'
             ], axis=1, inplace=True)

    # dates
    def handle_date(col_name):
        df[col_name] = df[col_name].apply(lambda x: np.nan if x == '0000-00-00' or x== np.nan else x)
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        if col_name != 'cancellation_datetime':
            df[col_name + '_month'] = pd.to_numeric(df[col_name].dt.month)
            df[col_name + '_date_day'] = pd.to_numeric(df[col_name].dt.day)
            df[col_name + '_weekday'] = pd.to_numeric(df[col_name].dt.weekday)
            df[col_name + '_year'] = pd.to_numeric(df[col_name].dt.year)
            df[col_name + '_quarter'] = pd.to_numeric(df[col_name].dt.quarter)
        # print(col_name)
        # print(df[col_name])
        # # if(col_name!='cancellation_datetime'):
        # #     df[col_name + '_epoch'] = df[col_name].apply(lambda x: pd.Timestamp.timestamp(x) - 1500000000 )

    handle_date('booking_datetime')
    handle_date('checkout_date')
    handle_date('checkin_date')
    if not isTest:
        handle_date('cancellation_datetime')

    # get date differences
    df['sub_checkout_checkin'] = (df['checkout_date'].dt.round('1d') - df['checkin_date'].dt.round('1d')) / np.timedelta64(1, 'D')
    df['sub_checkin_booking'] = (df['checkin_date'].dt.round('1d') - df['booking_datetime'].dt.round('1d')) / np.timedelta64(1, 'D')
    df[df['sub_checkin_booking'] == -1] = 0

    # add feature to check if booking includes weekend
    df['bus_days'] = np.busday_count(df['checkin_date'].values.astype('datetime64[D]'), df['checkout_date'].values.astype('datetime64[D]'))
    df['weekends'] = df['sub_checkout_checkin'] - df['bus_days']

    # clean up
    # df.drop(['booking_datetime', 'checkout_date', 'checkin_date'], axis=1, inplace=True)

    # df['sub_cancel_checkin'] = (df['cancellation_date'].dt.round('1d') - df['checkin_date'].dt.round('1d')) / np.timedelta64(1, 'D')

    return df


def train_classifier_model(X, Y):
    # normalize data
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    #fit model
    model = sklearn.neural_network.MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(50,10))
    model.fit(X, Y)
    
    # model = neural_network.MLPRegressor

    return model

def train_regressor_model(X, Y):
    # normalize data
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    #fit model
    model = sklearn.neural_network.MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(50,10))
    model.fit(X, Y)
    
    # model = neural_network.MLPRegressor

    return model

# Get features
train_set = pd.read_csv("../datasets/agoda_cancellation_train.csv")
labels = train_set['cancellation_datetime'].isnull().astype(int).replace({0:1, 1:0})
# train_set.drop(['cancellation_datetime'], axis=1, inplace=True)
features = design_matrix(train_set)
features['cancellation_datetime'] = pd.to_datetime(features['cancellation_datetime'], errors='coerce')
features['checkout_date'] = pd.to_datetime(features['checkout_date'], errors='coerce')
features['checkin_date'] = pd.to_datetime(features['checkin_date'], errors='coerce')
features['booking_datetime'] = pd.to_datetime(features['booking_datetime'], errors='coerce')

# cancellation date - booking date
def get_cancellation_date_diff(df): 
    date_sub = (df[df['cancellation_datetime'].notna()]['cancellation_datetime'] - df[df['cancellation_datetime'].notna()]['booking_datetime'])
    # print(date_sub.dt.days)
    no_cancel = df['cancellation_datetime'].isnull().astype(int)
    no_cancel = no_cancel.replace({0:-1, 1:0})
    # merge date_sub with no_cancel where date_sub is defined
    return pd.concat([no_cancel, date_sub.dt.days], axis=1).max(axis=1)


# SET LABELS
labels = get_cancellation_date_diff(features)

features.drop(['cancellation_datetime','booking_datetime', 'checkout_date', 'checkin_date'], axis=1, inplace=True)

# trained_model = train_regressor_model(features, labels)

test_data = pd.read_csv("test_set_week_2.csv")

test_data = design_matrix(test_data, isTest=True)
test_data['checkout_date'] = pd.to_datetime(test_data['checkout_date'], errors='coerce')
test_data['checkin_date'] = pd.to_datetime(test_data['checkin_date'], errors='coerce')
test_data['booking_datetime'] = pd.to_datetime(test_data['booking_datetime'], errors='coerce')
predictions = [0 for i in range(len(test_data))]
df = pd.DataFrame(predictions)
df.to_csv("predictions.csv", index=False)


# # Model accuracy
# y_class_pred= model_classifier.predict(X_test_classifier)
# # print(f'Accuracy of classifier model: {accuracy_score(y_test_classifier, y_pred)}')
# y_reg_predict = model_regressor.predict(X_test_regressor)>0.5
# # accuracy of regressor model
# print(f'Accuracy of regressor model: {accuracy_score(y_test_regressor, y_reg_predict)}')
# sklearn.metrics.mean_squared_error(y_test_regressor, model_regressor.predict(X_test_regressor))

# print(model_regressor.predict(X_test_regressor))

