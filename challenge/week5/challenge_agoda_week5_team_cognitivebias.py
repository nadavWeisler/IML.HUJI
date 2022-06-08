from datetime import datetime
import sklearn.ensemble
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, AdaBoostClassifier, \
    GradientBoostingClassifier, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn import neural_network, preprocessing
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#pd.set_option('max_columns', None)

def one_hot_encoding(df, DUMMIES):
    """
    Create a design matrix from a dataframe using one-hot encoding.
    """
    for item in DUMMIES:
        df = pd.concat([df, pd.get_dummies(df[item], prefix=item)], axis=1)
        df.drop(item, axis=1, inplace=True)
    return df

def booleans(df, bools):
    for i in bools:
        df[df[i].isnull()] = 0
    return df

def handle_date(df, col_name):
    df[col_name + '_month'] = pd.to_numeric(df[col_name].dt.month)
    df[col_name + '_date_day'] = pd.to_numeric(df[col_name].dt.day)
    df[col_name + '_weekday'] = pd.to_numeric(df[col_name].dt.weekday)
    df[col_name + '_year'] = pd.to_numeric(df[col_name].dt.year)
    df[col_name + '_quarter'] = pd.to_numeric(df[col_name].dt.quarter)
    return df

def cancellation_policy_feature_extraction(df):
    # first parse out policy tiers separated by underscore
    cancel_pol = df['cancellation_policy_code'].copy(deep=True)
    cancel_pol[cancel_pol == "UNKNOWN"] = "0D0P"
    cancel_pol = cancel_pol.str.split(pat='_', expand=True)
    # for each policy tier, parse out days from penalty level (before and after the 'D')
    penalty = pd.DataFrame()
    for k, v in cancel_pol.iteritems():
        v[v.isnull()] = "0D0P"
        # get time between booking and cancellation-policy start time, 0 if policy is immediate
        if k < 2:
            days = v.apply(lambda x: x.split('D')[0] if x.find('D') != -1 else "0")
            proximity = df['sub_checkin_booking'] - pd.to_numeric(days)
            proximity[proximity < 0] = 0
            df[f'policy_proximity_absolute{k}'] = proximity
            df[f'policy_proximity_proportional{k}'] = proximity / (df['sub_checkin_booking'] + 0.1)
        # get penalty level as number between 1 (full price) and 0 (no penalty)
        # this standardizes between the two kinds of penalty units - percentage of price or no. of nights
        fines = v.apply(lambda x: x if x.find('D') == -1 else x.split('D')[1])
        quotient = df['sub_checkout_checkin'].copy(deep=True)
        quotient[fines.str.endswith('P')] = 100
        fines = fines.apply(lambda x: int(x.rstrip('NP')))
        penalty[k] = fines / quotient
    # make penalties cumulative
    temp1 = pd.concat([penalty[0], penalty[1]], join='outer', axis=1).copy(deep=True)
    temp2 = pd.concat([penalty[1], penalty[2]], join='outer', axis=1).copy(deep=True)
    penalty[1] = temp1.max(axis=1)
    penalty[2] = temp2.max(axis=1)
    # take maximum and minimum penalties as features
    df['max_penalty'] = penalty.max(axis=1)
    df['min_penalty'] = penalty.min(axis=1)
    # calculate a cancellation risk factor based on the policy
    df['cancellation_risk'] = (df['sub_checkin_booking'] - df['policy_proximity_absolute0']) * penalty[0] \
                              + (df['sub_checkin_booking'] - df['policy_proximity_absolute1']) * (
                                          penalty[1] - penalty[0]) \
                              + (penalty[2] - penalty[1])
    # clean up
    df.drop(['cancellation_policy_code'], axis=1, inplace=True)
    return df

def design_matrix(df):
    # dates
    df = handle_date(df, 'booking_datetime')
    df = handle_date(df, 'checkout_date')
    df = handle_date(df, 'checkin_date')
    # get date differences
    df['sub_checkout_checkin'] = (df['checkout_date'].dt.round('1d') - df['checkin_date'].dt.round(
        '1d')) / np.timedelta64(1, 'D')
    df['sub_checkin_booking'] = (df['checkin_date'].dt.round('1d') - df['booking_datetime'].dt.round(
        '1d')) / np.timedelta64(1, 'D')
    df[df['sub_checkin_booking'] == -1] = 0
    # add feature to check if booking includes weekend
    df['bus_days'] = np.busday_count(df['checkin_date'].values.astype('datetime64[D]'),
                                     df['checkout_date'].values.astype('datetime64[D]'))
    df['weekends'] = df['sub_checkout_checkin'] - df['bus_days']

    # booleans
    df['is_first_booking'] = df['is_first_booking'].astype(int)
    df['is_user_logged_in'] = df['is_user_logged_in'].astype(int)
    bools = ['request_nonesmoke',
             'request_earlycheckin',
             'request_highfloor',
             'request_largebed',
             'request_twinbeds',
             'request_airport',
             'request_latecheckin']
    df = booleans(df, bools)

    # one-hots
    DUMMIES = ["original_payment_type",
               "original_payment_method",
               "charge_option",
               "accommadation_type_name",
               "original_payment_currency",
               "customer_nationality",
               "origin_country_code",
               "hotel_country_code",
               "language"]
    df = one_hot_encoding(df, DUMMIES)

    # policy cancellation codes
    df = cancellation_policy_feature_extraction(df)

    # columns dropped
    df.drop(['h_booking_id',
             'hotel_id',
             'hotel_live_date',
             'h_customer_id',
             'hotel_city_code',
             'hotel_brand_code',
             'hotel_chain_code',
             'hotel_area_code',
             'guest_nationality_country_name',
             'booking_datetime',
             'checkout_date',
             'checkin_date'
             ], axis=1, inplace=True)

    return df

def get_labels(data):
    # the function extracts labels relevant for binary classification and for regression
    # it also converts all dates in the dataset to datetime format
    df = data.copy(deep=True)
    binary = df['cancellation_datetime'].isnull().astype(int).replace({0:1, 1:0})
    df['cancellation_datetime'] = df['cancellation_datetime'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    df['booking_datetime'] = df['booking_datetime'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    df['checkin_date'] = df['checkin_date'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    df['checkout_date'] = df['checkout_date'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    df['cancellation_datetime'].loc[~(binary.astype(bool))] = df['checkout_date']
    date_sub1 = (df['cancellation_datetime'].dt.round('1d') - df['booking_datetime'].dt.round('1d')) / np.timedelta64(1, 'D')
    df['cancellation_datetime'].loc[~(binary.astype(bool))] = df['booking_datetime']
    date_sub2 = (df['checkout_date'].dt.round('1d') - df['cancellation_datetime'].dt.round('1d')) / np.timedelta64(1, 'D')
    df.drop(['cancellation_datetime'], axis=1, inplace=True)
    return df, binary, date_sub1, date_sub2

def test_dates(data):
    df = data.copy(deep=True)
    df['booking_datetime'] = df['booking_datetime'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    df['checkin_date'] = df['checkin_date'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    df['checkout_date'] = df['checkout_date'].apply(lambda x: pd.to_datetime(x, format=r'%Y-%m-%d')).dt.round('1d')
    return df

def train_regressors(X, Y1, Y2):
    # normalize data
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # fit model
    #model1 = sklearn.neural_network.MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(400, 80), verbose=True)
    #model2 = sklearn.neural_network.MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(400, 80), verbose=True)
    model1 = ExtraTreesRegressor(random_state=42, verbose=True)
    model2 = ExtraTreesRegressor(random_state=42, verbose=True)
    #model1 = RandomForestRegressor(random_state=42, verbose=1)
    #model2 = RandomForestRegressor(random_state=42, verbose=1)
    #model1 = BaggingRegressor(verbose=1)
    #model2 = BaggingRegressor(verbose=1)
    #model1 = GradientBoostingRegressor(verbose=1)
    #model2 = GradientBoostingRegressor(verbose=1)
    model1.fit(X, Y1)
    model2.fit(X, Y2)
    print("First regressor R2 score on training set:", model1.score(X, Y1))
    print("Second regressor R2 score on training set:", model2.score(X, Y2))
    return model1, model2

def train_classifier(X, Y):
    # normalize data
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # fit model
    # model = sklearn.neural_network.MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(50, 10))
    # model = RandomForestClassifier(verbose=True)
    model = AdaBoostClassifier()
    model.fit(X, Y)
    return model

def regression_classifier(booking_dates, checkout_dates, test_matrix, regression_model1, regression_model2):
    # function receives:
    # the test data design matrix,
    # the booking dates column and checkout dates column in the original datetime format,
    # and two trained regression models that predict time from booking to cancellation and time from cancellation to checkout.
    # The function predicts a date of cancellation from the two directions and finds the average date between them,
    # then it checks if the predicted date is in the required interval (between 7/12/18 and 13/12/18)
    # if predicted date is in interval, function returns 1, otherwise 0
    # start with regressed prediction of cancellation from booking going forward
    df1 = pd.DataFrame({'delta': np.ceil(regression_model1.predict(test_matrix)), 'booking': booking_dates})
    df1['delta'] = df1['delta'].apply(lambda x: pd.Timedelta(days=x))
    df1['date_pred'] = df1.apply(lambda x: x['booking'] + x['delta'], axis=1)
    # now we do the same with regression from checkout date going backwards
    df2 = pd.DataFrame({'delta': np.ceil(regression_model2.predict(test_matrix)), 'checkout': checkout_dates})
    df2['delta'] = df2['delta'].apply(lambda x: pd.Timedelta(days=x))
    df2['date_pred'] = df2.apply(lambda x: x['checkout'] - x['delta'], axis=1)
    # now we take the mid-date between the two regressed dates and check if it is in the interval
    mid_date = (df1['date_pred'] - df2['date_pred'])/2 + df1['date_pred']
    df = pd.DataFrame({'predicted_values': mid_date.apply(lambda x: 1 if (x >= pd.to_datetime('2018-12-07')) & (x <= pd.to_datetime('2018-12-13')) else 0)})
    return df


if __name__ == '__main__':
    np.random.seed(42)
    # load training data
    train_set = pd.read_csv("agoda_cancellation_train.csv")
    train_set, labels_class, labels_reg1, labels_reg2 = get_labels(train_set)
    features = design_matrix(train_set)

    """
    # load weeks 1-4 for evaluation
    test_set1 = pd.read_csv("../challenge/test_set_week_1.csv")
    test_set2 = pd.read_csv("../challenge/test_set_week_2.csv")
    test_set3 = pd.read_csv("../challenge/test_set_week_3.csv")
    test_set4 = pd.read_csv("../challenge/test_set_week_4.csv")
    test_set1_4 = pd.concat([test_set1, test_set2, test_set3, test_set4])
    test_set1_4 = test_dates(test_set1_4)
    bookingdates = test_set1_4['booking_datetime'].copy(deep=True)
    checkoutdates = test_set1_4['checkout_date'].copy(deep=True)
    features_weeks1_4 = design_matrix(test_set1_4)
    true_labels1 = pd.read_csv("../challenge/true_labels_week_1.csv")
    true_labels2 = pd.read_csv("../challenge/true_labels_week_2.csv")
    true_labels3 = pd.read_csv("../challenge/true_labels_week_3.csv")
    true_labels4 = pd.read_csv("../challenge/true_labels_week_4.csv")
    true_labels1_4 = pd.concat([true_labels1, true_labels2, true_labels3, true_labels4])
    """

    # load test data for week 5
    test_set5 = pd.read_csv("test_set_week_5.csv")
    test_set5 = test_dates(test_set5)
    bookingdates = test_set5['booking_datetime'].copy(deep=True)
    checkoutdates = test_set5['checkout_date'].copy(deep=True)
    test_matrix = design_matrix(test_set5)
    #true_labels4 = pd.read_csv("../challenge/true_labels_week_4.csv")

    # align one-hots
    # features, test = features.align(features_weeks1_4, join='inner', axis=1)
    features, test = features.align(test_set5, join='inner', axis=1)

    # train model
    # two regression models on training data
    regressor1, regressor2 = train_regressors(features, labels_reg1, labels_reg2)
    # classifier on training data
    # classifier = train_classifier(features, labels_class)

    # predict
    scaler = StandardScaler().fit(features)
    test = scaler.transform(test)
    final_pred = regression_classifier(bookingdates, checkoutdates, test, regressor1, regressor2)
    #final_pred = pd.DataFrame({'prediction': classifier.predict(test)})

    #evaluate
    #print("evaluation:", classification_report(true_labels1_4, final_pred))

    # prepare prediction for submission
    final_pred.to_csv("316493758_208770149_318031010.csv", index=False)

