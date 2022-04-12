import sklearn.neural_network
import sklearn.linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to Agoda training data

    Returns
    -------
    Design matrix as pandas.DataFrame and Series with corresponding response vector
    """

    df = pd.read_csv(filename)

    # one-hot encodings
    df = pd.concat([df, pd.get_dummies(df['hotel_country_code'], prefix='hotel_country_', dummy_na=True)],
                   axis=1).drop(['hotel_country_code'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['accommadation_type_name'], prefix='accommodation_', dummy_na=True)],
                   axis=1).drop(['accommadation_type_name'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['charge_option'], prefix='charge_', dummy_na=True)],
                   axis=1).drop(['charge_option'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['original_payment_currency'], prefix='currency_', dummy_na=True)],
                   axis=1).drop(['original_payment_currency'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['original_payment_type'], prefix='paytype_', dummy_na=True)],
                   axis=1).drop(['original_payment_type'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['original_payment_method'], prefix='paymethod_', dummy_na=True)],
                   axis=1).drop(['original_payment_method'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['language'], prefix='language_', dummy_na=True)],
                   axis=1).drop(['language'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['origin_country_code'], prefix='orig_country_', dummy_na=True)],
                   axis=1).drop(['origin_country_code'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['guest_nationality_country_name'], prefix='guest_nationality_', dummy_na=True)],
                   axis=1).drop(['guest_nationality_country_name'], axis=1)

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
             'hotel_area_code',
             'hotel_brand_code',
             'hotel_chain_code',
             'hotel_city_code',
             'customer_nationality'], axis=1, inplace=True)

    # dates
    def handle_date(col_name):
        df[col_name] = df[col_name].apply(lambda x: np.nan if x == '0000-00-00' else x)
        df[col_name] = pd.to_datetime(df[col_name])
        df[col_name + '_month'] = pd.to_numeric(df[col_name].dt.month)
        df[col_name + '_date_day'] = pd.to_numeric(df[col_name].dt.day)
        df[col_name + '_weekday'] = pd.to_numeric(df[col_name].dt.weekday)
        df[col_name + '_year'] = pd.to_numeric(df[col_name].dt.year)
        df[col_name + '_quarter'] = pd.to_numeric(df[col_name].dt.quarter)
        df[col_name + '_epoch'] = df[col_name].apply(lambda x: pd.Timestamp.timestamp(x) - 1500000000)

    handle_date('booking_datetime')
    handle_date('checkout_date')
    handle_date('checkin_date')

    # clean duds
    df = df[df['booking_datetime_year'] != 1970]

    # get date differences
    df['sub_checkout_checkin'] = (df['checkout_date'].dt.round('1d') - df['checkin_date'].dt.round('1d')) / np.timedelta64(1, 'D')
    df['sub_checkin_booking'] = (df['checkin_date'].dt.round('1d') - df['booking_datetime'].dt.round('1d')) / np.timedelta64(1, 'D')
    df[df['sub_checkin_booking'] == -1] = 0

    # add feature to check if booking includes weekend
    df['bus_days'] = np.busday_count(df['checkin_date'].values.astype('datetime64[D]'), df['checkout_date'].values.astype('datetime64[D]'))
    df['weekends'] = df['sub_checkout_checkin'] - df['bus_days']

    # clean up
    df.drop(['booking_datetime', 'checkout_date', 'checkin_date'], axis=1, inplace=True)

    # cancellation policy
    # first parse out policy tiers separated by underscore
    cancel_pol = df['cancellation_policy_code'].copy(deep=True)
    cancel_pol[cancel_pol == "UNKNOWN"] = "0D0P"
    cancel_pol = cancel_pol.str.split(pat='_', expand=True)
    # for each policy tier, parse out days (before 'D') from penalty level (after 'D')
    penalty = pd.DataFrame()
    for k, v in cancel_pol.iteritems():
        v[v.isnull()] = "0D0P"
        # get how close cancellation policy start time is to booking, 0 if immediate
        if k < 2:
            days = v.apply(lambda x: x.split('D')[0] if x.find('D') != -1 else "0")
            proximity = df['sub_checkin_booking'] - pd.to_numeric(days)
            proximity[proximity < 0] = 0
            df[f'policy_proximity_absolute{k}'] = proximity
            df[f'policy_proximity_proportional{k}'] = proximity / (df['sub_checkin_booking'] + 0.1)
        # get penalty level as number between 1 (full price) and 0 (no penalty)
        # requires standardizing percentage and night measurement units
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
                                + (df['sub_checkin_booking'] - df['policy_proximity_absolute1']) * (penalty[1] - penalty[0]) \
                                + (penalty[2] - penalty[1])
    # clean up
    df.drop(['cancellation_policy_code'], axis=1, inplace=True)

    labels = df['cancellation_datetime'].isnull().astype(int)
    df.drop(['cancellation_datetime'], axis=1, inplace=True)

    return df, labels


def train_model(X, Y):

    # split training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # normalize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # fit model
    model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(150, 80, 10), activation='relu', solver='adam',
                                                 alpha=0.0001, batch_size=64, max_iter=200, shuffle=True,
                                                 random_state=0, tol=0.0001, verbose=True, n_iter_no_change=10,
                                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # model = sklearn.linear_model.LogisticRegression(solver='saga', verbose=1)

    # model = KNeighborsClassifier(n_neighbors=15)

    # model = RandomForestClassifier()

    # model = sklearn.linear_model.SGDClassifier()

    model.fit(X_train, y_train)
    # test on validation set
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("accuracy on validation set:", score)

    return model, score


def predict(model, test):
    df = pd.read_csv(test)
    predict = model.predict(df)
    return predict


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    features, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    #features.to_csv("../challenge/agoda_design_matrix1.csv", index=False)
    #cancellation_labels.to_csv("../challenge/agoda_train_labels.csv", index=False)

    # train model and evaluate on validation set
    classifier, accuracy = train_model(features, cancellation_labels)

    #test_predictions = predict(classifier, "../challenge/test_set_week_1.csv")
    #test_predictions.to_csv("../challenge/id1_id2_318031010.csv")


"""

    Submission instructions:

        Export to specified file the prediction results of given estimator on given testset.

        Binary output - "cancel"/"not-cancel"

        File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
        predicted values.
        
        The file name should be id1_id2_id3.csv

        One team member submits the prediction csv + jupyter notebook or .py file with code in the Challenge Week 1 Submission
    
    """
