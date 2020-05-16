
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

def blight_model(): 
    
    def time(hearing_date, ticket_issued):
        if not hearing_date or type(hearing_date) != str:
            return 73
        hearing = datetime.strptime(hearing_date, '%Y-%m-%d %H:%M:%S')
        ticket = datetime.strptime(ticket_issued, '%Y-%m-%d %H:%M:%S')
        
        void = hearing - ticket
        return void.days

    
    training = pd.read_csv('train.csv', encoding = 'ISO-8859-1')
    testing = pd.read_csv('test.csv')
    training = training[(training['compliance'] == 0) | (training['compliance'] ==1)]
    geolocation = pd.read_csv('latlons.csv')
    address = pd.read_csv('addresses.csv')
    address = address.set_index('address').join(geolocation.set_index('address'), how = 'left')
    training = training.set_index('ticket_id').join(address.set_index('ticket_id'))
    testing = testing.set_index('ticket_id'). join(address.set_index('ticket_id'))
    
    
    
    training = training[~training['hearing_date'].isnull()]
    training['time_gap'] = training.apply(lambda row: time(row['hearing_date'],
                                                    row['ticket_issued_date']),
                                   axis = 1)
    testing['time_gap'] = testing.apply(lambda row: time(row['hearing_date'],
                                                    row['ticket_issued_date']),
                                   axis = 1)
    feature_split = ['agency_name', 'state','disposition']
    training.lat.fillna(method = 'pad', inplace = True)
    training.lon.fillna(method='pad', inplace = True)
    training.state.fillna(method = 'pad', inplace =True)
    
    testing.state.fillna(method = 'pad', inplace = True)
    testing.lat.fillna(method = 'pad', inplace = True)
    testing.lon.fillna(method = 'pad', inplace = True)
    
    training = pd.get_dummies(training, columns = feature_split)
    testing = pd.get_dummies(testing, columns = feature_split)
    
    remove_train = ['payment_status','payment_date',
                   'balance_due','collection_status',
                   'compliance_detail']
    remove_both = ['fine_amount', 'violator_name', 'zip_code',
                   'country', 'city','inspector_name',
                   'violation_street_number',
                   'violation_street_name',
                   'violation_zip_code',
                   'violation_description',
                   'mailing_address_str_number', 
                   'mailing_address_str_name',
                   'non_us_str_code','ticket_issued_date',
                   'hearing_date', 'grafitti_status',
                   'violation_code']
    
    training.drop(remove_train, axis = 1, inplace = True)
    training.drop(remove_both, axis =1 , inplace = True)
    testing.drop(remove_both, axis = 1 , inplace = True)
    
    
    feature_train = training.columns.drop('compliance')
    feature_train_set = set(feature_train)
    
    for feature in set(feature_train):
        if feature not in testing:
            feature_train_set.remove(feature)
    feature_train = list(feature_train_set)
    
    x_train = training[feature_train]
    y_train = training.compliance
    x_test = testing[feature_train]
    
    scaler = MinMaxScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.fit_transform(x_test)
    
    
    clf = MLPClassifier(hidden_layer_sizes = [100,20], alpha = 6,
                       random_state = 0, solver = 'lbfgs',
                       verbose = 0)
    clf.fit(x_train_scale, y_train)
    
    test_prob = clf.predict_proba(x_test_scale)[:,1]
    
    df_testing = pd.read_csv('test.csv', encoding = 'ISO-8859-1')
    df_testing['compliance'] = test_prob
    df_testing.set_index('ticket_id', inplace = True)
    
    
    return df_testing.compliance    
        

#Seeing the model output
model = blight_model()
model = model.to_frame().reset_index()
model



