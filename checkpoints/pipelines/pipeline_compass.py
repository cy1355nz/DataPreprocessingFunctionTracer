import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def COMPASS_pipeline(f1_path = '../data/compass/demographic.csv',f2_path = '../data/compass/jailrecord1.csv',f3_path = '../data/compass/jailrecord2.csv'):
    '''
    This pipeline specifically works for COMPAS Recidivism Algorithm problem.
    '''
    #read csv files
    df = pd.read_csv(f1_path)
    df1 = pd.read_csv(f2_path)
    df2 = pd.read_csv(f3_path)
    
    #drop columns inplace
    df.drop(columns=['Unnamed: 0','age_cat'],inplace=True)
    df1.drop(columns=['Unnamed: 0'],inplace=True)
    df2.drop(columns=['Unnamed: 0'],inplace=True)

    #JOIN dataframes column-wise and row-wise
    data = pd.concat([df1,df2],ignore_index=True)
    data = pd.merge(df, data, on=['id','name'])

    #drop rows that miss a few important features
    data = data.dropna(subset=['id', 'name','is_recid','days_b_screening_arrest','c_charge_degree','c_jail_out','c_jail_in'])

    #generate a new column conditioned on existed column
    data['age_cat'] = data.apply(lambda row:'<25' if row['age'] < 25 else '>45' if row['age']>45 else '25-45', axis=1)

    #PROJECTION
    data = data[['sex', 'dob','age','c_charge_degree', 'age_cat', 'race','score_text','priors_count','days_b_screening_arrest',\
                 'decile_score','is_recid','two_year_recid','c_jail_in','c_jail_out']]

    #SELECT based on some conditions
    data = data.loc[(data['days_b_screening_arrest'] <= 30)]
    data = data.loc[(data['days_b_screening_arrest'] >= -30)]
    data = data.loc[(data['is_recid'] != -1)]
    data = data.loc[(data['c_charge_degree'] != "O")]
    data = data.loc[(data['score_text'] != 'N/A')]
    # create a new feature 
    data['c_jail_out'] = pd.to_datetime(data['c_jail_out']) 
    data['c_jail_in'] = pd.to_datetime(data['c_jail_in']) 
    data['length_of_stay'] = data['c_jail_out'] - data['c_jail_in']
    #specify categorical and numeric features
    categorical = ['sex', 'c_charge_degree', 'age_cat', 'race', 'score_text', 'is_recid',
           'two_year_recid']
    numeric1 = ['age','priors_count', 'decile_score']
    numeric2 = ['days_b_screening_arrest','length_of_stay']

    #sklearn pipeline
    impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')), 
                                   ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')), 
                                   ('bin_discretizer', KBinsDiscretizer(n_bins=4, encode='uniform', strategy='uniform')])
    featurizer = ColumnTransformer(transformers=[
            ('impute1_and_onehot', one_hot_and_impute, categorical),
            ('impute2_and_bin', one_hot_and_impute, numeric1),
            ('std_scaler', StandardScaler(), numeric2),
        ])
                               
    pipeline = Pipeline([
        ('features', featurizer),
        ('learner', LogisticRegression())
    ])
    return pipeline
