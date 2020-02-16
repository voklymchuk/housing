#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip uninstall tensorflow


# In[ ]:





# In[ ]:


# pip install tensorflow-gpu==2.1


# In[1]:


#!pip install hyperopt datawig lightgbm pyod xgboost mlxtend category-encoders tensorflow


# ### Download the Data

# In[1]:


import os
import pandas as pd
import requests

# dataset is located in current directory
HOUSING_PATH = os.getcwd()

# function to load training dataset


def load_housing_data(filename="train.csv"):
    import numpy as np
    csv_path = os.path.join(HOUSING_PATH, filename)
    df = pd.read_csv(csv_path, dtype={'GarageYrBlt': np.float32, 'YearBuilt': np.float32, 'YrSold': np.float32, 'YearRemodAdd': np.float32,
                                      'LotFrontage': np.float32})

    return df


# In[ ]:





# In[2]:


# Read files
train = load_housing_data('train.csv')
test = load_housing_data('test.csv')

# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)


# ## Drop outliers records

# In[3]:


# From EDA obvious outliers
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])

# typo in test set:
test.loc[test['GarageYrBlt']>test['YrSold'], 'GarageYrBlt'] = test[test['GarageYrBlt']>test['YrSold']]['YearBuilt']


# In[4]:


max(test['GarageYrBlt'])


# In[5]:


# display histogram for each numerical attribute
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
train.hist(bins=50, figsize=(20,20))
plt.show()


# ## Prepare Data for Machine Learning Algorithms

# Let's revert to a clean training set and let's separate the predictors and the labels since we do not want to apply same transformations to the predictors and the target value

# In[6]:


#train_labels = train["SalePrice"]
#train = train.drop(["SalePrice"], axis=1) # drop labels for training set
import numpy as np
# Apply transformation

# New prediction
y_train = train.SalePrice.values
y_train_orig = train.SalePrice
y=train['SalePrice']
train.drop('SalePrice', axis = 1, inplace = True)


# In[7]:


train.shape


# ### Data Cleaning

# In[8]:


#missing data
import pandas as pd
data_features = pd.concat((train, test),sort=False).reset_index(drop=True)

print(data_features.shape)


data_features_na = data_features.isnull().sum()
data_features_na = data_features_na[data_features_na>0]
data_features_na.sort_values(ascending=False)


# In[9]:


#missing data percent plot
total = data_features.isnull().sum().sort_values(ascending=False)
percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[10]:


data_features.loc[data_features['YrSold']<data_features['YearBuilt'], ['YearBuilt','YrSold']]


# In[11]:


import pandas as pd
import numpy as np
import datawig
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import norm, skew

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_sparse = True, cor_cols_to_drop = []):
        """Impute missing values.
              Columns of dtype object are imputed with "NA" value 
                in column.
              Columns of other types are imputed with zeros,
        """
        self.cor_cols_to_drop = cor_cols_to_drop
        self.drop_sparse = drop_sparse
        
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].mode()[0]  if (X[c]).values.dtype == np.dtype('O') else 0 for c in X],index=X.columns)
        return self

    def transform(self, X, y=None):
        # encode categorical labels as ordinal features 
        # to avoid typos in dataset
        X.loc[X['GarageYrBlt']>X['YrSold'], 'GarageYrBlt'] = X[X['GarageYrBlt']>X['YrSold']]['YrSold']
        X.loc[X['YrSold']<X['YearBuilt'], 'YearBuilt'] = X[X['YrSold']<X['YearBuilt']]['YrSold']
        # Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string        
        
        for var in ['Exterior1st','Exterior2nd','SaleType']:
            if var in X.columns:
                X[var] = X[var].fillna('Other')
        
        for var in ['SaleType']:
            if var in X.columns:
                X[var] = X[var].fillna('Oth')        
        
        common_vars = ['Electrical','KitchenQual']
        for var in common_vars:
            if var in X.columns:
                X[var] = X[var].fillna(X[var].mode()[0])
    
        # 'RL' is by far the most common value. So we can fill in missing values with 'RL'
        if 'MSZoning' in X.columns:
            #X['MSZoning'] = X.groupby(['Neighborhood','MSSubClass'])['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
            X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
            #X[X['MSZoning']== "C (all)"] = 'C' 
        
        
        # # data description says NA means "No Pool", majority of houses have no Pool at all in general.
        # features[] = features["PoolQC"].fillna("None")
        # Replacing missing data with None
        for col in ['PoolQC','Fence','MiscFeature','GarageType','GarageFinish','GarageQual', 'GarageCond','FireplaceQu','BsmtFinType1', 'BsmtFinType2', 'BsmtCond','BsmtExposure', 'BsmtQual', 'Alley']:
            if col in X.columns:
                X[col] = X[col].fillna('NA')
        
        for col in ['MasVnrType','Utilities']:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        
        for col in ['Utilities']:
            if col in X.columns:
                X[col] = X.groupby(['MSSubClass','MSZoning'])['Utilities'].transform(lambda x: x.fillna(x.mode()[0]))
                    
        # Replacing missing data with 0 (Since No garage = no cars in such garage.)
        for col in ('GarageYrBlt','GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):
            if col in X.columns:
                X[col] = X[col].fillna(0)
        
        #for col in ['GarageYrBlt']:
        #    if col in X.columns:
        #        X[col] = X[col].fillna(round(X[col].mean())-100)

        # group by neighborhood and fill in missing value by the mean, median LotFrontage of the neighborhood
        if ('LotFrontage' in X.columns) and ('Neighborhood' in X.columns):
            X['LotFrontage'] = X.replace( object ,np.nan, regex=True)
            # impute with mean LotFrintage value per LotConfig for each neighborhood
            X['LotFrontage'] = X.groupby(['Neighborhood','LotConfig'])['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
            # impute the rest by the median LotFrontage of a Neighborhood
            X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
            #print('Features size:', X.shape)    
        
        # data description says NA means typical
        if 'Functional' in X.columns:
            X['Functional'] = X['Functional'].fillna('Typ')
        
        # impute missing values with zeros and Mode
        #X = (X.fillna(self.fill))
        
        #categorical_features = X.select_dtypes(include=['object']).columns
        #numerical_features = X.select_dtypes(exclude = ["object"]).columns.to_list()
        #X[numerical_features]=X[numerical_features].astype(np.float32) 
        
        return X
    
    
    


# In[12]:


pd.options.mode.chained_assignment = None

#lets separate numerical and categorical variables
temp = DataFrameImputer().fit_transform(data_features)

num_attribs = (temp.select_dtypes(include=[np.number]).columns.to_list())

cat_attribs = temp.select_dtypes(exclude=[np.number]).columns.to_list()


# In[13]:


# missing data including validation set
total = temp.isnull().sum().sort_values(ascending=False)
percent = (temp.isnull().sum()/temp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[ ]:





# ### Custom Transformers

# In[14]:


from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import skew 

def add_extra_features(X, add_bedrooms_per_room=True):
    # transform numeric columns
    #encode ordinal features
    cleanup_nums = { 
            "PoolQC": {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4, 'NA': 0},
            "PavedDrive": {"N": 0, "P": 1, "Y": 2},
            "GarageCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5, "NA": 0},
            "GarageQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5, "NA": 0},
            "Utilities": {"ELO":1, "NoSeWa":2, "NoSewr":3, "AllPub": 4, "None": 0},
            "FireplaceQu": {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5, "NA": 0},                
            "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd":3, "Ex":4},                
            "CentralAir": {"N": 0, "Y": 1},                 
            "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, "NA": 0},                                
            "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd":4, "NA": 0},                
            "BsmtCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5, "NA": 0},
            "BsmtQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5, "NA": 0},
            "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd":3, "Ex":4},
            "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd":3, "Ex":4},
            "GarageFinish": {"Unf":1, "RFn":2, "Fin":3, "NA": 0},
            "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, "NA": 0},
            "Electrical": {"Mix": 0, "FuseP": 0, "FuseF": 1, "FuseA":3, "SBrkr":4}, 
            "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5,"Min1": 6,"Typ": 7}, 
            "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd":3, "Ex":4},
            }
    
    X = X.replace(cleanup_nums)
    
    
    X['TotQual'] = X["GarageCond"]+X["BsmtCond"]+X["BsmtQual"]+X["GarageQual"]
    
        
    X['TotCond'] = X["ExterCond"]+X['Functional']+X['HeatingQC']+X["KitchenQual"]
   
    X['Overall'] = X['OverallQual']*X['OverallCond']
    
    X['GarageAge'] = X['YrSold'] - X['GarageYrBlt']
    
    X['Age'] = X['YrSold'] - X['YearBuilt']
    X['RemodAddAge'] = X['YrSold'] - X['YearRemodAdd']
    X["total_SF_per_bedroom"] = (X['TotalBsmtSF'] - X['BsmtUnfSF'] + X['1stFlrSF'] + X['2ndFlrSF']) / (X["BedroomAbvGr"]+ 0.1)
    
    X["garage_cars_per_bedroom"] = np.log(X["GarageCars"] / (X["BedroomAbvGr"]+ 0.5)+1)
    X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) + X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
    X["bathrooms_per_bedroom"] = np.log(X["Total_Bathrooms"] / (X["BedroomAbvGr"]+ 0.5)+1)
    
    
    
    #Numeric to categorical mappings:
    #X['Age'] = pd.cut(X['Age'], bins = 5, labels = ['AgeNew', 'Age1', 'Age2', 'Age3', 'AgeOld']).astype(str)
    
    # create aditional features for spikes in our dataset
    #X['has2ndflr'] = X['2ndFlrSF'].apply(lambda row: 1 if row > 0 else 0)
    X['TotalFinSF'] =  X['TotalBsmtSF']- X['BsmtUnfSF'] + X['1stFlrSF'] + X['2ndFlrSF']-X['LowQualFinSF']
    
    X['Garage'] = X['GarageCars']+X['GarageFinish']+X['GarageQual']
        
    X['Total_porch_sf'] = (X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF'])
    #X = X.drop(['OpenPorchSF','3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])
    
    
    #X['haspool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    #X['hasgarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    #X['hasbsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    #X['hasfireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
 
    str_vars = ['MSSubClass','YrSold','MoSold']
    for var in str_vars:
        if var in X.columns:
            X[var] = X[var].apply(str)
    
    # Plot skew value for each numerical value
    from scipy.stats import skew 
    
    num_cols = X.select_dtypes(include=[np.number]).columns.to_list()
    #X[num_cols]=X[num_cols].astype(np.float32)
    
    skewness = X[num_cols].apply(lambda x: skew(x))
    skewness.sort_values(ascending=False)
    
    skewness = skewness[abs(skewness) > 0.5]
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    
    for feat in skewed_features:
        if feat == 'YearBuilt' or feat == 'GarageYrBlt' or feat =='YrSold' or feat == 'LotArea ' or feat == '1stFlrSF ':
            #b = boxcox1p(X[feat], boxcox_normmax(X[feat] + 1))
            pass#X[feat] = np.log(X[feat]+1)#b#(b/b.max())*np.finfo(np.float32).max/2
        else:
            try:
                X[feat] = boxcox1p(X[feat], boxcox_normmax(X[feat] + 1))
            except:
                print(feat)
                print(min(X[feat]))
                X[feat] = boxcox1p(X[feat], boxcox_normmax(X[feat] + 1))
  
    
    #replace zero values in folowing columns with sample from original distribution
    #for col in ['LotFrontage']:##['BsmtFinType1','BsmtUnfSF','2ndFlrSF','OpenPorchSF','WoodDeckSF','GarageArea', 'MasVnrArea','TotalBsmtSF', 'BsmtFinSF1']:
    #    tmp  = r[r[col]!= 0]
    #    n = (r[col]==0).sum()
    #    values = tmp.sample(frac=n/len(tmp.index), replace=True)[col].values
    #    r[col][r[col]==0] = values.reshape(-1,1)
    
    # Not normaly distributed can not be normalised and has no central tendecy
    for col in ['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF',
                '3SsnPorch', 'BsmtFinSF2', 'BsmtCond','BsmtFinType2','BsmtHalfBath', 
                'EnclosedPorch', 'Functional', 'GarageCond', 'LowQualFinSF','MiscVal',
                'PavedDrive','PoolArea', 'PoolQC','ScreenPorch','Utilities','BsmtExposure',
                'ExterCond','HeatingQC', 'KitchenAbvGr', 'Electrical', 'ExterCond','ExterQual',
                'FireplaceQu', 'hasPool','GarageCars','GarageFinish','GarageQual', 'YearBuilt',
                'GarageYrBlt','YearRemodAdd'
                ]:
        if col in X.columns:
            X = X.drop(col, axis=1)
            
    #num_to_drop = ["PavedDrive","HalfBath", "GarageYrBlt","YearBuilt","YearRemodAdd", "ScreenPorch", "PoolQC", "PoolArea", "Fence", "Utilities", "MiscVal", "LowQualFinSF", "GarageQual", "GarageCond", "EnclosedPorch", "ExterCond", "Electrical", "CentralAir", "BsmtHalfBath", "BsmtFinType2", "BsmtFinSF2", "BsmtCond", "Alley", "3SsnPorch"]       
    #if self.drop_sparse:
    #    r.drop(num_to_drop, axis=1, inplace=True)
            
    #correlated_columns_to_drop = ['TotalBsmtSF','GarageAge','FireplaceQu','TotRmsAbvGrd','totalFlrSF','RemodAddAge','GrLivArea','ExterQual','FullBath']
    #cat_to_drop = ['Street','LandContour']
    
    
    
    #X['YrSold']=X['YrSold'].astype(object) 
    #categorical_features = X.select_dtypes(include=['object']).columns
    #numerical_features = X.select_dtypes(exclude = ["object"]).columns.to_list()
    #X[numerical_features]=X[numerical_features].astype(float) 
    
    
    return X

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(temp.copy())

#num_attribs1 = housing_extra_attribs.columns.to_list()

categorical_features = housing_extra_attribs.select_dtypes(include=['object']).columns
numerical_features = housing_extra_attribs.select_dtypes(exclude = ["object"]).columns.to_list()
#print("dataset size:" )
#print(housing_extra_attribs.shape)


# ## Check Variable distributions

# In[ ]:





# In[15]:


# display histogram for each numerical attribute
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing_extra_attribs[numerical_features].hist(bins=50, figsize=(20,20))
plt.show()


# In[16]:


skewness = housing_extra_attribs[numerical_features].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features after Box Cox transform".format(skewness.shape[0]))
print("Mean skewnees: {}".format(np.mean(skewness)))
skewness.sort_values(ascending=False)


# ## Numerical and Categorical Features

# In[17]:


num_cols =  housing_extra_attribs.select_dtypes(exclude = ["object"]).columns.to_list()
cat_cols = housing_extra_attribs.select_dtypes(include = ["object"]).columns.to_list()
print(num_cols, "\n", cat_cols)


# ## Numeric Pipeline

# Now let's build a pipeline for preprocessing the numerical attributes (note that we could use CombinedAttributesAdder() instead of FunctionTransformer(...) if we preferred):

# In[18]:


from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

def onehot(X): 
    return pd.get_dummies(X)

def scale_features(X, scaler = RobustScaler()): 
    scaler.fit_transform(X)
    return pd.DataFrame(scaler.fit_transform(X),index = X.index, columns = X.columns)


def hash_encode(X):
    from sklearn.feature_extraction import FeatureHasher
    features = []
    
    fh = FeatureHasher(n_features=2, input_type='string')
    hashed_features = fh.fit_transform(X['Neighborhood'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Neigb1', 'Neighb2'])], 
          axis=1).drop(['Neighborhood'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['MoSold'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['MonthSold'])], 
          axis=1).drop(['MoSold'], axis = 1)
       
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['MSSubClass'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['SubClass'])], 
          axis=1).drop(['MSSubClass'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['Exterior2nd'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Exter2nd'])], 
          axis=1).drop(['Exterior2nd'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['Exterior1st'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Exter1st'])], 
          axis=1).drop(['Exterior1st'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['Condition1'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Cond1'])], 
          axis=1).drop(['Condition1'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['Condition2'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Cond2'])], 
          axis=1).drop(['Condition2'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['RoofMatl'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['RoofMl'])], 
          axis=1).drop(['RoofMatl'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['Heating'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Heat'])], 
          axis=1).drop(['Heating'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['MSZoning'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['MZone'])], 
          axis=1).drop(['MSZoning'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['SaleType'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['SaleTyp'])], 
          axis=1).drop(['SaleType'], axis = 1)
    
    fh = FeatureHasher(n_features=1, input_type='string')
    hashed_features = fh.fit_transform(X['MiscFeature'])
    hashed_features = hashed_features.toarray()
    X = pd.concat([X, pd.DataFrame(hashed_features, columns = ['Misc'])], 
          axis=1).drop(['MiscFeature'], axis = 1)
    
    
    
    
    return X


# In[19]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

num_pipeline = Pipeline([
        ('imputer', DataFrameImputer()),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        #('hash', FunctionTransformer(hash_encode, validate=False)),
        ('onehot', FunctionTransformer(onehot, validate=False)),
        #('std_scaler', RobustScaler())
        ('scaler', FunctionTransformer(scale_features, validate=False)),
    ])

data_tr = num_pipeline.fit_transform(data_features)

data_tr.shape


# In[20]:


data_tr.isna().any().any()


# In[21]:


data_tr.shape


# In[22]:


import warnings
warnings.filterwarnings('ignore')


# ## Add outlier Scores and a label

# 

# In[23]:


from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF

from pyod.utils.data import get_outliers_inliers
outliers_fraction = 0.1
random_state = 42
from sklearn.cluster import KMeans

classifiers = {
    'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
    'K Nearest Neighbors (KNN)' :  KNN(contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state)
}

clf =  KNN(contamination=outliers_fraction)
#clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state)

clf.fit(data_tr)

data_tr["outlier"] = clf.predict(data_tr)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(data_tr)
    # predict raw anomaly score
    data_tr[clf_name+' score']=clf.decision_function(data_tr) * -1
    data_tr[clf_name+' score'].replace(-np.inf, np.min(data_tr[clf_name+' score'].replace(-np.inf, np.nan)))
    if clf_name=='Cluster-based Local Outlier Factor (CBLOF)':
        clf.fit(data_tr)
        data_tr["CBLOFlabel"]=clf.cluster_labels_


# In[24]:


#dX.drop(['outlier','score'], axis = 1, inplace = True)
#dX_test.drop(['outlier', 'score'], axis = 1, inplace = True)


# ## Columns with zeros

# In[25]:


# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 
overfit = []
for i in data_tr.columns:
    counts = data_tr[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(data_tr) * 100 > 99.96:
        overfit.append(i)

overfit = list(overfit)
#overfit.append('MSZoning_C (all)')


data_tr = data_tr.drop(overfit, axis=1).copy()

overfit


# ## Training and Validation Sets

# In[26]:


## split transformed data back to train and test set

#prepare data for use in algorithms
X = data_tr.iloc[0:len(train.index),:]
X_test =  data_tr.iloc[len(train.index):, :]
#y = housing_labels


# In[27]:


X.shape, X_test.shape, y.shape


# In[28]:


import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=train['OverallQual'])

X_train.shape, X_val.shape


# ## Target Variable

# In[29]:


np.isnan(y).any()


# In[30]:


y = np.log1p(y) 


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# target feature transformed
sns.distplot(y , fit=norm);
(mu, sigma) = norm.fit(y)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.title('Price (Log)');
print("Skewness: %f" % pd.DataFrame(y).skew())


# In[32]:


y.shape


# ## Correlations

# In[33]:


corr_temp = X[num_cols]
corr_temp['SalePrice'] = y
# Find correlations with the target and sort
correlations = corr_temp.corr(method='spearman')['SalePrice'].sort_values(ascending=False)
correlations_abs = correlations.abs()
print('\nTop 15 correlations (absolute):\n', correlations_abs.head(26))


# In[34]:


corr_mat=pd.concat([y, X[num_cols]], axis = 1).corr()
mask = np.zeros_like(corr_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


## Plot sizing. 
f, ax = plt.subplots(figsize = (30,20))
cmap = sns.diverging_palette(220, 10, as_cmap=True,center="light")
#cmap="BrBG"
## plotting heatmap.  
sns.heatmap(corr_mat, mask=mask, cmap=cmap,annot=True, vmax=1,vmin=-1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
## Set the title. 
plt.title("Heatmap of all the Features", fontsize = 30);


# ## Columns with Identical Values

# In[35]:


#find names of duplicate columns
def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)

# Get list of duplicate columns
duplicateColumnNames = getDuplicateColumns(X)
 
print('Duplicate Columns are as follows')



for col in duplicateColumnNames:
    print('Column name : ', col)
    
#for dset in [X, X_test, X_train]:
#    dset.drop(duplicateColumnNames, axis =1, inplace = True)


# In[ ]:





# ## Select and Train a Model 

# In[36]:


from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[37]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=kfolds, n_jobs = 1))
    return (rmse)

# rmsle scoring function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[62]:



lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4, #was 3
                                       learning_rate=0.01, 
                                       n_estimators=8000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2, # 'was 0.2'
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                      max_depth=3, min_child_weight=0,
                                      gamma=0, subsample=0.7,
                                      colsample_bytree=0.7,
                                      objective='reg:linear', nthread=-1,
                                      scale_pos_weight=1, seed=27,
                                      reg_alpha=0.00006)



# setup models hyperparameters using a pipline
# The purpose of the pipeline is to assemble several steps that can be cross-validated together, while setting different parameters.
# This is a range of values that the model considers each time in runs a CV
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]


# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds))

# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lightgbm),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)

# store models, scores and prediction values 
models = {'Ridge': ridge,
          'Lasso': lasso, 
          'ElasticNet': elasticnet,
          'lightgbm': lightgbm,
          #'xgboost': xgboost,
          'Stack':stack_gen,
         }
    
predictions = {}
scores = {}


# In[63]:




for name, model in models.items():
    
    model.fit(X, y)
    predictions[name] = np.expm1(model.predict(X))
    
    score = cv_rmse(model, X=X)
    scores[name] = (score.mean(), score.std())

(pd.DataFrame.from_dict(scores, orient='index', columns = ['RMSE Mean', 'RMSE SD']).reset_index()).sort_values(by=['RMSE Mean'], ascending=True)


# In[64]:


# get the performance of each model on training data(validation set)
print('---- Score with CV_RMSLE-----')
score = cv_rmse(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(stack_gen)
print("stack_gen score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(xgboost)
print("xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#Fit the training data X, y
print('----START Fit----',datetime.now())
print('Elasticnet')
elastic_model = elasticnet.fit(X, y)
print('Lasso')
lasso_model = lasso.fit(X, y)
print('Ridge')
ridge_model = ridge.fit(X, y)
print('Lightgbm')
lgb_model = lightgbm.fit(X, y)
print('stack_gen')
stack_model = stack_gen.fit(X, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)


# In[ ]:





# In[65]:


def blend_models_predict(X):
    return ((0.25  * elastic_model.predict(X)) +             (0.25 * lasso_model.predict(X)) +             (0.25 * ridge_model.predict(X)) +             (0.25 * lgb_model.predict(np.array(X))) + #             (0.1 * xgb_model_full_data.predict(X)) + \
            (0 * stack_gen.predict(np.array(X)))
           )


# In[66]:


print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))


# ## Submission

# In[67]:


print('Predict submission')
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_test)))


# In[68]:


submission.to_csv("submission.csv", index=False)


# ## Plot Resifuals

# In[84]:


#r = ridge
#r.fit(X_train, y_train)

y_train_pred = blend_models_predict(X_train)
y_val_pred   = blend_models_predict(X_val)

#resid = pd.DataFrame()


# In[85]:



train2 = pd.DataFrame(np.vstack((y_train_pred, y_train)).T, columns = ['predicted', 'real values'])
train2['dataset'] =  'Train'
train2['residuals'] = (y_train_pred - y_train)


train3 = pd.DataFrame(np.vstack((y_val_pred, y_val)).T, columns = ['predicted', 'real values'])
train3['dataset'] =  'Validation'
train3['residuals'] = y_val_pred - y_val

dat = pd.concat([train2, train3], axis =0).reset_index(drop = True)

sns.pairplot(dat, x_vars=["residuals", "real values"], y_vars=["predicted"],
            hue = "dataset", height=5, aspect=1, kind="reg");


# In[ ]:





# In[96]:


g = sns.JointGrid(x=y_train_pred, y=y_train_pred - y_train) 
g.plot_joint(sns.regplot, order=2) 
g.plot_marginals(sns.distplot)


# ## Most Important Features

# In[86]:


from sklearn.ensemble import RandomForestRegressor
rf_imp = RandomForestRegressor(n_estimators=1200,
                               max_depth=15,
                               min_samples_split=5,
                               min_samples_leaf=5,
                               max_features=None,
                               oob_score=True,
                               random_state=42,
                               n_jobs = -1)
pipe = make_pipeline(RobustScaler())

rf_imp.fit(pipe.fit_transform(X_train), y_train)
importances = rf_imp.feature_importances_
df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(len(X_train.columns)-1):
    feat = X_train.columns[i]
    coeff = importances[i]
    df_param_coeff.loc[i] = (feat, coeff)
df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
df_param_coeff = df_param_coeff.reset_index(drop=True)
print("Top 10 features:\n{}".format(df_param_coeff.head(50)))

importances = rf_imp.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order

names = [X_train.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Top 10 Most Important Features") # Create plot title
plt.bar(range(10), importances[indices][:10]) # Add bars
plt.xticks(range(10), names[:10], rotation=90) # Add feature names as x-axis labels
#plt.bar(range(X_train.shape[1]), importances[indices]) # Add bars
#plt.xticks(range(X_train.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show() # Show plot


# In[87]:


from sklearn.decomposition import PCA

pca = PCA(n_components=0.99)
X_reduced = pca.fit_transform(X)
X_reduced.shape


# In[89]:


X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size = 0.2, random_state=42, stratify=train['OverallQual'])

X_train.shape, X_val.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Cluster Approach

# In[90]:


from sklearn.linear_model import LogisticRegression, LinearRegression
from mlxtend.regressor import StackingCVRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn import neighbors

models = []

# setup models hyperparameters using a pipline
# The purpose of the pipeline is to assemble several steps that can be cross-validated together, while setting different parameters.
# This is a range of values that the model considers each time in runs a CV
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003,
           0.0004, 0.0005, 0.0006, 0.0007, 0.0008]


lightgbm = LGBMRegressor(objective='regression',
                         num_leaves=4,  # was 3
                         learning_rate=0.01,
                         n_estimators=8000,
                         max_bin=200,
                         bagging_fraction=0.75,
                         bagging_freq=5,
                         bagging_seed=7,
                         feature_fraction=0.2,  # 'was 0.2'
                         feature_fraction_seed=7,
                         verbose=-1,
                         )

xgboost = make_pipeline(RobustScaler(), XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                                     max_depth=3, min_child_weight=0,
                                                     gamma=0, subsample=0.7,
                                                     colsample_bytree=0.7,
                                                     objective='reg:linear', nthread=-1,
                                                     scale_pos_weight=1, seed=27,
                                                     reg_alpha=0.00006))


xgb_reg = make_pipeline(RobustScaler(), XGBRegressor(
    objective='reg:squarederror', random_state=42))

# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(
    max_iter=1e7, random_state=42, cv=kfolds))

# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(
    RobustScaler(), ElasticNetCV(max_iter=1e7, cv=kfolds))
# Gradient Boost
gbr = make_pipeline(RobustScaler(), GradientBoostingRegressor())

# SVR
svr = make_pipeline(RobustScaler(), SVR())

# =======
# SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
#    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

# Stack Regression :
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lightgbm),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)
# Stack Regression1 :
stack_gen1 = StackingRegressor(regressors=(ridge, svr, xgb_reg, gbr),
                               meta_regressor=svr,
                               use_features_in_secondary=False)


# Random Forest
rf = make_pipeline(RobustScaler(), RandomForestRegressor(
    n_estimators=100, random_state=7, n_jobs=-1))


models.append(('LGBM Regressor', lightgbm))
# Kernel Ridge Regression : made robust to outliers
models.append(('RidgeCV', ridge))
# LASSO Regression : made robust to outliers
models.append(('LassoCV', lasso))
# Elastic Net Regression : made robust to outliers
models.append(('ElasticNetCV', elasticnet))
# XGBoost Linear Regression :
models.append(('XGBoost L', xgboost))
# XGBoost Squared Regression :
models.append(('XGBoost Sq', xgb_reg))
# SVR Regression
models.append(('SVR', svr))
# Gradient Boost
models.append(('Gradient Boosting', gbr))
# Stack Regression :
models.append(('Stack', stack_gen))
# Stack Regression 1 :
models.append(('Stack1', stack_gen1))
# Random Forest
models.append(('Random Forest', rf))


# set table to table to populate with performance results
rmse_results = []
names = []
col = ['Algorithm', 'RMSE Mean', 'RMSE SD']
df_results = pd.DataFrame(columns=col)

# evaluate each model using cross-validation
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
i = 0


for name, model in models:
    # -mse scoring
    print(name)
    cv_mse_results = model_selection.cross_val_score(
        model, np.array(X), np.array(y), cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1)

    # calculate and append rmse results
    cv_rmse_results = np.sqrt(-cv_mse_results)
    rmse_results.append(cv_rmse_results)
    names.append(name)
    df_results.loc[i] = [name,
                         round(cv_rmse_results.mean(), 4),
                         round(cv_rmse_results.std(), 4)]
    i += 1

df_results.sort_values(by=['RMSE Mean'], ascending=True).reset_index(drop=True)


# In[91]:


def blend_models_predict(X):
    return ((0  * elasticnet.predict(X)) +             (0 * lasso.predict(X)) +             (0.3333 * ridge.predict(X)) +             #(0.15 * stack_gen1.predict(X)) + \
            (0.3333 * xgboost.predict(X)) + \
            (0.3333 * lightgbm.predict(np.array(X))))


# In[92]:


# fit models to entire dataset
for name, model in models:
    # -mse scoring
    print(name)
    model.fit(X, y)


# In[93]:


print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))


# In[ ]:





# In[94]:


print('Predict submission')
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_test)))
submission.to_csv("last.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Fine Tune XGBoost

# In[ ]:





# In[42]:


import xgboost as xgb
xgb_regressor = xgb.XGBRegressor(random_state=42)


# In[43]:


from time import time
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

start = time() # Get start time
cv_sets_xgb = ShuffleSplit(random_state = 10) # shuffling our data for cross-validation
scorer_xgb = make_scorer(mean_squared_error)

parameters_xgb = {'n_estimators':range(5000, 6000, 5), 
              'learning_rate':[0.01,0.6], 
              'max_depth':[4, 5],
              'min_child_weight':[3,2],
            'objective': ['reg:squarederror', 'reg:tweedie'],
            'reg_alpha': [0, 0.05],
            'gamma': [0.6, 0.3], 
            }

grid_obj_xgb = RandomizedSearchCV(xgb_regressor, 
                                  parameters_xgb,
                                  scoring = scorer, 
                                  cv = cv_sets_xgb,
                                  random_state= 99, n_jobs = -1)
grid_fit_xgb = grid_obj_xgb.fit(dI_train, yI_train)
xgb_opt = grid_fit_xgb.best_estimator_

end = time() # Get end time
xgb_time = (end-start)/60 # Calculate training time
print('It took {0:.2f} minutes for RandomizedSearchCV to converge to optimised parameters for the RandomForest model'.format(xgb_time))
# ## Print results
print('='*20)
print("best params: " + str(grid_fit_xgb.best_estimator_))
print("best params: " + str(grid_fit_xgb.best_params_))
print('best score:', grid_fit_xgb.best_score_)
print('='*20)


# In[68]:


# XGBoost with tuned parameters
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_opt = xgb.XGBRegressor(learning_rate=0.01,
                           n_estimators=5065,
                           max_depth=4,
                           min_child_weight=3,
                           gamma=0.6,
                           subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:squarederror',
                           nthread=-1,
                           scale_pos_weight=1,
                           seed=27,
                           reg_alpha=0.05,
                           random_state=42)


# ## Gradient Boosting Regressor

# In[84]:


gbr = GradientBoostingRegressor()

from time import time
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

start = time() # Get start time
cv_sets_gbr = ShuffleSplit(random_state = 10) # shuffling our data for cross-validation
parameters_gbr = {'n_estimators': [3300], 
              'learning_rate':[0.001,0.01,0.07], 
              'max_depth':[3,5,4],
              'min_samples_leaf':[2,4,3],
                'min_samples_split': [5,6,8], 
                 'loss': ["huber"],
                 'random_state': [42]}
scorer_gbr = make_scorer(mean_squared_error)
grid_obj_gbr = RandomizedSearchCV(gbr, 
                                  parameters_gbr,
                                  scoring = scorer, 
                                  cv = cv_sets_gbr,
                                  random_state= 42, n_jobs = -1)
grid_fit_gbr = grid_obj_gbr.fit(X_train, y_train)
gbr_opt = grid_fit_gbr.best_estimator_

end = time() # Get end time
xgb_time = (end-start)/60 # Calculate training time
print('It took {0:.2f} minutes for RandomizedSearchCV to converge to optimised parameters for the RandomForest model'.format(xgb_time))
# ## Print results
print('='*20)
print("best params: " + str(grid_fit_gbr.best_estimator_))
print("best params: " + str(grid_fit_gbr.best_params_))
print('best score:', grid_fit_gbr.best_score_)
print('='*20)


# In[77]:


from sklearn.ensemble import GradientBoostingRegressor
gbr_opt = GradientBoostingRegressor(n_estimators=3300,
                                learning_rate=0.001,
                                max_depth=3,
                                max_features='sqrt',
                                min_samples_leaf=2,
                                min_samples_split=5,
                                loss='huber',
                                random_state=42)


# ## LGBM Regressor

# In[63]:


lightgbm = LGBMRegressor(random_state=42)


from time import time
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

start = time() # Get start time
cv_sets = ShuffleSplit(random_state = 10) # shuffling our data for cross-validation
parameters = {'n_estimators':[3000, 4000, 7000], 
              'learning_rate':[0.001,0.01,0.07], 
              'max_bin':[150,200,250],
              'bagging_fraction': [0.8, 0.5, 0.9],
                'bagging_freq': [5,4,6], 
                 'feature_fraction': [0.2, 0.3, 0.1],
                  'min_sum_hessian_in_leaf': [11, 10, 9, 15],
                 'random_state': [42],
             'objective': ['regression']}
#scorer = make_scorer(mean_squared_error)
grid_obj = RandomizedSearchCV(lightgbm, 
                                  parameters,
                                  scoring = scorer, 
                                  cv = cv_sets,
                                  random_state= 42, n_jobs = -1)
grid_fit = grid_obj.fit(X_train, y_train)
opt = grid_fit.best_estimator_

end = time() # Get end time
xgb_time = (end-start)/60 # Calculate training time
print('It took {0:.2f} minutes for RandomizedSearchCV to converge to optimised parameters for the RandomForest model'.format(xgb_time))
# ## Print results
print('='*20)
print("best params: " + str(grid_fit.best_estimator_))
print("best params: " + str(grid_fit.best_params_))
print('best score:', grid_fit.best_score_)
print('='*20)


# In[70]:



from lightgbm import LGBMRegressor
lightgbm = LGBMRegressor(objective='regression', 
                         num_leaves=6,
                         learning_rate=0.001, 
                         n_estimators=3000,
                         max_bin=150, 
                         bagging_fraction=0.9,
                         bagging_freq=5, 
                         bagging_seed=8,
                         feature_fraction=0.1,
                         feature_fraction_seed=8,
                         min_sum_hessian_in_leaf = 15,
                         verbose=-1,
                         random_state=42)


# ## Random Forest Regressor

# In[86]:


start = time() # Get start time
rf_regressor = RandomForestRegressor(random_state=42)
cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
parameters = {'n_estimators': range(300, 500, 5), 
               'min_samples_leaf': [8, 4, 5], 
               'max_depth':[3, 5, 15],
              'min_samples_split': [3, 5, 6],
             }
#scorer = make_scorer(mean_squared_error)
n_iter_search = 10
grid_obj = RandomizedSearchCV(rf_regressor, 
                               parameters, 
                               n_iter = n_iter_search, 
                               scoring = scorer, 
                               cv = cv_sets,
                               random_state= 99, n_jobs = -1)
grid_fit = grid_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
end = time() # Get end time
rf_time = (end-start)/60 # Calculate training time
print('It took {0:.2f} minutes for RandomizedSearchCV to converge to optimised parameters for the RandomForest model'.format(rf_time))
## Print results
print('='*20)
print("best params: " + str(grid_fit.best_estimator_))
print("best params: " + str(grid_fit.best_params_))
print('best score:', grid_fit.best_score_)
print('='*20)


# In[71]:


from sklearn.ensemble import RandomForestRegressor
# RandomForest with tuned parameters
rf_reg = RandomForestRegressor(n_estimators=100, 
                               random_state=7)
rf_opt = RandomForestRegressor(n_estimators=460,
                               max_depth=3,
                               min_samples_split=5,
                               min_samples_leaf=4,
                               max_features=None,
                               oob_score=True,
                               random_state=42)


# In[50]:


rf_imp = RandomForestRegressor(n_estimators=1200,
                               max_depth=15,
                               min_samples_split=5,
                               min_samples_leaf=5,
                               max_features=None,
                               oob_score=True,
                               random_state=42)
rf_imp.fit(dI_train, yI_train)
importances = rf_imp.feature_importances_
df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(len(dI_train.columns)-1):
    feat = dI_train.columns[i]
    coeff = importances[i]
    df_param_coeff.loc[i] = (feat, coeff)
df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
df_param_coeff = df_param_coeff.reset_index(drop=True)
print("Top 10 features:\n{}".format(df_param_coeff.head(211)))

importances = rf_imp.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order

names = [dI_train.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Top 10 Most Important Features") # Create plot title
plt.bar(range(10), importances[indices][:10]) # Add bars
plt.xticks(range(10), names[:10], rotation=90) # Add feature names as x-axis labels
#plt.bar(range(X_train.shape[1]), importances[indices]) # Add bars
#plt.xticks(range(X_train.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show() # Show plot


# In[103]:


to_drop = ['AsphShn','AsbShng.001', 'ImStucc', 'WdShing', 'Other', 'Others.002', 'Stone','Stucco.001','AsbShng',
           'Others.001','CompShg','BrkCmn','Shed','Mansard','Flat','PosN','2.5S','2fmCon','Others','Stone.002',
           'Wood','Norm.001','GasW','Others.003','2Types','RRNn','Basment','RRNe','CarPort','RRAn']
to_drop =['Functional_Min1', 'Functional_Mod', 'Neighborhood_Blueste','RoofMatl_Membran','HeatingQC_Po','RoofStyle_Shed']


# In[104]:


for dset in [X, X_test]:
    dset.drop(to_drop, axis =1, inplace = True)


# ## Ridge Regression

# In[ ]:





# In[ ]:





# In[72]:


from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
kfolds = KFold(n_splits=5, shuffle=True, random_state=7)
rcv_alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5] #[14, 16, 0.1, 0.001, 5 , 4]
ridge = RidgeCV(alphas=rcv_alphas, 
                cv=kfolds)


# ## SVR

# In[73]:


from sklearn.svm import SVR
svr = SVR(C= 20, 
          epsilon= 0.008, 
          gamma=0.0003)


# ## LassoCV

# In[74]:


lassoCV_opt = LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds)


# ## Others

# In[89]:



elasticNetCV_opt = ElasticNetCV(max_iter=1e7, 
                         cv=kfolds, l1_ratio=e_l1ratio)
lgbm_opt = LGBMRegressor(objective='regression',
               num_leaves=4, #was 3
               learning_rate=0.007, 
               n_estimators=12000,
               max_bin=200, 
               bagging_fraction=0.75,
               bagging_freq=5, 
               bagging_seed=7,
               feature_fraction=0.2, # 'was 0.2'
               feature_fraction_seed=7,
               verbose=-1)
ridgeCV_opt = RidgeCV(alphas=alphas_alt, cv=kfolds)


# In[87]:


e_alphas


# ## Model Performance Review

# In[92]:


from time import time
from sklearn import model_selection
from mlxtend.regressor import StackingCVRegressor, StackingRegressor

# selection of algorithms to consider
start = time() # Get start time
models = []
models.append(('Ridge Regression', ridge))
models.append(('Random Forest', rf_opt))
models.append(('XGBoost Regressor', xgb_opt))
models.append(('Gradient Boosting Regressor', gbr_opt))
models.append(('LGBM Regressor',lightgbm))
models.append(('SVR',svr))
models.append(('KNN',neighbors.KNeighborsRegressor(n_neighbors = 11)))

models.append(('RidgeCV', ridgeCV_opt))
models.append(('LassoCV', lassoCV_opt))
models.append(('ElasticNetCV', elasticNetCV_opt))
models.append(('LGBM Regressor', lgbm_opt))

models.append(('StackingRegressor',StackingRegressor(regressors=(ridge,svr, xgb_opt, gbr_opt),
                                                     meta_regressor=svr,
                                                     use_features_in_secondary=False)))
models.append(('StackingRegressor1',StackingRegressor(regressors=(lassoCV_opt,elasticNetCV_opt,ridgeCV_opt ),
                                                     meta_regressor=lassoCV_opt,
                                                     use_features_in_secondary=True)))

# set table to table to populate with performance results
rmse_results = []
names = []
col = ['Algorithm', 'RMSE Mean', 'RMSE SD']
df_results = pd.DataFrame(columns=col)

# evaluate each model using cross-validation
kfold = model_selection.KFold(n_splits=5, shuffle = True, random_state=7)
i = 0
for name, model in models:
    print("Evaluating {}...".format(name))
    # -mse scoring
    cv_mse_results = model_selection.cross_val_score(
        model, np.array(dX_train), np.array(y_train), cv=kfold, scoring='neg_mean_squared_error',n_jobs = -1)
    # calculate and append rmse results
    cv_rmse_results = np.sqrt(-cv_mse_results)
    rmse_results.append(cv_rmse_results)
    names.append(name)
    df_results.loc[i] = [name,
                         round(cv_rmse_results.mean(), 4),
                         round(cv_rmse_results.std(), 4)]
    i += 1
end = time() # Get end time
eval_time = (end-start)/60 # Calculate training time
print('Evaluation completed.\nIt took {0:.2f} minutes to evaluate all models using a 5-fold cross-validation.'.format(eval_time))
df_results.sort_values(by=['RMSE Mean'], ascending=True).reset_index(drop=True)


# ## Blending ML Algorithms with StackingCVRegressor

# In[93]:


#from mlxtend.regressor import StackingCVRegressor, StackingRegressor
stack_gen = StackingCVRegressor(regressors=(gbr_opt,
                                            xgb_opt,
                                            #lightgbm,
                                            #model,
                                            ridge, 
                                            svr),
                                meta_regressor=svr,
                                use_features_in_secondary=False)


# In[107]:


stack_gen = StackingRegressor(regressors=(lassoCV_opt,elasticNetCV_opt,ridgeCV_opt ),
                                                     meta_regressor=lassoCV_opt,
                                                     use_features_in_secondary=True)


# In[94]:


print('Fitting models to the training data:')
start = time() # Get start time

df_train_ml = dX
target_ml = y
print('xgboost....')
xgb_model_full_data = xgb_opt.fit(df_train_ml, target_ml)
print('GradientBoosting....')
gbr_model_full_data = gbr_opt.fit(df_train_ml, target_ml)
#print('lightgbm....')
#lgb_model_full_data = lightgbm.fit(df_train_ml, target_ml)
print('RandomForest....')
rf_model_full_data = rf_opt.fit(df_train_ml, target_ml)
print('Ridge....')
ridge_model_full_data = ridge.fit(df_train_ml, target_ml)
print('SVR....')
svr_model_full_data = svr.fit(df_train_ml, target_ml)
print('Stacking Regression....')
stack_gen_model = stack_gen.fit(np.array(df_train_ml), np.array(target_ml))



print('LassoCV....')
lassoCVmodel = lassoCV_opt.fit(np.array(df_train_ml), np.array(target_ml))
print('ElasticNetCV....')
elasticNetCVmodel = elasticNetCV_opt.fit(np.array(df_train_ml), np.array(target_ml))
print('LGBM....')
lgbmmodel = lgbm_opt.fit(np.array(df_train_ml), np.array(target_ml))
print('RidgeCV....')
ridgeCVmodel = ridgeCV_opt.fit(np.array(df_train_ml), np.array(target_ml)) 



end = time() # Get end time
fitting_time = (end-start)/60 # Calculate training time
print('Fitting completed.\nIt took {0:.2f} minutes to fit all the models to the training data.'.format(fitting_time))


# In[ ]:





# ## Submission 

# In[96]:


def blend_models_predict(X):
    return (((1/2) * stack_gen_model.predict(np.array(X))) +             ((1/2) * svr_model_full_data.predict(X))             #((1/5) * lassoCVmodel.predict(X)) +\
            #((1/5) * elasticNetCVmodel.predict(X)) +\
            #((1/7) * gbr_model_full_data.predict(X)) + \
            #((1/7) * lgb_model_full_data.predict(X)) + \
            #((1/7) * xgb_model_full_data.predict(X)) + \
            #((1/7) * rf_model_full_data.predict(X)) + \
            #((1/5) * ridgeCVmodel.predict(X)) \
            #((1/4)* knn_model.predict(X))
           )


# In[101]:


# Generate predictions from the blend
y_pred_final = (np.expm1(blend_models_predict(dX_test)))
my_submission2 = pd.DataFrame(y_pred_final, index = dX_test.index, columns = ['SalePrice'])
my_submission2.to_csv('submission-070719_v3.csv')


# In[111]:


from time import time
from sklearn import model_selection
from mlxtend.regressor import StackingCVRegressor, StackingRegressor

# selection of algorithms to consider
start = time() # Get start time
models = []
models.append(('Ridge Regression', ridge))
models.append(('Random Forest', rf_opt))
models.append(('XGBoost Regressor', xgb_opt))
models.append(('Gradient Boosting Regressor', gbr_opt))
models.append(('LGBM Regressor',lightgbm))
models.append(('SVR',svr))
models.append(('KNN',neighbors.KNeighborsRegressor(n_neighbors = 11)))

models.append(('RidgeCV', ridgeCV_opt))
models.append(('LassoCV', lassoCV_opt))
models.append(('ElasticNetCV', elasticNetCV_opt))
models.append(('LGBM Regressor', lgbm_opt))

models.append(('StackingRegressor',StackingRegressor(regressors=(ridge,svr, xgb_opt, gbr_opt),
                                                     meta_regressor=svr,
                                                     use_features_in_secondary=False)))
models.append(('StackingRegressor1',StackingRegressor(regressors=(lassoCV_opt,elasticNetCV_opt,ridgeCV_opt ),
                                                     meta_regressor=lassoCV_opt,
                                                     use_features_in_secondary=True)))

# set table to table to populate with performance results
rmse_results = []
names = []
col = ['Algorithm', 'RMSE Mean', 'RMSE SD']
df_results = pd.DataFrame(columns=col)

# evaluate each model using cross-validation
kfold = model_selection.KFold(n_splits=5, shuffle = True, random_state=7)
i = 0
for name, model in models:
    print("Evaluating {}...".format(name))
    # -mse scoring
    cv_mse_results = model_selection.cross_val_score(
        model, dO_train, yO_train, cv=kfold, scoring='neg_mean_squared_error')
    # calculate and append rmse results
    cv_rmse_results = np.sqrt(-cv_mse_results)
    rmse_results.append(cv_rmse_results)
    names.append(name)
    df_results.loc[i] = [name,
                         round(cv_rmse_results.mean(), 4),
                         round(cv_rmse_results.std(), 4)]
    i += 1
end = time() # Get end time
eval_time = (end-start)/60 # Calculate training time
print('Evaluation completed.\nIt took {0:.2f} minutes to evaluate all models using a 5-fold cross-validation.'.format(eval_time))
df_results.sort_values(by=['RMSE Mean'], ascending=True).reset_index(drop=True)


# In[112]:





# In[114]:


print('Fitting models to the training data:')
start = time() # Get start time

df_train_ml = dO
target_ml = yO
print('xgboost....')
xgb_model_full_data = xgb_opt.fit(df_train_ml, target_ml)
print('GradientBoosting....')
gbr_model_full_data = gbr_opt.fit(df_train_ml, target_ml)
#print('lightgbm....')
#lgb_model_full_data = lightgbm.fit(df_train_ml, target_ml)
print('RandomForest....')
rf_model_full_data = rf_opt.fit(df_train_ml, target_ml)
print('Ridge....')
ridge_model_full_data = ridge.fit(df_train_ml, target_ml)
print('SVR....')
svr_model_full_data = svr.fit(df_train_ml, target_ml)
print('Stacking Regression....')
stack_gen_model = stack_gen.fit(np.array(df_train_ml), np.array(target_ml))





end = time() # Get end time
fitting_time = (end-start)/60 # Calculate training time
print('Fitting completed.\nIt took {0:.2f} minutes to fit all the models to the training data.'.format(fitting_time))


# In[115]:



# Generate predictions from the blend
#svr_model_full_data = svr.fit(dO, yO)



y_pred_final2 = np.expm1(stack_gen_model.predict(dO_test)) 
my_submission2 = pd.DataFrame({'Id': dO_test.index, 'SalePrice': y_pred_final2})


# In[116]:


my_submission = pd.concat([my_submission2,my_submission1])


# In[117]:


X.shape


# In[98]:


# Generate submission dataframe
#my_submission = pd.DataFrame({'Id': housing_test.index, 'SalePrice': y_pred_final})

# Exporting submission to CSV
my_submission2.to_csv('submission-070719_v3.csv', index=False)


# ### Prepare Polynomial Features

# In[103]:


from sklearn.preprocessing import PolynomialFeatures

def add_extra_features_2(X, poly=False):
    xx=X.copy()
    if poly:
        poly_features = PolynomialFeatures(degree=2, include_bias = False)
        xx = poly_features.fit_transform(xx.values)
    else:
        xx = pd.concat((xx, (xx**2)), axis = 1)
    
    return xx



# squared numeric features  
X_2 = add_extra_features_2(X, poly=False)


# polynomial numeric features

X_poly = add_extra_features_2(X, poly=True)


# In[31]:


X.shape


# In[32]:


X_test_2.shape


# In[33]:


X_train_poly.shape


# ## Select and Train a Model

# ### Training and Evaluating on the Validation Set

# In[194]:


from sklearn.linear_model import Ridge
import sklearn
ridge_reg = Ridge(alpha = 0.001, solver = "cholesky",random_state=42)
ridge_reg.fit(dO_train, yO_train )

from sklearn.metrics import *
# evaluate on validation set
housing_predictions = ridge_reg.predict(dO_val)
lin_mse = sklearn.metrics.mean_squared_error(yO_val, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[117]:


X_test.shape


# In[178]:


from sklearn.tree import DecisionTreeRegressor
# try a decision tree
tree_reg = DecisionTreeRegressor()#(random_state=42,max_depth=14, min_samples_leaf=4, max_features=22)
tree_reg.fit(dX_train.values, y_train)
# evaluate on validation set
housing_predictions = tree_reg.predict(dX_val)
tree_mse = mean_squared_error(y_val, housing_predictions)
#tree_rmse = np.sqrt(tree_mse)
tree_mse


# In[ ]:





# ## Better Evaluation Using Cross-Validation

# In[ ]:





# In[123]:



from sklearn.svm import SVR

svm_reg = SVR(kernel="sigmoid", gamma='scale')
#svm_reg.fit(X_train, y_train)

display_cv_scores(svm_reg, X, y)


# In[79]:


scaler = StandardScaler()


# In[83]:


import keras
dir(keras.losses)


# In[36]:



from keras import backend as K
from keras.losses import *
import math

# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return abs( 1 - SS_res/(SS_tot + K.epsilon()) )


from keras.losses import logcosh

from keras import regularizers
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Activation, Dense

def create_model(epochs=100, batch_size=32, dropout_rate=0.005, l=0.0001, lr=0.001, loss=mean_squared_error ):
    NN_model = Sequential()
    # The Input Layer :
    NN_model.add(Dense(6, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    # The Hidden Layers :
    #NN_model.add(Dense(1, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    #NN_model.add(Dense(1, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    #NN_model.add(Dropout(rate =dropout_rate))
    #NN_model.add(Dense(16, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    NN_model.add(Dense(4, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    from keras import optimizers
    adam = optimizers.Adam(lr=lr)
    
    # Compile the network :
    NN_model.compile(loss=loss, optimizer=adam, metrics=['mse'])

    #NN_model.summary()

    return NN_model

NN_model = create_model()
NN_model.summary()

from keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn=create_model)


# In[71]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

checkpoint_name = "Weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint_name = "weights.best.hdf5"

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [EarlyStopping(monitor='val_loss', patience=10),checkpoint]


# In[72]:


scaler = StandardScaler()


# In[73]:


NN_model.fit(X, y, validation_split=0.33, epochs=200, batch_size=52,verbose = 1, callbacks=callbacks_list)


# In[48]:



#NN_model.load_weights("weights.best.hdf5")
#NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])



housing_predictions = np.expm1((NN_model.predict(X_test)))
submission = pd.DataFrame(data=housing_predictions, index=test_ID , columns = ["SalePrice"])
submission.to_csv("newsixteenth.csv")


# In[76]:


X.isnull().any().any()


# In[120]:



# Generate predictions from the blend
y_pred_final1 = (np.expm1(NN_model.predict(dI_test)))
my_submission1 = pd.DataFrame({'Id': dI_test.index, 'SalePrice': y_pred_final1.ravel()})


# In[125]:



# Generate predictions from the blend
y_pred_final2 = (np.expm1(NN_model.predict(dO_test)))
my_submission2 = pd.DataFrame({'Id': dO_test.index, 'SalePrice': y_pred_final2.ravel()})


# In[126]:


submission = pd.concat([my_submission1, my_submission2])
submission.to_csv("sixteenth.csv")


# ## Fine-Tune Your Model
# 
# ### Grid Search

# In[102]:


import scipy
scipy.test()


# In[34]:


from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.layers.core import Dropout
from keras import regularizers
from keras.losses import logcosh
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from keras.losses import *
from keras import backend as K
from keras.losses import *
import math


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from joblib import Parallel, delayed, parallel_backend

with parallel_backend('threading'):
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        #{'n_estimators': [30, 180, 220], 'max_features': [16, 32, 64, 128]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'epochs': [100, 50, 200], 'batch_size': [64, 32, 128], 'dropout_rate': [0.005, 0.01], 'l': [0.01, 0.0001, 0.05], 'lr': [0.001, 0.01, 0.003], 
        'loss': [hinge, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error]},
        ]

    #forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    grid_search = GridSearchCV(model, param_grid, cv=3,
                                   scoring='neg_root_mean_squared_error', return_train_score=True, n_jobs = -1)

    grid_search.fit(X, y)


# In[ ]:


grid_search.best_params_ 


# In[175]:



housing_predictions = np.expm1((grid_search.best_estimator_.predict(X_test)))
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# In[ ]:





# ### Embedings

# In[335]:


dfX_train.shape


# In[336]:


dfX_val.shape


# In[363]:



import seaborn as sns

df_num = num_pipeline.set_params(imputer__cor_cols_to_drop = correlated_columsns_to_drop).fit_transform(housing[num_attribs])
df_cat = DataFrameImputer().fit_transform(housing[cat_attribs])
dfX_train = pd.concat([df_num, df_cat], axis=1, sort=False)

df_num = num_pipeline.set_params(imputer__cor_cols_to_drop = correlated_columsns_to_drop).fit_transform(strat_test_set.drop("SalePrice", axis=1)[num_attribs])
df_cat = DataFrameImputer().fit_transform(strat_test_set.drop("SalePrice", axis=1)[cat_attribs])
dfX_val = pd.concat([df_num, df_cat], axis=1, sort=False)

df_num = num_pipeline.set_params(imputer__cor_cols_to_drop = correlated_columsns_to_drop).fit_transform(housing_test[num_attribs])
df_cat = DataFrameImputer().fit_transform(housing_test[cat_attribs])
dfX_test = pd.concat([df_num, df_cat], axis=1, sort=False)

dfX = pd.concat([dfX_train, dfX_val], axis =0)


# In[364]:


dfX.shape


# In[362]:


data_df = dfX
embed_cols=[i for i in data_df.select_dtypes(include=['object'])]
for i in embed_cols:
    print(i,data_df[i].nunique())


# In[49]:



#df_train = housing_full
import keras

from keras.layers import *
from keras.models import *


embed_cols=[i for i in dfX.select_dtypes(include=['object'])]

#converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test









# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:



for categorical_var in dfX_train.select_dtypes(include=['object']):
    
    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'
  
    no_of_unique_cat  = dfX_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
  
    print('Categorica Variable:', categorical_var,
        'Unique Categories:', no_of_unique_cat,
        'Embedding Size:', embedding_size)


# In[218]:


for categorical_var in dfX_train.select_dtypes(include=['object']):
    
    input_name= 'Input_' + categorical_var.replace(" ", "")
    print(input_name)


# In[106]:


input_models=[]
output_embeddings=[]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

for categorical_var in dfX_train.select_dtypes(include=['object']):
    
    #Name of the categorical variable that will be used in the Keras Embedding layer
    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'
  
    # Define the embedding_size
    no_of_unique_cat  = dfX_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
  
    #One Embedding Layer for each categorical variable
    input_model = Input(shape=(1,))
    output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)
    output_model = Reshape(target_shape=(embedding_size,))(output_model)    
  
    #Appending all the categorical inputs
    input_models.append(input_model)
  
    #Appending all the embeddings
    output_embeddings.append(output_model)
  
#Other non-categorical data columns (numerical). 
#I define single another network for the other columns and add them to our models list.
input_numeric = Input(shape=(len(dfX_train.select_dtypes(include=numerics).columns.tolist()),))
embedding_numeric = Dense(128)(input_numeric) 
input_models.append(input_numeric)
output_embeddings.append(embedding_numeric)

#At the end we concatenate altogther and add other Dense layers
output = Concatenate()(output_embeddings)
output = Dense(256, kernel_initializer="normal",kernel_regularizer=regularizers.l2(0.1))(output)
output = Activation('relu')(output)
output = Dense(512, kernel_initializer="normal",kernel_regularizer=regularizers.l2(0.1))(output)
output = Activation('relu')(output)
output= Dropout(rate = 1-0.25)(output)
output = Dense(512, kernel_initializer="normal",kernel_regularizer=regularizers.l2(0.1))(output)
output = Activation('relu')(output)
output = Dense(1024, kernel_initializer="normal",kernel_regularizer=regularizers.l2(0.1))(output)
#output= Dropout(0.3)(output)
output = Dense(1, activation='linear')(output)




model = Model(inputs=input_models, outputs=output)
model.compile(loss='mean_squared_error', optimizer='Adam',metrics=['mae'])


# In[ ]:





# In[107]:


X_train_list,X_val_list,X_test_list = preproc(dfX_train,dfX_val, dfX_test)


# In[108]:


from keras.callbacks import ModelCheckpoint

checkpoint_name = "Weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint_name = "weights.best3.hdf5"
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [EarlyStopping(monitor='val_loss', patience=10),checkpoint]


# In[109]:


history  =  model.fit(X_train_list,y_train,validation_data=(X_val_list,y_val) , epochs =  100 , batch_size = 32,callbacks=callbacks_list, verbose= 2)

#NN_model.fit(X_square, y, epochs=100, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


model.load_weights("weights.best3.hdf5")
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])


# In[111]:


housing_predictions = np.expm1(model.predict(X_test_list))
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[328]:





# In[ ]:





# In[329]:





# In[ ]:


best model submission so far using NN model, second best - using model


# In[ ]:





# In[83]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    train_errors, val_errors = [], []
    for m in range (1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),"r-+", linewidth = 2,label = "train")
    plt.plot(np.sqrt(val_errors),"b-", linewidth = 3,label = "val")
    
plot_learning_curves(NN_model, X, y)


# In[84]:


batch_size = 64
epochs = 20
num_classes = 1


# In[94]:


from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LeakyReLU
from keras.layers import Flatten
import keras
from keras.layers.core import Reshape


fashion_model = Sequential()

fashion_model.add(Reshape([X.shape[1],1], input_shape=(X.shape[1],)))
#print(fashion_model.input_shape)
#print(fashion_model.output_shape)
fashion_model.add(Conv1D(64,kernel_size=1,activation='selu',padding='valid'))
fashion_model.add(LeakyReLU(alpha=0.1))
print(fashion_model.input_shape)
print(fashion_model.output_shape)
fashion_model.add(MaxPooling1D(2, padding='same'))
fashion_model.add(Dropout(0.25))
print(fashion_model.output_shape)
fashion_model.add(Conv1D(128, 3, activation='selu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.25))
print(fashion_model.output_shape)
fashion_model.add(Conv1D(256,  3, activation='selu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size= 2,padding='same'))
fashion_model.add(Dropout(0.4))
print(fashion_model.output_shape)
fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3)) 

print(fashion_model.output_shape)

fashion_model.add(Dense(1, kernel_initializer='normal', activation='linear'))



print(fashion_model.output_shape)


# In[ ]:





# In[95]:


fashion_model.summary()


# In[99]:


fashion_model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(),metrics=['mean_absolute_error'])


# In[100]:


from keras.callbacks import ModelCheckpoint

checkpoint_name = "Weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint_name = "weights.best2.hdf5"
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [EarlyStopping(monitor='val_loss', patience=20),checkpoint]

fashion_model.fit(X, y, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



#fashion_train_dropout = fashion_model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, y_val))


# In[92]:


X.shape


# In[187]:


fashion_model.load_weights("weights.best2.hdf5")
fashion_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# In[188]:


housing_predictions = np.expm1(fashion_model.predict(X_test))
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2843]:


from sklearn.metrics import accuracy_score
y_pred = dnn_reg.predict_scores(X_val,as_iterable=False)
np.sqrt(metrics.mean_squared_error(y_val, y_pred))
#dnn_reg


# In[ ]:





# ## Cross Validation NN

# In[ ]:


from sklearn.model_selection import KFold
kf = KFold(5)

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from sklearn import metrics

oos_y = []
oos_pred = []
fold = 0
for train, test in kf.split(X):
    fold+=1
    print("Fold #{}".format(fold))

    x_train = X[train]
    y_train = y[train]
    x_test = X[test]
    y_test = y[test]

    model = Sequential()
    model.add(Dense(128, input_dim = X.shape[1], activation = "relu"))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    
    
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam',metrics=['mean_absolute_error'])

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,patience=5, verbose=1, mode = 'auto')
    
    model.fit(x_train, y_train, validation_data = (x_test, y_test), callbacks = [monitor], verbose=0, epochs=1000)

    pred = model.predict(x_test)

    oos_y.append(y_test)
    oos_pred.append(pred)

    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print("Fold score (MAE): {}".format(score))

#Build the oos predictions list and calculate hte error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score = np.sqrt(metrics.mean_squared_error(oos_pred, oos_y))
print("Final, out of sample score (MAE): {}".format(score))


# In[ ]:





# In[ ]:





# In[173]:


y_train.info()


# In[ ]:





# In[81]:


import xgboost
from sklearn.model_selection import GridSearchCV

train_x=X_train
train_y=y_train

#for tuning parameters
parameters_for_testing = {
    'colsample_bytree':[0.4,0.6,0.8],
    'gamma':[0,0.03,0.1,0.3],
    'min_child_weight':[1.5,6,10],
    'learning_rate':[0.1,0.07],
    'max_depth':[3,5],
    #'n_estimators':[10000],
    'reg_alpha':[1e-5, 1e-2,  0.75],
    'reg_lambda':[1e-5, 1e-2, 0.45],
    'subsample':[0.6,0.95]  
}

                    
xgb_model = xgboost.XGBRegressor(learning_rate =0.1, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=42)

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
gsearch1.fit(train_x,train_y)
print (gsearch1.grid_scores_)
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)

best_xgb_model  = gsearch1.best_model_
best_xgb_model.fit(train_x,train_y)


# In[92]:


best_xgb_model  = xgboost.XGBRegressor(colsample_bytree= 0.4, gamma= 0, learning_rate= 0.1, max_depth= 5, min_child_weight= 6, reg_alpha= 1e-05, reg_lambda= 1e-05, subsample= 0.95)


# In[93]:


best_xgb_model.fit(train_x,train_y)


# In[ ]:


best_xgb_model.fit(train)


# In[94]:


housing_predictions = best_xgb_model.predict(X_test)
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# In[ ]:





# In[ ]:





# ## Fine-Tune Your Model
# 
# ### Grid Search

# In[157]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [10, 20, 50], 'max_features': [8, 16, 32, 50], 'max_depth': range(3,7)},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [10, 50], 'max_features': [12, 35, 50], 'max_depth': range(3,7)},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
grid_search.fit(X, y.ravel())


# ##### 

# In[158]:


best_params = grid_search.best_params_
best_params


# In[162]:


housing_predictions = y_sc.inverse_transform(grid_search.predict(X_test).reshape(-1, 1))
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# In[ ]:





# In[98]:



cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:





# ### Randomized Search

# In[99]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


# In[100]:



cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# {'max_features': 7, 'n_estimators': 180} produces smallest error of 33034.91023010842 

# ## Ensemble Methods

# ### Voting Classifiers

# In[ ]:


VotingRegressor


# In[290]:


from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor(
    estimators = [('nn', NN_model),('ada',best_ada_reg)]
)
voting_reg.fit(X, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Random Forests

# In[101]:



from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=500, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[102]:



housing_predictions = forest_reg.predict(X_val)
forest_mse = mean_squared_error(y_val, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[103]:


forest_scores = cross_val_score(forest_reg, X, y,
                             scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-lin_scores)
display_scores(forest_rmse_scores)


# In[ ]:





# ### Extra-Trees

# In[104]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import ExtraTreeRegressor
from scipy.stats import randint

param_distribs = {
        #'max_depth': randint(low=3, high=200),
        #'max_features': randint(low=1, high=100),
        #'min_samples_leaf': randint(low=3, high=7),
        #'min_samples_split': randint(low=3, high=5),
    }

extra_forest_reg = ExtraTreeRegressor(random_state=42,max_depth=10, min_samples_leaf=4)
rnd_search = RandomizedSearchCV(extra_forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


# In[105]:



cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
print("Best rnd search model's score: {}".format(np.min(np.sqrt(-cvres["mean_test_score"]))))


# In[106]:


best_extra_forest_reg = rnd_search.best_estimator_


# In[107]:



housing_predictions = best_extra_forest_reg.predict(X_val)
forest_mse = mean_squared_error(y_val, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[2815]:


#housing_predictions = best_extra_forest_reg.predict(X_test)
#submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
#submission.to_csv("sixth.csv")


# ### Boosting

# In[276]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(
    DecisionTreeRegressor(random_state=42,max_depth=14, min_samples_leaf=4), 
    random_state=42,
    #n_estimators=200,
    #learning_rate=0.5
)
ada_reg.fit(X, y)


# In[277]:


ada_reg.estimators_


# In[278]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import ExtraTreeRegressor
from scipy.stats import randint, norm
import random

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'learning_rate': norm(0.5, 0.06),
}


rnd_search = RandomizedSearchCV(ada_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X, y)


# In[279]:



cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
print("Best rnd search model's score: ", np.min(np.sqrt(-cvres["mean_test_score"])))


# In[280]:


best_ada_reg = rnd_search.best_estimator_
best_ada_reg


# In[281]:


housing_predictions = best_ada_reg.predict(X_val)
forest_mse = mean_squared_error(y_val, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[282]:


housing_predictions = best_ada_reg.predict(X_test)
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# In[378]:


Z = best_ada_reg.predict(X).reshape(-1,1)

from keras import regularizers
from keras.layers.core import Dropout

NN1_model = Sequential()

# The Input Layer :
NN1_model.add(Dense(128, kernel_initializer='normal',input_dim = Z.shape[1], activation='relu'))

# The Hidden Layers :
NN1_model.add(Dense(256, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
NN1_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN1_model.add(Dropout(0.25))
NN1_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN1_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN1_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN1_model.summary()



# In[379]:


from keras.callbacks import ModelCheckpoint

checkpoint_name = "Weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint_name = "weights1.best.hdf5"
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[380]:



NN1_model.fit(Z, y, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[382]:


NN1_model.load_weights("weights1.best.hdf5")
NN1_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


housing_predictions = [(y)/2 for x, y in zip(NN_model.predict(X_test), NN1_model.predict(best_ada_reg.predict(X_test)))]
NN1_model.predict(best_ada_reg.predict(X_test))
submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
submission.to_csv("sixteenth.csv")


# In[376]:


Z.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2841]:


#housing_predictions = best_ada_reg.predict(pca.transform(X_test))
#submission = pd.DataFrame(data=housing_predictions, index=housing_test.index , columns = ["SalePrice"])
#submission.to_csv("eighth.csv")


# This is our best model so far

# ## Gradient Boosting

# In[2795]:


from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=42,max_depth=14, min_samples_leaf=4, max_features=200, n_estimators = 180, learning_rate = 0.01)
gbrt.fit(X_train, y_train)


# In[2796]:


housing_predictions = gbrt.predict(X_val)
forest_mse = mean_squared_error(y_val, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[2797]:


forest_scores = cross_val_score(gbrt, X, y,
                             scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-lin_scores)
display_scores(forest_rmse_scores)


# In[2344]:


gbrt = GradientBoostingRegressor(max_depth =2, warm_start=True)

min_val_error = float("inf")
error_going_up = 0 
for n_estimators in range(1,220):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up =0 
    else:
        error_going_up += 1
        if error_going_up == 40:
            break # early stopping


# In[2345]:


bst_n_estimators = n_estimators
n_estimators


# In[2346]:


gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)


# In[2347]:


housing_predictions = best_extra_forest_reg.predict(X_val)
forest_mse = mean_squared_error(y_val, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[2732]:


forest_scores = cross_val_score(best_extra_forest_reg, X, y,
                             scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-lin_scores)
display_scores(forest_rmse_scores)


# ## Clustering

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



outliers_fraction = 0.01
random_state = 42

clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state)




# In[ ]:





# In[ ]:





# In[ ]:





# In[186]:


import pandas as pd
import numpy as np

# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from pyod.utils.data import get_outliers_inliers

outliers_fraction = 0.1


# In[209]:


from pyod.models.knn import KNN
outliers_fraction = 0.01
random_state = 42

classifiers = {
    'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state, n_clusters=3),
    'Isolation Forest': IForest(contamination=outliers_fraction/10,random_state=random_state),
    'K Nearest Neighbors (KNN)' :  KNN(contamination=outlier_fraction)
}


# In[135]:



outliers_fraction = 0.01
random_state = 42

clf =  KNN(contamination=outlier_fraction)

clf.fit(X)

y_pred = clf.predict(X_reduced[:,0:2])

scores_pred = clf.decision_function(X_reduced[:,0:2]) * -1


# In[141]:


strat_test_set["outlier"]=clf.predict(X_val)
strat_train_set["outlier"]=clf.predict(X_train)
housing_test["outlier"] = clf.predict(X_test)

#strat_train_set, strat_test_set
X_train = full_pipeline.fit_transform(strat_train_set.drop("SalePrice", axis=1))
#y_train = housing_labels

X_val = full_pipeline.fit_transform(strat_test_set.drop("SalePrice", axis=1))
#y_val = strat_test_set["SalePrice"]

X = full_pipeline.fit_transform(pd.concat([strat_train_set, strat_test_set]))
#y = pd.concat([strat_train_set['SalePrice'], strat_test_set['SalePrice']])

X_test = full_pipeline.fit_transform(housing_test)


# In[144]:


strat_test_set.shape


# In[ ]:





# ## PCA

# In[193]:


XX = X.drop(["CBLOFlabel", "outlier", 'Angle-based Outlier Detector (ABOD)', 'Cluster-based Local Outlier Factor (CBLOF)','Isolation Forest','K Nearest Neighbors (KNN)'], axis = 1)


# In[194]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(XX)


# In[ ]:



cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(clusters), vmax=max(clusters))
colors = [cmap(normalize(value)) for value in clusters]


# In[210]:


from scipy import stats

xx , yy = np.meshgrid(np.linspace(np.amin(X_reduced[:,0]), np.amax(X_reduced[:,0]),200), np.linspace(np.amin(X_reduced[:,1]), np.amax(X_reduced[:,1]), 200))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(XX)#X_reduced[:,0:2])
    # predict raw anomaly score
    scores_pred = clf.decision_function(XX)*-1#X_reduced[:,0:2]) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(XX)#X_reduced[:,0:2])
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))
    
    
    #dfx['outlier'] = y_pred.tolist()
    
    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 =  X_reduced[y_pred == 0,0]#.reshape(-1,1)
    IX2 =  X_reduced[y_pred == 0,1]#.reshape(-1,1)
    try:
        colors = clf.cluster_labels_[y_pred == 0]
    except:
        colors = 'white'
    
    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 =  X_reduced[y_pred == 1,0]#.reshape(-1,1)
    OX2 =  X_reduced[y_pred == 1,1]#.reshape(-1,1)
         
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
        
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
        
    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) * -1
    Z = Z.reshape(xx.shape)
          
    # fill blue map colormap from minimum anomaly score to threshold value
    try:
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
    except:
        print((Z.min(), threshold, Z.max()))
        plt.contourf(xx, yy, Z, levels=np.linspace(min(Z.min(), threshold),max(Z.min(), threshold), 7),cmap=plt.cm.Blues_r)
        
    
    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        
    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        
    b = plt.scatter(IX1,IX2, c=colors,s=20, edgecolor='k')
    
    c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
       
    plt.axis('tight')  
    
    # loc=2 is used for the top left corner 
    plt.legend(
        [ a.collections[0],b,c],
        [ 'learned decision function', 'inliers','outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)
      
    #plt.xlim((0, 1))
    #plt.ylim((0, 1))
    plt.title(clf_name)
    plt.show()


# In[211]:


X_reduced[y_pred == 0,0].shape


# In[92]:


xx.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## PCA

# In[216]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(XX)


# In[89]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_pred)
plt.axis([np.amin(X_reduced[:,0]), np.amax(X_reduced[:,0]), np.amin(X_reduced[:,1]), np.amax(X_reduced[:,1])])
plt.show()


# In[ ]:





# In[155]:


np.amin(X_reduced[:,0]), np.amin(X_reduced[:,1])


# In[221]:


import matplotlib

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state, n_clusters=3)
clf.fit(XX)

cmap = matplotlib.cm.get_cmap('viridis')
#normalize = matplotlib.colors.Normalize(vmin=min(clf.cluster_labels_), vmax=max(clf.cluster_labels_))
colors = [cmap(normalize(value)) for value in clf.cluster_labels_]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =X_reduced[:,0]
yy =X_reduced[:,1]
z =X_reduced[:,2]


ax.scatter(x, yy,z, c=colors, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

plt.show()


# In[330]:







# In[ ]:





# In[218]:


min(y)


# In[216]:


max(y)


# In[ ]:




