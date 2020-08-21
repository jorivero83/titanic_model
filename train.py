import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pickle

# load data
df_train = pd.read_csv('data/train.csv')

# ================================= Data preprocessor ======================================
disc_vars = ['SibSp','Parch']
cat_vars = ['Pclass','Sex','Embarked']
num_vars = ['Age','Fare']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

disc_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999))])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='none')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_vars),
    ('disc', disc_transformer, disc_vars),
    ('cat', cat_transformer, cat_vars)])

# ================================= Building the model ======================================

# Spliting the data into test and train sets
X = df_train[num_vars + cat_vars + disc_vars]
y = df_train["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size =.20, random_state=1)

# Fit the model
gbm_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', lgb.LGBMClassifier())])
scores = cross_validate(gbm_model, X_train, y_train, scoring='roc_auc')
print('-' * 80)
print(str(gbm_model.named_steps['classifier']))
print('-' * 80)
for key, values in scores.items():
    print(key, ' mean ', values.mean())
    print(key, ' std ', values.std())
print('-' * 80)
gbm_model.fit(X_train, y_train)

# ================================= Saving the model ======================================
pickle.dump(gbm_model, open('models/gbm_model.pickle', 'wb'))
