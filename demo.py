import pandas as pd
import sweetviz as sv

# load data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# generate report
sel_cols = ['PassengerId', 'Survived', 'Age']
in_args = {'source': [df_train[sel_cols],'Train'], 'target_feat': 'Survived',
           'feat_cfg': sv.FeatureConfig(skip="PassengerId"),
           'pairwise_analysis': 'on'}
titanic_report = sv.analyze(**in_args)
titanic_report.show_html('doc/titanic_report.html')

# Compare
my_report = sv.compare([df_train, "Train"], [df_test, "Test"], "Survived")
my_report.show_html('doc/titanic_report_train_test.html')

# Some eda
df_train.groupby(['Pclass']).agg({'PassengerId':'count','Survived':'mean'})
df_train.groupby(['Sex']).agg({'PassengerId':'count','Survived':'mean'})
df_train.groupby(['Fare']).agg({'PassengerId':'count','Survived':'mean'})
df_train.groupby(['SibSp']).agg({'PassengerId':'count','Survived':'mean'})
pd.cut(df_train['Fare'], bins=3).value_counts()


df_test.groupby(['SibSp']).agg({'PassengerId':'count'})
df_test.groupby(['Pclass']).agg({'PassengerId':'count'})
df_test.groupby(['Sex']).agg({'PassengerId':'count'})
