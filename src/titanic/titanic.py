
class DataPreprocessor:
    def __init__(self):
        self.features = None

    def recode(self, df):

        if 'SibSp' in df.columns:
            df['SibSp'] = df['SibSp'].map(lambda x: x if x < 4 else 4)
        if 'Parch' in df.columns:
            df['Parch'] = df['Parch'].map(lambda x: x if x < 3 else 3)


