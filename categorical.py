import string
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def pre_process(df):
    # drop stuff we don't need, these are high cardinality (> 222) fields
    df = df.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1)
    
    # some binary fields are t/f y/n
    df['bin_3'] = df['bin_3'].map({'T': 1, 'F': 0})
    df['bin_4'] = df['bin_4'].map({'Y': 1, 'N': 0})
    
    # one-hot encode the nominal fields
    df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])
    
    # ord_5 is a bit more complicated
    ord_5_mapping = dict()
    index = 0
    for i in string.ascii_letters:
        for j in string.ascii_letters:
            ord_5_mapping[i+j] = index
            index = index + 1
    
    # ordinate the ordinal fields
    ordinal_cols_mapping = [{
        'col' : 'ord_1',    
        'mapping': {
            'Novice': 1, 
            'Contributor': 2, 
            'Expert': 3, 
            'Master': 4, 
            'Grandmaster': 5
        }}, {
        'col' : 'ord_2',    
        'mapping': {
            'Freezing': 1, 
            'Cold': 2, 
            'Warm': 3, 
            'Hot': 4, 
            'Boiling Hot': 5,
            'Lava Hot': 6
        }}, {
        'col' : 'ord_3',    
        'mapping': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
                    'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16}
        }, {
        'col' : 'ord_4',    
        'mapping': dict(zip(string.ascii_uppercase, range(1,27)))
        }, {
        'col' : 'ord_5',    
        'mapping': ord_5_mapping
        }
    ]
    encoder = ce.OrdinalEncoder(mapping=ordinal_cols_mapping, return_df=True)  
    df = encoder.fit_transform(df)
    
    return df.values

def local_dev():
    # split some of the training data off as test data because the Kaggle supplied test data has no y value to verify on...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # try a few different regularization hyperparameters
    for c in [.1, .3, .5, .9, 1]:
        # had to use , class_weight='balanced', as 0 is 2:1 more likely in y than 1, and we were getting all zeros
        lr = LogisticRegression(random_state=42, solver='liblinear', C=c, class_weight='balanced', max_iter=1000).fit(X_train, y_train)
        score = lr.score(X_test, y_test)
        print('C: ', c)
        print('accuracy: ', score)
        predictions = lr.predict(X_test)
        #print(predictions[0:100])
        #print(y_test[0:100])

        # earlier, were getting poor predictions, so did precision,recall,F1 score for more insight...
        true_positives = np.sum(np.where((y_test == 1) & (predictions == 1), 1, 0))    
        false_positives = np.sum(np.where((y_test == 0) & (predictions == 1), 1, 0))
        false_negatives = np.sum(np.where((y_test == 1) & (predictions == 0), 1, 0))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        print('precision: ', precision)
        print('recall: ', recall)
        print('F1:', f1)
        print('\n')

def submission():
    X_test_kaggle = pre_process(kaggle_test_data)
    kaggle_test_ids = kaggle_test_data['id'].values
    
    lr = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced' , max_iter=1000).fit(X, y)
    predictions = lr.predict(X_test_kaggle)
    predictions = np.concatenate([kaggle_test_ids.reshape(-1,1), predictions.reshape(-1,1)], axis=1)
    out = pd.DataFrame(data=predictions, columns=['id','target'])
    out.to_csv('out.csv', index = False)
    

train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
kaggle_test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

# extract our ground truth...
y = train_data['target'].to_numpy()
train_data = train_data.drop(['target'], axis=1)

X = pre_process(train_data)

local_dev()
#submission()
