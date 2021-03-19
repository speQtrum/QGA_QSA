import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score





def Binned_matrix(data):
    NOF = len(list(data.columns)) - 1
    new_df = pd.DataFrame()
    M = data.iloc[:,0:NOF]
    col_names = list(M.columns)


    for i in range(len(col_names)):
        new_col = M[col_names[i]].values
        uniques = len(np.unique(new_col))

        if uniques < 100:
            label_encoder = LabelEncoder()
            new_col = label_encoder.fit_transform(new_col)
            new_col = new_col + 1
            print('Label Encoding...')

        elif uniques >= 100:
            a = (np.max(new_col) - np.mean(new_col))/np.std(new_col)
            b = np.abs((np.min(new_col) - np.mean(new_col))/np.std(new_col))
            bins = np.ceil(a + b).astype(int)
            new_col = new_col.reshape(len(new_col),1)
            binning = KBinsDiscretizer(n_bins = bins,encode='ordinal')   
            new_col = binning.fit_transform(new_col)
            new_col = new_col.astype(int)
            new_col = new_col.reshape(new_col.size) + 1  
            print('Binning...') 

        new_df[col_names[i]] = new_col
        new_col = 0
        uniques = 0
    return new_df.values

def New_data(best_feature,data):
    best_feature = best_feature + [1]
    col_names = list(data.columns)
    dropped_cols = []
    for i in range(len(best_feature)):
        if best_feature[i] == 0:
            dropped_cols.append(col_names[i])

    new_data = data.drop(columns = dropped_cols)
    return new_data


def best_system(results,qubo):
    outcomes = [item[0] for item in results]
    outcome_probs = [item[1] for item in results]
    best_outcome = outcomes[outcome_probs.index(max(outcome_probs))]
    X = np.array(best_outcome)
    X_T = np.transpose(X)
    energy = np.matmul(X_T,np.matmul(qubo,X))
    return best_outcome, energy

def stringfy(bitstring):
    answer = ''
    for item in bitstring:
        answer += str(item)
    return answer


def prediction(dataset):
    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,-1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 123)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 123)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return accuracy_score(Y_test,Y_pred)*100










