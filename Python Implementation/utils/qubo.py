import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



def Mutual_Information1(X):
    mutual_info = drv.information_mutual(X.T)
    return mutual_info



def Mutual_Information2(X):
    mutual_info = drv.information_mutual(X.T)
    for i in range(mutual_info.shape[0]):
        mutual_info[i,i] = 0
    return mutual_info



def Diagonal_Matrix1(X,data):  # Output Class is Categorical String
    NOF = X.shape[1]
    D = np.zeros([NOF,NOF])
    class_label_encoder = LabelEncoder()
    labeled_class = class_label_encoder.fit_transform(data.iloc[:,NOF])
    labeled_class = labeled_class + 1
    print(labeled_class)
    for i in range(NOF):
        diag_element = drv.information_mutual(X[:,i], labeled_class)
        D[i,i] = diag_element
    return D



def mRMR(D,M):       #-------- minimun Redundancy Maximum Relevance
    NOF = D.shape[0]
    return ((M/NOF)-D)



def mRMR1(binned_data, data):
    D = Diagonal_Matrix1(binned_data,data)
    MI = Mutual_Information2(binned_data.astype(int))
    NOF = D.shape[0]
    return ((MI/NOF)-D)