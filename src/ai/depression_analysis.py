import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os






def run_depression_analysis():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    dataset = pd.read_csv('data/depression_analysis/depression_data.csv')
    print(dataset)
    dataset.head()


# dataset = pd.read_csv('/kaggle/input/depression-dataset/depression_data.csv')
# dataset.head()


# dataset.describe()

# dataset.info()


# dataset.drop(['Name'], axis=1, inplace=True)


# dataset_sample = dataset.sample(n=50000, random_state=1)

# X = dataset_sample.iloc[:, :-1].values
# y = dataset_sample.iloc[:, -1].values


# from sklearn.preprocessing import LabelEncoder
# le_employement = LabelEncoder()
# X[:, 6] = le_employement.fit_transform(X[:, 6])




# le_mental_illness = LabelEncoder()
# X[:, 11] = le_mental_illness.fit_transform(X[:, 11])


# le_substance_abuse = LabelEncoder()
# X[:, 12] = le_substance_abuse.fit_transform(X[:, 12])


# le_fam_history = LabelEncoder()
# X[:, 13] = le_fam_history.fit_transform(X[:, 13])
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Applying one hot encoding on various columns
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 4, 5, 8, 9, 10])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# #Spliting dataset into training and testing
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# #Applying normalisation¶
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# #Model testing
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

# #Creating model dictionary