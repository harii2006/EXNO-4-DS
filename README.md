# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("bmi.csv")
df

df.head()

df.dropna()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Hs','Ws']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df[['Hm','Wm']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)

import pandas as pd
df = pd.read_csv("titanic_dataset.csv")
df.columns

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Selecting features and target
X = df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Feature selection
selector = SelectKBest(score_func=f_classif, k=4)

# Fit and transform
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

print("Selected Features:")
print(selected_features)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

tips.time.unique()

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="260" height="359" alt="image" src="https://github.com/user-attachments/assets/4e6a0ebc-25a2-4f53-8f80-a34747ab320f" />

<img width="379" height="181" alt="image" src="https://github.com/user-attachments/assets/d82d87f0-2d8b-46c1-bf60-9fe15ea2457a" />


<img width="447" height="363" alt="image" src="https://github.com/user-attachments/assets/aced50c8-6043-487b-b8a2-036550b6de43" />


<img width="432" height="305" alt="image" src="https://github.com/user-attachments/assets/5748d4fd-dcbc-4b84-8326-ee97cea76436" />

<img width="534" height="302" alt="image" src="https://github.com/user-attachments/assets/f922c277-7991-4286-9f41-96a3633fe47d" />

<img width="628" height="77" alt="image" src="https://github.com/user-attachments/assets/bfeb0293-a012-487c-ba94-6c456cfd69c1" />

<img width="465" height="61" alt="image" src="https://github.com/user-attachments/assets/27d06c72-c60b-413a-89f1-4d0dfd9baa89" />

<img width="385" height="181" alt="image" src="https://github.com/user-attachments/assets/86a48175-863f-4fc9-b204-eec4d7c6dcf0" />

<img width="389" height="58" alt="image" src="https://github.com/user-attachments/assets/04d7bb71-1c4f-4af9-9aff-759154532796" />

<img width="225" height="81" alt="image" src="https://github.com/user-attachments/assets/f4f1838a-0824-4ec7-a132-b3226ee6bd2e" />

<img width="330" height="39" alt="image" src="https://github.com/user-attachments/assets/fe4ccb5c-aa38-4b4a-841d-de13233856b5" />



 
# RESULT:

   Thus, Feature selection and Feature scaling has been used on the given dataset.
