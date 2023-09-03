# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```
#read the dataset
df=pd.read_csv('Churn_Modelling data.csv')
df
```
```
#drop unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
```
```
#checking for null, duplicates, outliers in lasrt column
df.isnull().sum()

df.duplicated()

df['Exited'].describe()
```
```
#normalising data to normal distribution
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
df2
```
```
#split dataset
x=df2.iloc[:,:-1].values #all rows from all except last column
x
y=df2.iloc[:,-1].values #all rows from only last column
y
```
```
##creating training and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
```
```
print(X_test)
print("Size of X_test: ",len(X_test))
```

## OUTPUT:
## Dataset and Its Properties
![1](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/96f5d358-5d3f-4139-afdb-92c24538b61d)
![2](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/79aa36e3-a048-4b5f-96c0-a25b59aaf0e1)
![3](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/c959f281-1139-4c01-badd-f27f3d58b87f)

## Normalised Dataset
![6](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/55492f6a-aad8-44af-bb48-429d775f2c96)
## X and Y Column Data
![7](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/1158ee81-2387-442d-a2b0-2381a96f8eff)
![8](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/64241af8-cf89-49a6-8d8b-d4d8a96d3c16)
## Training Data
![9](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/8ca958f4-158c-4efe-8e91-ec787d2a2802)
## Test Data
![10](https://github.com/subalakshmivenkat/Ex.No.1---Data-Preprocessing/assets/119393477/ad5e6107-6c51-49ed-acd6-5849e8ddd958)

## RESULT
Thus, the Data preprocessing is performed over a data set successfully.
