
**#####Telco customer churn project**
According to this article, the probability of selling to a new customer is 60-70%, while the probability of selling to a new prospect is 5-20%. 
So knowing if a customer is at risk of leaving is one of the most important tasks a company has to perform in order to keep growing its business.
In this Notebook we will analyse a dataset containing information about customers of a telephone company.We will predict if a customer will churn based on his informations.

##Objective : "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]
**##Dataset description**
This dataset contains information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents

**###Exploratory data analysis**
![image](https://github.com/user-attachments/assets/0d78ab28-0614-4d35-a771-9940b483d739)
In the above image you can see that the dataset has 7043 entries and 21 columns

![image](https://github.com/user-attachments/assets/e413ec7f-e22b-4546-9f50-9706521a5fd5)
Here you can see the datatypes of the features beloonging to the dataset

![image](https://github.com/user-attachments/assets/bf6352bf-6127-4043-9cef-4aed02b0bd63)
In the above image you can see that the dataset has 18 categorical featues and 3 numerical features that will be addressed later
We  can also see the descriptive analysis of the data.

The dataset did not contain any null values.

![image](https://github.com/user-attachments/assets/3e8f2e37-8798-4b29-bf6f-d09127fe8ab2)
In image we have seen how to get the categorical and numerical features from the dataset that we will use to
plot the distributions of the data, where I started with the categorical features
![image](https://github.com/user-attachments/assets/b4ecff0d-ec61-4f67-bff3-58225711c21d)
![image](https://github.com/user-attachments/assets/e8717a92-394d-4ce7-bb78-8fa4267af3ca)
![image](https://github.com/user-attachments/assets/85bd3ebc-b217-4f90-9d7f-25b3fcdd89b6)
![image](https://github.com/user-attachments/assets/48a19574-1259-49ad-9d38-f593fd6a8b94)
With these visualisations I started to have a clear overview of the structure ofg the data and come up with
very effective customer retension strategies.I was able to know how to approach the marketing strategies by figuring out what the most significant featues are

![image](https://github.com/user-attachments/assets/adcc2ed1-115f-4a42-afaa-a6e42956d564)
In the above picture you can see the correlation Matrix of the numerical features which essentially shows how the relationship between the numerical features is.
From my analysis, I can conclude that:
*The tenure is strongly related to Monthly charges therefore both features but be very crucial in the model
*Also the senior citizen column is also stronly correlated to the Monthly charges, confirming that indeed Monthly charges is a crucial column.





**###Cardinality in features**
Due to the size of the dataset and the model I chose to use, I decided to find out the cardinality of some of the features
![image](https://github.com/user-attachments/assets/6b71f9fb-57df-4fdd-a164-c466e313ab51)
With that function I was able to accurately find the features with the highest cardinality further confirming my theory that CustomerID and TotalCharges were far too unique to be included in modelling 
due to the fact that a model can't identify any sort of pattern within the data.

**##Feature Engineering**
![image](https://github.com/user-attachments/assets/df000582-9a40-4107-b5d0-8dc3393896b0)
Here I decided to add more features to the dataset to help the model understand the patterns in the data and come up with accurate predictions

**##Encoding for the categorical features**
I used One hot encoding for encoding the categorical colums ready for modelling
![image](https://github.com/user-attachments/assets/ace0e7e5-d96f-41b8-bce7-371b601200ca)

**##Scaling for Numerical Features**
![image](https://github.com/user-attachments/assets/a8b11e30-3ec0-4491-bea0-91970b158bfb)

The resulting DataFrame had some boolean values that I converted into intergers and floats ready for clustering with KMeans
![image](https://github.com/user-attachments/assets/c0ed7b43-8299-449c-a681-473a01840c1a)


**##Customer segmentationusing KMeans Clustering**
![image](https://github.com/user-attachments/assets/bbf7cb4f-a545-470c-8d6d-0cadc9b231b5)

**##Training, Testing the model with Linear Regression**
![image](https://github.com/user-attachments/assets/2ab07dd6-f1c9-43a6-8d1f-13368734b414)

**##Accuracy Mtrics of the linear model**
![image](https://github.com/user-attachments/assets/cdfb6e91-914b-46a4-b496-0d832aa09631)

**##Confusion matrix as an accuracy evaluation tactic**
![image](https://github.com/user-attachments/assets/022be7d1-ca25-4b89-8611-71a30d099e34)

I then moved ahead to install the SHAP library used to figure out what features and the condtions that led to the model making its prediction
![image](https://github.com/user-attachments/assets/b69f2bcd-7c34-4610-a0cb-44c7dbc84c34)

I used a summaryplot to figure out which feaures impacted mostly on the models prediction
![image](https://github.com/user-attachments/assets/92162d00-b441-4fd1-ac0f-2ef62bded25c)

I also used a force plot to  to esssentially do the same function above
![image](https://github.com/user-attachments/assets/fa4d0b00-657c-4751-9791-fa6a11f7c130)

I also used a depence plot to understand what feature simpacted the models predictions
![image](https://github.com/user-attachments/assets/b8182b3d-d91a-4e5b-a8f4-42ebdf0bde79)






































Metrics and model selection
Data preprocessing
Training and evaluation
Conclusion
1. Exploratory data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
#We load the dataset
data.head().T
#An overview of our data
0	1	2	3	4
customerID	7590-VHVEG	5575-GNVDE	3668-QPYBK	7795-CFOCW	9237-HQITU
gender	Female	Male	Male	Male	Female
SeniorCitizen	0	0	0	0	0
Partner	Yes	No	No	No	No
Dependents	No	No	No	No	No
tenure	1	34	2	45	2
PhoneService	No	Yes	Yes	No	Yes
MultipleLines	No phone service	No	No	No phone service	No
InternetService	DSL	DSL	DSL	DSL	Fiber optic
OnlineSecurity	No	Yes	Yes	Yes	No
OnlineBackup	Yes	No	Yes	No	No
DeviceProtection	No	Yes	No	Yes	No
TechSupport	No	No	No	Yes	No
StreamingTV	No	No	No	No	No
StreamingMovies	No	No	No	No	No
Contract	Month-to-month	One year	Month-to-month	One year	Month-to-month
PaperlessBilling	Yes	No	Yes	No	Yes
PaymentMethod	Electronic check	Mailed check	Mailed check	Bank transfer (automatic)	Electronic check
MonthlyCharges	29.85	56.95	53.85	42.3	70.7
TotalCharges	29.85	1889.5	108.15	1840.75	151.65
Churn	No	No	Yes	No	Yes
data.shape
(7043, 21)
There are 7043 customers and 20 features. The target variable we want to be able to predict is Churn which is equal to 'Yes' if the customer churned, and 'No' otherwise

data.dtypes
customerID           object
gender               object
SeniorCitizen         int64
Partner              object
Dependents           object
tenure                int64
PhoneService         object
MultipleLines        object
InternetService      object
OnlineSecurity       object
OnlineBackup         object
DeviceProtection     object
TechSupport          object
StreamingTV          object
StreamingMovies      object
Contract             object
PaperlessBilling     object
PaymentMethod        object
MonthlyCharges      float64
TotalCharges         object
Churn                object
dtype: object
There are :

3 numeric variables – tenure, MonthlyCharges and TotalCharges
18 categorical variables (SeniorCitizen is equal to 1 if the customer is considered 'Senior' and 0 otherwise)
np.unique(data.TotalCharges.values)
array([' ', '100.2', '100.25', ..., '999.45', '999.8', '999.9'],
      dtype=object)
There are missing values in TotalCharges column that are equal to ' ', that's why it's considered as an "object". Let's correct this

data=data.replace(to_replace=" ",value=np.nan) #this will allow us to convert the feature to numeric
data.TotalCharges=pd.to_numeric(data.TotalCharges) 
data.dtypes 
customerID           object
gender               object
SeniorCitizen         int64
Partner              object
Dependents           object
tenure                int64
PhoneService         object
MultipleLines        object
InternetService      object
OnlineSecurity       object
OnlineBackup         object
DeviceProtection     object
TechSupport          object
StreamingTV          object
StreamingMovies      object
Contract             object
PaperlessBilling     object
PaymentMethod        object
MonthlyCharges      float64
TotalCharges        float64
Churn                object
dtype: object
Now dtypes are correct

np.unique(data.TotalCharges.values) #we check if ' ' values disappeared
array([18.8 , 18.85, 18.9 , ...,   nan,   nan,   nan])
 data.describe() #now let's look at basic statistics
SeniorCitizen	tenure	MonthlyCharges	TotalCharges
count	7043.000000	7043.000000	7043.000000	7032.000000
mean	0.162147	32.371149	64.761692	2283.300441
std	0.368612	24.559481	30.090047	2266.771362
min	0.000000	0.000000	18.250000	18.800000
25%	0.000000	9.000000	35.500000	401.450000
50%	0.000000	29.000000	70.350000	1397.475000
75%	0.000000	55.000000	89.850000	3794.737500
max	1.000000	72.000000	118.750000	8684.800000
This table show that there are 11 missing values in TotalCharges. On average people stay 1 month, but the dispersion of values is high with a standard deviation of 24

data.isnull().sum() #how many missing values
customerID           0
gender               0
SeniorCitizen        0
Partner              0
Dependents           0
tenure               0
PhoneService         0
MultipleLines        0
InternetService      0
OnlineSecurity       0
OnlineBackup         0
DeviceProtection     0
TechSupport          0
StreamingTV          0
StreamingMovies      0
Contract             0
PaperlessBilling     0
PaymentMethod        0
MonthlyCharges       0
TotalCharges        11
Churn                0
dtype: int64
numeric_features = data.select_dtypes(include=[np.number])
numeric_features.columns
Index(['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'], dtype='object')
categorical_features = data.select_dtypes(include=[np.object])
categorical_features.columns
Index(['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'],
      dtype='object')
data.skew()
SeniorCitizen     1.833633
tenure            0.239540
MonthlyCharges   -0.220524
TotalCharges      0.961642
dtype: float64
data.kurt()
SeniorCitizen     1.362596
tenure           -1.387372
MonthlyCharges   -1.257260
TotalCharges     -0.231799
dtype: float64
data.groupby('gender').size()
gender
Female    3488
Male      3555
dtype: int64
This feature is balanced

data.groupby('Churn').size()/len(data) # What is the percentage of churners
Churn
No     0.73463
Yes    0.26537
dtype: float64
Now, let's make some visual charts

Univariate analysis
sns.countplot(x='gender',data=data)
<matplotlib.axes._subplots.AxesSubplot at 0x1638b4624e0>

sns.countplot(x='SeniorCitizen',data=data)
<matplotlib.axes._subplots.AxesSubplot at 0x1638b4e6080>

There are 6 times less seniors

sns.countplot(x='Dependents',data=data)
#two times more people with dependents
<matplotlib.axes._subplots.AxesSubplot at 0x1638b535160>

There are two times more people with dependents

sns.distplot(data.tenure) #distribution with kernel density estimation, 20 bins by default
<matplotlib.axes._subplots.AxesSubplot at 0x1638b4abbe0>

This feature don't follow a normal distribution. The plot shows that there are many people who stay for a short time, and many people who stay for more than 60 months (5 years), this kind of distribution was expected

sns.countplot(x='Contract',data=data)
<matplotlib.axes._subplots.AxesSubplot at 0x1638d6a6ef0>

Most customers have a Month-to-month contract

sns.countplot(x='PaperlessBilling',data=data)
<matplotlib.axes._subplots.AxesSubplot at 0x1638d6e8a58>

A lot of customers have a paperless billing

ax = sns.countplot(x='PaymentMethod',data=data)
plt.setp(ax.get_xticklabels(),rotation=30)[1]

Electronic check, Mailed check, Bank transfer, Credit card

sns.countplot(x='Churn',data=data)
<matplotlib.axes._subplots.AxesSubplot at 0x1638d7690b8>

sns.distplot(data.MonthlyCharges)
<matplotlib.axes._subplots.AxesSubplot at 0x1638d741a20>

A lot of people have minimum monthly charges because they choose the cheapest offer

sns.distplot(data.dropna().TotalCharges)
<matplotlib.axes._subplots.AxesSubplot at 0x1638d86ae48>

#We plot synthesis of the charts of all the features
plt.figure(figsize=(18,18))
for k in range(1,len(categorical_features.columns)):
    plt.subplot(4,4,k)
    sns.countplot(x=categorical_features.columns[k],data=data)

Let's visualize categorical features based on Churn

Bivariate analysis
plt.figure(figsize=(18,18))
for k in range(1,len(categorical_features.columns)):
    plt.subplot(4,4,k)
    sns.countplot(x=categorical_features.columns[k],data=data,hue='Churn')

Among those who have optical fiber, there are many who churn, those who have no technical support, no device protection probably due to a lack of quality. Indeed, according to this study by Oracle, 89% churn because they look for a better quality
Those who have a month-to-month contract are more likely to churn
These will be important features for the classification
sns.countplot(x='SeniorCitizen',data=data,hue='Churn')
<matplotlib.axes._subplots.AxesSubplot at 0x1638d9069b0>

Now we will analyse important features : tenure, TotalCharges and MonthlyCharges.
It will help us answer the question : Which customers have the most value for the company ?

data.drop(columns='SeniorCitizen').groupby('InternetService').describe().T 
#we plot the statistists according to the Internet Service
InternetService	DSL	Fiber optic	No
MonthlyCharges	count	2421.000000	3096.000000	1526.000000
mean	58.102169	91.500129	21.079194
std	16.259522	12.663039	2.164221
min	23.450000	67.750000	18.250000
25%	46.200000	80.550000	19.700000
50%	56.150000	91.675000	20.150000
75%	69.900000	101.150000	20.900000
max	94.800000	118.750000	26.900000
TotalCharges	count	2416.000000	3096.000000	1520.000000
mean	2119.789259	3205.304570	665.220329
std	1880.169236	2570.220105	555.158112
min	23.450000	68.500000	18.800000
25%	435.250000	795.125000	159.875000
50%	1600.950000	2660.650000	523.675000
75%	3492.737500	5451.375000	1110.050000
max	6859.050000	8684.800000	2006.950000
tenure	count	2421.000000	3096.000000	1526.000000
mean	32.821561	32.917959	30.547182
std	24.812178	24.425332	24.356507
min	0.000000	1.000000	0.000000
25%	9.000000	9.000000	8.000000
50%	29.000000	30.000000	25.000000
75%	56.000000	56.000000	52.750000
max	72.000000	72.000000	72.000000
We have seen previously that there is a very high churn rate for those who have a fiber optic service, but we can observe in these statistics that they are also those who bring the most money to the company on average. They are definitely customers that must be kept

data.drop(columns='SeniorCitizen').groupby('Partner').describe().T
# We plot the statistics according to the 'Partner' variable
Partner	No	Yes
MonthlyCharges	count	3641.000000	3402.000000
mean	61.945001	67.776264
std	29.060087	30.875503
min	18.250000	18.400000
25%	34.050000	39.362500
50%	68.650000	74.800000
75%	85.400000	94.050000
max	118.650000	118.750000
TotalCharges	count	3639.000000	3393.000000
mean	1584.960429	3032.271648
std	1874.788687	2407.614842
min	18.850000	18.800000
25%	191.200000	929.450000
50%	811.800000	2347.900000
75%	2337.300000	5000.200000
max	8547.150000	8684.800000
tenure	count	3641.000000	3402.000000
mean	23.357869	42.017637
std	21.769526	23.698742
min	0.000000	0.000000
25%	4.000000	21.000000
50%	16.000000	46.000000
75%	39.000000	65.000000
max	72.000000	72.000000
data.drop(columns='SeniorCitizen').groupby('Dependents').describe().T
Dependents	No	Yes
MonthlyCharges	count	4933.000000	2110.000000
mean	67.002808	59.522156
std	29.274849	31.301735
min	18.250000	18.700000
25%	44.850000	24.500000
50%	73.900000	60.975000
75%	90.650000	85.950000
max	118.650000	118.750000
TotalCharges	count	4933.000000	2099.000000
mean	2187.709254	2507.955669
std	2241.593582	2309.884010
min	18.850000	18.800000
25%	323.250000	596.275000
50%	1322.550000	1551.600000
75%	3581.400000	4161.325000
max	8684.800000	8672.450000
tenure	count	4933.000000	2110.000000
mean	29.806000	38.368246
std	24.259877	24.213340
min	1.000000	0.000000
25%	7.000000	16.000000
50%	25.000000	39.000000
75%	52.000000	62.000000
max	72.000000	72.000000
data.drop(columns='SeniorCitizen').groupby('OnlineBackup').describe().T
OnlineBackup	No	No internet service	Yes
MonthlyCharges	count	3088.000000	1526.000000	2429.000000
mean	71.938002	21.079194	83.081597
std	21.021906	2.164221	21.462684
min	23.450000	18.250000	28.450000
25%	54.600000	19.700000	66.300000
50%	74.750000	20.150000	85.800000
75%	88.812500	20.900000	100.700000
max	112.950000	26.900000	118.750000
TotalCharges	count	3087.000000	1520.000000	2425.000000
mean	1828.148364	665.220329	3876.923629
std	1883.027587	555.158112	2397.985666
min	23.450000	18.800000	29.850000
25%	279.950000	159.875000	1782.400000
50%	1131.200000	523.675000	3899.050000
75%	2877.500000	1110.050000	5869.400000
max	8240.850000	2006.950000	8684.800000
tenure	count	3088.000000	1526.000000	2429.000000
mean	23.680699	30.547182	44.565253
std	21.551583	24.356507	23.223725
min	0.000000	0.000000	0.000000
25%	4.000000	8.000000	25.000000
50%	17.000000	25.000000	50.000000
75%	39.000000	52.750000	66.000000
max	72.000000	72.000000	72.000000
data.drop(columns='SeniorCitizen').groupby('Contract').describe().T
#the aim would then be to push customers to take on longer-term contracts
Contract	Month-to-month	One year	Two year
MonthlyCharges	count	3875.000000	1473.000000	1695.000000
mean	66.398490	65.048608	60.770413
std	26.926599	31.840539	34.678865
min	18.750000	18.250000	18.400000
25%	45.850000	26.900000	24.025000
50%	73.250000	68.750000	64.350000
75%	88.875000	94.800000	90.450000
max	117.450000	118.600000	118.750000
TotalCharges	count	3875.000000	1472.000000	1685.000000
mean	1369.254581	3034.683084	3728.933947
std	1613.879008	2229.730075	2571.252806
min	18.850000	18.800000	20.350000
25%	160.100000	989.350000	1278.800000
50%	679.550000	2657.550000	3623.950000
75%	2066.500000	4859.525000	5999.850000
max	8061.500000	8684.800000	8672.450000
tenure	count	3875.000000	1473.000000	1695.000000
mean	18.036645	42.044807	56.735103
std	17.689054	19.035883	18.209363
min	1.000000	0.000000	0.000000
25%	3.000000	27.000000	48.000000
50%	12.000000	44.000000	64.000000
75%	29.000000	58.000000	71.000000
max	72.000000	72.000000	72.000000
data.drop(columns='SeniorCitizen').groupby('PaymentMethod').describe().T
PaymentMethod	Bank transfer (automatic)	Credit card (automatic)	Electronic check	Mailed check
MonthlyCharges	count	1544.000000	1522.000000	2365.000000	1612.000000
mean	67.192649	66.512385	76.255814	43.917060
std	30.555200	30.612424	24.053655	26.314665
min	18.400000	18.250000	18.850000	18.700000
25%	41.087500	38.662500	60.150000	20.150000
50%	73.100000	73.025000	80.550000	34.700000
75%	92.962500	90.850000	94.700000	64.912500
max	118.750000	117.500000	118.650000	118.600000
TotalCharges	count	1542.000000	1521.000000	2365.000000	1604.000000
mean	3079.299546	3071.396022	2090.868182	1054.483915
std	2357.735491	2407.402613	2155.435320	1442.869051
min	19.250000	19.300000	18.850000	18.800000
25%	1052.362500	989.050000	308.050000	114.550000
50%	2474.650000	2453.300000	1253.900000	467.350000
75%	4943.150000	5016.250000	3340.550000	1294.125000
max	8684.800000	8670.100000	8564.750000	8331.950000
tenure	count	1544.000000	1522.000000	2365.000000	1612.000000
mean	43.656736	43.269382	25.174630	21.830025
std	23.197901	23.339581	22.382301	21.218374
min	0.000000	0.000000	1.000000	0.000000
25%	23.000000	23.000000	5.000000	3.000000
50%	48.000000	47.000000	18.000000	15.000000
75%	66.000000	66.000000	43.000000	36.000000
max	72.000000	72.000000	72.000000	72.000000
#We plot the correlation matrix, the darker a box is, the more features are correlated
plt.figure(figsize=(12,10))
corr = data.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap='Blues')

Internet service, Online security, Online Backup, DeviceProtection, Tech Support and streaming are highly correlated features
Total charges and customer ID are also very correlated, maybe the ID is chosen according to high-potential customers
The most correlated to churn : Senior, Partner, Multiple lines, online backup, Monthly charges
Now we analyse numeric features regarding the Churn

sns.catplot(x='Churn',y='tenure',data=data)
<seaborn.axisgrid.FacetGrid at 0x1638f92cd30>

Among those who have churned, there is a large part that has remained for a short time, which is logical, while for those who stayed, the distribution is fairly homogeneous

sns.catplot(x='Churn',y='MonthlyCharges',data=data)
<seaborn.axisgrid.FacetGrid at 0x1638f92c3c8>

Among those who have churned, many had high charges > 65

sns.catplot(x='Churn',y='TotalCharges',data=data)
<seaborn.axisgrid.FacetGrid at 0x1638f96f2e8>

The overall charge for those who have churned is low in general, because they tend to stay for a short time

Now let's observe the distribution of observations within categories

sns.catplot(x='Churn',y='tenure',kind='box',data=data)
<seaborn.axisgrid.FacetGrid at 0x1638f93f978>

On average, those who have churned stayed less than a year (10 months) while those who stayed have been there for 3 years
75% of those who have churned stayed less than 30 months
A quarter of loyal customers have been here for more than 5 years
sns.catplot(x='Churn',y='MonthlyCharges',kind='box',data=data)
<seaborn.axisgrid.FacetGrid at 0x1638e2b69e8>

On average, loyal customers have less monthly charges

sns.catplot(x='Churn',y='TotalCharges',kind='box',data=data)
<seaborn.axisgrid.FacetGrid at 0x1638f97cda0>

Model quality metrics
Our task is a binary classification problem with imbalanced classes. So, instead of using accuracy, we'll use AUC score, because it's easy to achieve a good accuracy by predicting only 'No', while AUC will be more relevant by taking into account the False Positive Rate and the True Positive Rate

AUC score is the Area Under ROC curve, a score close to 1 means that we have a good model. The curve is drawn by computing true positive rate against false positive rate for various tresholds
ROC-AUC

Model selection
It's a binary classification problem, we will try different models for this task and select the best according to it's performance (score, speed..) :

Logistic regression
Decision tree classifier
Support vector machines
Data preprocessing
Data as such can't be used. We'll transform the data so that we can feed it to a machine learning algorithm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV #we will do hyperparameters tuning
missing_index=np.array(data.isnull()).nonzero()[0] #indices of missing values
data.columns
Index(['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'],
      dtype='object')
#target
y=data.Churn
X=data.drop(columns=['Churn','customerID'])
y=y.drop(index=missing_index)
#label encoding for gender feature
X['gender']=X['gender'].map({'Male':1, 'Female':0})

#label encoding for other binary features
binary_variables=['Partner','Dependents','PhoneService','PaperlessBilling']
X[binary_variables]=X[binary_variables].replace({'Yes':1, 'No':0})

#we scale numeric features because they have different magnitudes which can impact the performance of our model
X['tenure']=(X['tenure']-X['tenure'].mean())/X['tenure'].std()
X['MonthlyCharges']=(X['MonthlyCharges']-X['MonthlyCharges'].mean())/X['MonthlyCharges'].std()
X['TotalCharges']=(X['TotalCharges']-X['TotalCharges'].mean())/X['TotalCharges'].std()
#categorical features with more than 2 options
other_variables=['MultipleLines','InternetService','OnlineSecurity',
                 'OnlineBackup','DeviceProtection','TechSupport',
                 'StreamingTV','StreamingMovies','Contract','PaymentMethod']
onehot_encoded=pd.get_dummies(X[other_variables])
onehot_encoded.head()
MultipleLines_No	MultipleLines_No phone service	MultipleLines_Yes	InternetService_DSL	InternetService_Fiber optic	InternetService_No	OnlineSecurity_No	OnlineSecurity_No internet service	OnlineSecurity_Yes	OnlineBackup_No	...	StreamingMovies_No	StreamingMovies_No internet service	StreamingMovies_Yes	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
0	0	1	0	1	0	0	1	0	0	0	...	1	0	0	1	0	0	0	0	1	0
1	1	0	0	1	0	0	0	0	1	1	...	1	0	0	0	1	0	0	0	0	1
2	1	0	0	1	0	0	0	0	1	0	...	1	0	0	1	0	0	0	0	0	1
3	0	1	0	1	0	0	0	0	1	1	...	1	0	0	0	1	0	1	0	0	0
4	1	0	0	0	1	0	1	0	0	1	...	1	0	0	1	0	0	0	0	1	0
5 rows × 31 columns

#features to scale to address the problem of features that are on different magnitudes
numeric=['tenure','MonthlyCharges','TotalCharges']
X=X.drop(columns=other_variables) #we drop these variables, then concatenate the table with the one-hot encoded version
X.head()
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges
0	0	0	1	0	-1.277354	0	1	-1.160241	-0.994123
1	1	0	0	0	0.066323	1	0	-0.259611	-0.173727
2	1	0	0	0	-1.236636	1	1	-0.362635	-0.959581
3	1	0	0	0	0.514215	0	0	-0.746482	-0.195234
4	0	0	0	0	-1.236636	1	1	0.197351	-0.940391
X=pd.concat([X,onehot_encoded],axis=1)
X.head()
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges	MultipleLines_No	...	StreamingMovies_No	StreamingMovies_No internet service	StreamingMovies_Yes	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
0	0	0	1	0	-1.277354	0	1	-1.160241	-0.994123	0	...	1	0	0	1	0	0	0	0	1	0
1	1	0	0	0	0.066323	1	0	-0.259611	-0.173727	1	...	1	0	0	0	1	0	0	0	0	1
2	1	0	0	0	-1.236636	1	1	-0.362635	-0.959581	1	...	1	0	0	1	0	0	0	0	0	1
3	1	0	0	0	0.514215	0	0	-0.746482	-0.195234	0	...	1	0	0	0	1	0	1	0	0	0
4	0	0	0	0	-1.236636	1	1	0.197351	-0.940391	1	...	1	0	0	1	0	0	0	0	1	0
5 rows × 40 columns

feature_names=X.columns
#There are only 11 missing values, so we can simply drop them
X=X.dropna()
#label encoding for target feature
y=y.map({'Yes':1, 'No':0})
#We split the dataset into train set (70%) and test set (30%)
l=len(X)
split=int(l*0.7) 
X_train, X_test=X[:split], X[split:]
y_train, y_test=y[:split], y[split:]

#We convert the pandas dataframes into numpy matrices
X_train, X_test=np.array(X_train), np.array(X_test)
y_train, y_test=np.array(y_train), np.array(y_test)
Training models
First we try base models

Logistic Regression
%%time
logit = LogisticRegression(random_state=75,solver='liblinear') #We set a random state so that the results are reproducible
logit.fit(X_train,y_train) #We train the model
y_logit_pred=logit.predict_proba(X_test) #We make predictions
Wall time: 47.9 ms
logit_score=roc_auc_score(y_true=y_test,y_score=y_logit_pred[:,1])
print('Logistic Regression score : {}'.format(logit_score))
Logistic Regression score : 0.8389119900833897
Decision Trees
%%time
first_tree=DecisionTreeClassifier(random_state=75)
first_tree.fit(X_train,y_train)
y1_tree_pred=first_tree.predict_proba(X_test)
Wall time: 48 ms
tree1_score=roc_auc_score(y_true=y_test,y_score=y1_tree_pred[:,1])
print('First Decision Tree Score : {}'.format(tree1_score))
First Decision Tree Score : 0.6553606040117197
This is a bad score compared to the Logistic Regression model. This is because this tree model have overfitted, we'll limit the maximum depth of the tree to avoid this

%%time
tree=DecisionTreeClassifier(max_depth=3,random_state=75)
tree.fit(X_train,y_train)
y_tree_pred=tree.predict_proba(X_test)
Wall time: 18 ms
tree_score=roc_auc_score(y_true=y_test,y_score=y_tree_pred[:,1])
print('Second Decision Tree Score : {}'.format(tree_score))
Second Decision Tree Score : 0.8148850574712644
Now the training is faster and the score is much better and comparable to that of the Logistic Regression model. Let's visualize how the tree make the splits for predictions

#We import the useful libraries to visualize the tree
from ipywidgets import Image
from io import StringIO
import pydotplus
from sklearn.tree import export_graphviz
dot_data = StringIO()
export_graphviz(tree, feature_names=feature_names,out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(value=graph.create_png())
Failed to display Jupyter Widget of type Image.

If you're reading this message in Jupyter Notebook or JupyterLab, it may mean that the widgets JavaScript is still loading. If this message persists, it likely means that the widgets JavaScript library is either not installed or not enabled. See the Jupyter Widgets Documentation for setup instructions.

If you're reading this message in another notebook frontend (for example, a static rendering on GitHub or NBViewer), it may mean that your frontend doesn't currently support widgets.

From that visualization we can observe that the type of contract and the internet service are the most discriminant feature, according to the gini criterion. As we have seen in the exploratory data analysis the type of contract and the internet service are very determinant

Support Vector Machines
%%time
svc=SVC(random_state=75, probability=True)
svc.fit(X_train,y_train)
y_svc_pred=svc.predict_proba(X_test)
C:\Users\Soriba\Desktop\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\lib\site-packages\sklearn\svm\base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
Wall time: 11.1 s
svc_score=roc_auc_score(y_true=y_test,y_score=y_svc_pred[:,1])
print('SVM score : {}'.format(svc_score))
SVM score : 0.8067089249492903
The training takes much more time (it is of the order of the second, against milliseconds for the previous ones), and the score is worse than the last two. Maybe when optimizing hyperparameters the scores will be better.

Hyperparameter optimization
To optimize the parameters we will use GridSearch, an exhaustive search with a selected subset of relevant hyperparameters. Scikit-learn has a library specially dedicated to this task : GridSearchCV, we will use it with 5-fold cross-validation 

%%time
#hyperparameter optimization
#logistic regression
from sklearn.model_selection import StratifiedKFold #we'll use stratified folds
Cs=np.logspace(-3,1,10)
skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=75)
logit_grid=GridSearchCV(estimator=logit, param_grid={'C':Cs},cv=skf,scoring='roc_auc')
logit_grid.fit(X_train, y_train)
print('best C: {} \nbest cross-validation score: {}'.format(logit_grid.best_params_,logit_grid.best_score_))
best C: {'C': 10.0} 
best cross-validation score: 0.8487436027626342
Wall time: 3.31 s
best_logit_pred=logit_grid.predict_proba(X_test) #it automatically makes a prediction with the best parameters
best_logit_score=roc_auc_score(y_true=y_test,y_score=best_logit_pred[:,1])
print('Optimized Logistic Regression score : {}'.format(best_logit_score))
Optimized Logistic Regression score : 0.838590827135452
%%time
#decision trees
max_depth_values = [k for k in range(1,15)]
max_features_values = [k for k in range(1,15)]
tree_params = {'max_depth': max_depth_values,
               'max_features': max_features_values}
tree_grid=GridSearchCV(estimator=tree, param_grid=tree_params, cv=skf, scoring='roc_auc')
tree_grid.fit(X_train, y_train)
print('best params: {} \nbest score: {}'.format(tree_grid.best_params_,tree_grid.best_score_))
best params: {'max_depth': 6, 'max_features': 12} 
best score: 0.825437758861926
Wall time: 22.7 s
best_tree_pred=tree_grid.predict_proba(X_test)
best_tree_score=roc_auc_score(y_true=y_test,y_score=best_tree_pred[:,1])
print('Optimized Decision Tree score : {}'.format(best_tree_score))
Optimized Decision Tree score : 0.8240500338066262
The scores don't really improve by optimizing the hyperparameters

Feature selection
The Logistic Regression model has the best score, now we will perform feature selection (sequentially) to see if that improves the score. Feature selection is important because it may lead to an improvement of our score by reducing overfitting, redundancy, decreasing training time etc.. To do this there is a very useful library, mlxtend that offers extensions for machine learning, we'll use the package called 'SequentialFeatureSelector'. The principle is simple :

The algorithm takes the whole d-dimensional feature set as input, we select an integer k, less than d
In the first place it will select the feature that maximizes the score
At each step it will add a new feature that, when added to the previous subset maximizes the score, and so on..
We stop adding features when there are k features (the stopping criterion)
%%time
#We choose logistic regression
#let's do sequential feature selection with the mlxtend library
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
final_model=LogisticRegression(C=10,random_state=75,solver='liblinear') # Final model with C=10 according to the result of the grid search
sfs=SFS(estimator=final_model,
        k_features=35, #here k = 35
        forward=True,
        floating=False,
        scoring='roc_auc',
        cv=skf)
sfs.fit(X_train,y_train)
Wall time: 1min 33s
print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_)) #The score and the indices of the best combination of features
best combination (ACC: 0.849): (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39)

#We train our final model with the selected features
best_X_train=X_train[:,sfs.k_feature_idx_]
best_X_test=X_test[:,sfs.k_feature_idx_]
final_model.fit(best_X_train,y_train)
final_pred=final_model.predict_proba(best_X_test)
final_score=roc_auc_score(y_true=y_test,y_score=final_pred[:,1])
print('Score of the final Logistic Regression model : {}'.format(final_score))
Score of the final Logistic Regression model : 0.8387452107279694
Conclusion
Finally, this task allowed us to identify the parameters that influence the departure of a client. It also permitted to develop a predictive model that will help the company to target more easily and quickly people that are likely to leave.

We an AUC score of 0.84 which is quite correct, optimizing the parameters didn't led to a better score. We can try to use more complex models such as Random Forest, Gradient Boosting etc. But before, there is a work to do on the data. We can improve the results by creating new features for example
