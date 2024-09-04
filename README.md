
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
