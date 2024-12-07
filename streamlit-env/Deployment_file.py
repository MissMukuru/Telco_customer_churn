import streamlit as st 
import pandas as pd
import joblib
import seaborn as sns 
import matplotlib.pyplot as plt         


st.title("Telco Customer Churn App ")

st.sidebar.tittle('Navigation')
option = st.sidebar.radio('Where Would you like to navigate to?', ['Homepage', 'Infomation On Data', 'Data visualisations', 'Churn Prediction'])

df = pd.read_csv(r'C:\Users\HPPC\Desktop\ML PROJECTS\Telco-Customer-Churn.csv')

if option == 'Homepage':
    st.title('HOMEPAGE')
    st.write('''Welcome to the homepage, In today’s competitive business landscape,
             retaining customers is more important than ever.
             Our Customer Churn Prediction solution helps you identify at-risk customers and take proactive steps to keep them engaged. 
             Using advanced machine learning models and predictive analytics, we analyze key customer behavior patterns, transaction history, 
             and engagement data to forecast which customers are most likely to leave your business,
            With these actionable insights, you can:

    Improve Retention: Target customers with personalized offers and communication.
    Maximize Revenue: Focus on high-value customers before they churn.
    Optimize Resources:
    Invest in the right strategies to retain your customer base efficientlyhu
    ''')

elif: option == 'Infomation On Data':
    st.write("### Data Information")
    st.write('This is the information the data provided after much querying.The inghts gained were used in compiling churn reduction strategies')
    if st.checkbox('Do you want to check the missingg values in this data?'):
        st.write(df.isnull().null())

    if st.checkbox("check the total number of duplicate values in this data"):
        st.write(df.duplicated())

    if st.checkbox("check duplicated totals"):
        st.write(df.dulplicated().sum())

    if st.checkbox('Number of rows and columns?'):
        st.write(df.shape)

elif: option = "Data visualisations":
    st.write('## Data visualisation')

    st.write("""Data Visualisation for the Categorical features(columns)
    Pie charts and countplots were used""")

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(exclude='object').columnns

    def plot_categorical_data(dataframe):
    number_cols = len(cat_cols)
    fig, axes = plt.subplots(nrows = number_cols, ncols = 2, figsize =  (10,6 *  len(num_cols)))

    for i, col in enumerate(cat_cols):
        value_counts = dataframe[col].value_counts()
        axes[i, 0].pie(value_counts, labels = value_counts.index, autopct = '%1.1f%%')
        axes[i, 0].  set_title(f'Distribution of {col}')

        sns.countplot(data = dataframe, x = col, ax = axes[i, 1])
        axes[i, 0].set_title(f'Count Plot of {col}')
x
    plt.tight_layout()
    plt.show()

    return dataframe

    df = plot_categorical_data(df)

    st.write('## Corrlation matrix between the numerical columns')
    correlation_matrix =  df.corr(numeric_only = True)
    plt.figure(figsize = (10,6))
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


elif option == 'Churn Prediction':
st.write('Now that we understand the data in question we can move onto using the model to predict the customer churn in this app')

st.write("In modelling I chose a few models to try and predict customer churn in, I used Decision trees, Random forest and Linear regression")

st.write('I picked the Linear Regression model as my prediction model because it showed better values in Evaluation Metrics compared to the other models')
    
st.write('## WELCOME TO THE MODEL PREDICTION')

linear_regression = joblib.load('C:\Users\HPPC\Desktop\Telco_customer_churn-main\best_model.pkl')

st.write('The model was loaded successfully')

st.write('Please input your details here: ')





