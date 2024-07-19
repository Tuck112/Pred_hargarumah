import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Fungsi untuk memuat dan membersihkan data
def load_data():
    df = pd.read_csv('kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])
    df['bathrooms'] = df['bathrooms'].astype('int')
    df['bedrooms'] = df['bedrooms'].replace(33, 3)
    return df

# Fungsi untuk menampilkan statistik deskriptif
def display_data_statistics(df):
    st.write("### Data Head")
    st.write(df.head())
    st.write("### Data Shape")
    st.write(df.shape)
    st.write("### Data Info")
    st.write(df.info())
    st.write("### Data Description")
    st.write(df.describe())

# Fungsi untuk menampilkan plot
def display_plots(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(df['bedrooms'], ax=ax[0])
    ax[1].boxplot(df['bedrooms'])
    st.pyplot(fig)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(df['bathrooms'], ax=ax[0])
    ax[1].boxplot(df['bathrooms'])
    st.pyplot(fig)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df['sqft_living'].plot(kind='kde', ax=ax[0])
    ax[1].boxplot(df['sqft_living'])
    st.pyplot(fig)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(df['grade'], ax=ax[0])
    ax[1].boxplot(df['grade'])
    st.pyplot(fig)
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.countplot(df['yr_built'], ax=ax[0])
    ax[1].boxplot(df['yr_built'])
    st.pyplot(fig)
    
    sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'], y_vars=['price'], height=5, aspect=0.75)
    st.pyplot()

# Fungsi untuk melakukan regresi linier
def linear_regression(df):
    x = df.drop(columns='price')
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    
    coef_dict = {'features': x.columns, 'coef_value': lin_reg.coef_}
    coef = pd.DataFrame(coef_dict, columns=['features', 'coef_value'])
    
    y_pred = lin_reg.predict(x_test)
    score = lin_reg.score(x_test, y_test)
    
    st.write("### Coefficients")
    st.write(coef)
    st.write("### Intercept")
    st.write(lin_reg.intercept_)
    st.write("### R^2 Score")
    st.write(score)
    st.write("### Prediction for [3, 2, 1800, 7, 1990]")
    st.write(lin_reg.predict([[3, 2, 1800, 7, 1990]]))

# Main function untuk aplikasi Streamlit
def main():
    st.title("House Price Analysis and Prediction")
    df = load_data()
    display_data_statistics(df)
    display_plots(df)
    linear_regression(df)

if __name__ == "__main__":
    main()
