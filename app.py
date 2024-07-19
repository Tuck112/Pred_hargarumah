import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Fungsi untuk memuat dan membersihkan data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File {file_path} tidak ditemukan. Pastikan file tersebut ada di direktori yang benar.")
        return None
    df = pd.read_csv(file_path, usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])
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
    
    return lin_reg

# Fungsi untuk memprediksi harga berdasarkan input pengguna
def predict_price(model):
    st.write("## Predict House Price")
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    sqft_living = st.number_input("Square Footage of Living Area", min_value=500, max_value=10000, value=1800)
    grade = st.number_input("Grade", min_value=1, max_value=13, value=7)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2022, value=1990)
    
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, grade, yr_built]], 
                              columns=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'])
    
    prediction = model.predict(input_data)
    st.write(f"### Predicted Price: ${prediction[0]:,.2f}")

# Main function untuk aplikasi Streamlit
def main():
    st.title("House Price Analysis and Prediction")
    file_path = 'kc_house_data.csv'
    df = load_data(file_path)
    if df is not None:
        display_data_statistics(df)
        display_plots(df)
        model = linear_regression(df)
        predict_price(model)

if __name__ == "__main__":
    main()
