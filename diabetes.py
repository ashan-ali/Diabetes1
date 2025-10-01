import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#----------------------------------------------------
data = pd.read_csv("diabetes.csv")

x = data.loc[:, 'Glucose' : 'Age' ]
y = data.loc[:, ['Outcome']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)

model = LogisticRegression()

model.fit(x_train, y_train)

#-------------------------streamlit code----
st.title("Diabetes Prediction")
st.write("=" * 33)

g = st.number_input("Enter Glucose label ==>")
bp = st.number_input("Enter BloodPressure ==>")
sk = st.number_input("Enter SkinThickness ==>")
isu = st.number_input("Enter Insulin ==>")
bmi = st.number_input("Enter BMI ==>")
dp = st.number_input("Enter DiabetesPedigreeFunction ==>")
age = st.number_input("Enter age ==>")

res = model.predict([[g, bp, sk, isu, bmi, dp, age]])

if st.button("Predict Diabetes "):
	st.write(f"you have {res}")
	