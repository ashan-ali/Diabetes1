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
st.image("a-doctor-writing-the-word-diabetes.jpg") 
st.write("=" * 33)
st.sidebar.title("Sample Patients Record ") 
if st.sidebar.button("!!Click!!"):
	st.sidebar.dataframe(data) 
g, bp = st.columns(2) 
g.number_input("Enter Glucose label ==>")
bp.number_input("Enter BloodPressure ==>")
sk, isu = st.columns(2) 
sk.number_input("Enter SkinThickness ==>")
isu.number_input("Enter Insulin ==>")
bmi, dp = st.columns(2) 
bmi.number_input("Enter BMI ==>")
dp.number_input("Enter DiabetesPedigreeFunction ==>")
age, name = st.columns(2) 
age.number_input("Enter age ==>")
name.text_input("Enter patient name ==>") 

res = model.predict([[g, bp, sk, isu, bmi, dp, age]])
t = res[0]

if st.button("Predict Diabetes "):
	if t == 1:
		st.write(f"{name}!You have Diabetes") 
	else:
		st.write(f"{name}!No Diabetes") 
	
	
