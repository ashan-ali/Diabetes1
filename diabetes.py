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
rad =st.sidebar.radio("Menu",["Sample Patients Records","Making Prediction","About Us"])
if rad == "Sample Patients Records":
	st.dataframe(data) 
if rad == "Making Prediction":
	c1, c2 = st.columns(2) 
	with c1:
		g = st.number_input("Enter Glucose label ==>")
	with c2:
		bp=st.number_input("Enter BloodPressure ==>")
	c3, c4 = st.columns(2) 
	with c3:
		sk=st.number_input("Enter SkinThickness ==>")
	with c4:
		isu= st.number_input("Enter Insulin ==>")
	c5, c6 = st.columns(2) 
	with c5:
		bmi= st.number_input("Enter BMI ==>")
	with c6:
		dp= st.number_input("Enter DiabetesPedigreeFunction ==>")
	c7, c8 = st.columns(2) 
	with c7:
		age= st.number_input("Enter age ==>")
	with c8:
		name =st.text_input("Enter patient name ==>") 
	res = model.predict([[g, bp, sk, isu, bmi, dp, age]])
	t = res[0]
	if st.button("Predict Diabetes "):
		if t == 1:
			st.write(f"{name}!You have Diabetes") 
		else:
			st.write(f"{name}!No Diabetes") 
if rad == "About Us":
	st.write("Ashan Ali") 
	
	
