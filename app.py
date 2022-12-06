import streamlit as st
import joblib 
import pandas as pd

Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")
data = pd.read_csv("train_ctrUa4K.csv")
data.drop(columns=['Loan_ID','Loan_Status'],inplace=True)
def predict(gender, married, dependents, education, self_employed,applicant_income, coapplicant_income, loanamount,loan_amount_term,credit_history,property_area):
    test_df = pd.DataFrame(columns = Inputs)
    test_df.at[0,"Gender"] = gender
    test_df.at[0,"Married"] = married
    test_df.at[0,"Dependents"] = dependents
    test_df.at[0,"Education"] = education
    test_df.at[0,"Self_Employed"] = self_employed
    test_df.at[0,"ApplicantIncome"] = int(applicant_income)
    test_df.at[0,"CoapplicantIncome"] = float(coapplicant_income)
    test_df.at[0,"LoanAmount"] = float(loanamount)
    test_df.at[0,"Loan_Amount_Term"] = float(loan_amount_term)
    test_df.at[0,"Credit_History"] = float(credit_history)
    test_df.at[0,"Property_Area"] = str(property_area)
    return Model.predict(test_df)[0]

def main():
    st.title("Zomato Resturants App")
    gender = st.selectbox("Gender" , ['Male', 'Female'])
    married = st.selectbox("Married" , ['Yes', 'No'])
    dependents = st.selectbox("Dependents" , ['0', '1','2','3+'])
    education = st.selectbox("Education" , ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed" , ['Yes', 'No'])
    applicant_income = st.slider("ApplicantIncome" , min_value=50, max_value=5000, value=0, step=1)
    coapplicant_income = st.slider("CoapplicantIncome" , min_value=50, max_value=5000, value=0, step=1)
    loanamount = st.slider("LoanAmount" , min_value=9, max_value=700, value=0, step=1)
    loan_amount_term = st.selectbox("Loan Amount Term" , ['12.0', '36.0','60.0','84.0','120.0','180.0','240.0','300.0','480.0','360.0'])#float
    credit_history = st.selectbox("Credit History" , ['0.0', '1.0'])#float
    property_area = st.selectbox("Property Area" , ['Urban', 'Semiurban','Rural'])
    if st.button("Predict"):
        result = predict(gender, married, dependents, education, self_employed,applicant_income, coapplicant_income, loanamount,loan_amount_term,credit_history,property_area)
        label = ["Fail","Success"]
        st.text("The output is {}".format(label[result]))
if __name__ == '__main__':
    main()
