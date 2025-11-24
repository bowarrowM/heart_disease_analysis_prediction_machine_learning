from all_imports import *

model = joblib.load('heart_disease_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("AI Powered Heart Disease Screening App")
st.write("Enter patient vitals below to generate a heart disease risk prediction.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalch = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina?", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)

# Processing the Input to match the Model
if st.button("Analyze Patient Risk"):
    
    # Creates a dataframe with all columns initialized to 0
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 # Initializes row with 0s
    
    input_data['age'] = age
    input_data['chol'] = chol
    input_data['thalch'] = thalch
    input_data['trestbps'] = trestbps
    input_data['oldpeak'] = oldpeak
    
    # Categorical Mappings
    # Sex
    if sex == 'Male':
        if 'sex_Male' in input_data.columns: input_data['sex_Male'] = 1
        
    if exang == 'Yes':
        if 'exang_True' in input_data.columns: input_data['exang_True'] = 1
        
    # Chest Pain 
    if cp == 'Atypical Angina':
        if 'cp_atypical angina' in input_data.columns: input_data['cp_atypical angina'] = 1
    elif cp == 'Non-anginal Pain':
        if 'cp_non-anginal pain' in input_data.columns: input_data['cp_non-anginal pain'] = 1
    elif cp == 'Asymptomatic':
        if 'cp_asymptomatic' in input_data.columns: input_data['cp_asymptomatic'] = 1

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of 1

    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK DETECTED (Probability: {probability:.2%})")
        st.write("Recommendation: Further cardiological consultation is advised.")
    else:
        st.success(f"✅ LOW RISK DETECTED (Probability: {probability:.2%})")
        st.write("Recommendation: Standard annual monitoring.")