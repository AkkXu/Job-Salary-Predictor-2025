import streamlit as st
import pandas as pd
import numpy as np
import joblib

#add all model and encoder
model = joblib.load('linear_model.pkl')
encoder = joblib.load('onehot_encoder.pkl')
mlb = joblib.load('multilabel_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("AI Job Salary Predictor 2025")

#input for user
st.subheader("Enter Job Details")
job_title = st.selectbox("Job Title", ['AI Research Scientist', 'AI Software Engineer', 'AI Specialist',
       'NLP Engineer', 'AI Consultant', 'AI Architect',
       'Principal Data Scientist', 'Data Analyst',
       'Autonomous Systems Engineer', 'AI Product Manager',
       'Machine Learning Engineer', 'Data Engineer', 'Research Scientist',
       'ML Ops Engineer', 'Robotics Engineer', 'Head of AI',
       'Deep Learning Engineer', 'Data Scientist',
       'Machine Learning Researcher', 'Computer Vision Engineer'])
experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'])
employment_type = st.selectbox("Employment Type", ['CT', 'FL', 'PT', 'FT'])
remote_ratio = st.slider("Remote Ratio [0 is on-site, 50 is Hybrid, 100 is remote] (%)", 0, 100, 100)
company_location = st.selectbox("Company Location", ['China', 'Canada', 'Switzerland', 'India', 'France', 'Germany',
       'United Kingdom', 'Singapore', 'Austria', 'Sweden', 'South Korea',
       'Norway', 'Netherlands', 'United States', 'Israel', 'Australia',
       'Ireland', 'Denmark', 'Finland', 'Japan'])
company_size = st.selectbox("Company Size", ["S", "M", "L"])
employee_residence = st.selectbox("Employee Residence", ['China', 'Ireland', 'South Korea', 'India', 'Singapore', 'Germany',
       'United Kingdom', 'France', 'Austria', 'Sweden', 'Norway',
       'Israel', 'United States', 'Netherlands', 'Denmark', 'Switzerland',
       'Finland', 'Japan', 'Canada', 'Australia'])
skills = st.multiselect("Required Skills", ['AWS', 'Azure', 'Computer Vision', 'Data Visualization', 'Deep Learning', 'Docker', 'GCP', 'Git', 'Hadoop', 'Java', 'Kubernetes', 'Linux', 'MLOps', 'Mathematics', 'NLP', 'PyTorch', 'Python', 'R', 'SQL', 'Scala', 'Spark', 'Statistics', 'Tableau', 'TensorFlow'])
education_required = st.selectbox("Education Level", ['Bachelor', 'Master', 'Associate', 'PhD'])
industry = st.selectbox("Industry", ['Automotive', 'Media', 'Education', 'Consulting', 'Healthcare',
       'Gaming', 'Government', 'Telecommunications', 'Manufacturing',
       'Energy', 'Technology', 'Real Estate', 'Finance', 'Transportation',
       'Retail'])
company_name = st.selectbox("Company Name", ['Smart Analytics', 'TechCorp Inc', 'Autonomous Tech',
       'Future Systems', 'Advanced Robotics', 'Neural Networks Co',
       'DataVision Ltd', 'Cloud AI Solutions', 'Quantum Computing Inc',
       'Predictive Systems', 'AI Innovations', 'Algorithmic Solutions',
       'Cognitive Computing', 'DeepTech Ventures',
       'Machine Intelligence Group', 'Digital Transformation LLC'])


#Prediction
if st.button("Predict Salary"):
    #categorical input transform to DataFrame
    input_cat = pd.DataFrame([{
        "job_title": job_title,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_location": company_location,
        "employee_residence": employee_residence,
        "company_size": company_size,
        "education_required": education_required,
        "industry": industry,
        "company_name": company_name,
    }])
    #encoding skills with onehot
    input_cat = input_cat.reindex(columns=encoder.feature_names_in_)

    encoded_cat = encoder.transform(input_cat)
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())

    #encoding skills with multihot
    skill_list = [s.strip() for s in skills]
    encoded_skill = mlb.transform([skill_list])
    skill_df = pd.DataFrame(encoded_skill, columns=mlb.classes_)

    # 3.add numeric columns
    numeric_df = pd.DataFrame([[remote_ratio, len(skill_list)]], columns=["remote_ratio", "skill_count"])

    # 4.combine all columns
    X_input = pd.concat([encoded_cat_df, skill_df, numeric_df], axis=1)
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)

    # 5.predict
    log_salary = model.predict(X_input)[0]
    salary = np.expm1(log_salary)
    st.success(f"Estimated Salary: ${salary:,.2f} USD")
