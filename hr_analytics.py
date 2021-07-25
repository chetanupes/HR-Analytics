import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(layout="wide")
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_pandas_profiling import st_profile_report
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Widgets libraries
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive

#Data Analysis Library
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import model_selection
from yellowbrick.model_selection import FeatureImportances
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from streamlit_yellowbrick import st_yellowbrick

# Common sklearn Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# from sklearn.datasets import make_classification

# sklearn modules for performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score

##########################################################################################################################################
# Problem Preview

#st.title('HCM-Analytics')
st.markdown("<h1 style='text-align: center; color:black;'>HCM-Analytics</h1>", unsafe_allow_html=True)
st.write('This project demonstrates how analytics can be used to tackle a costly problem of Employee Attrition. '
'Understanding why and when employees are most likely to leave can lead to actions to improve employee retention as well as possibly planning new hiring in advance.')
st.write('In this work, we will attempt to get insights about:')

st.write('1. What are the key indicators about an employee leaving the company?')
st.write('Data used in this study was retrieved from IBM Website') 
st.markdown( """<a href="https://developer.ibm.com/technologies/data-science/patterns/data-science-life-cycle-in-action-to-solve-employee-attrition-problem/?mhsrc=ibmsearch_a&mhq=hr%20analytics%20dataset">Dataset Link</a>""", unsafe_allow_html=True,)

#Loading the data
data=pd.read_csv('HR-Employee-Attrition.csv')

#Creating a profile report
st.title('Dataset Report')

profile_hr=ProfileReport(data, title='Profiling Reoprt')

if st.checkbox('Preview Profile Report'):
    st_profile_report(profile_hr)
st.text('The profiling function presents the user with a descriptive statistical summary of all the features of the dataset.')

#Getting insights into the dataset

new_col=['Department', 'EducationField', 'Gender', 'JobRole',
       'MaritalStatus', 'OverTime','JobLevel','WorkLifeBalance','StockOptionLevel','EnvironmentSatisfaction','JobSatisfaction','RelationshipSatisfaction','BusinessTravel']

#Creating a dataframe for each attribute which explores the dataset by the people left the company

st.title('Data visualization by target: Attrition')
st.write('Exploring the dataset to understand employees who left the company')

st.sidebar.markdown("# Data Vizualization")
Select_viz= st.sidebar.selectbox('Select a variable',('Categorical', 'Numerical'))


df_all=[]
for i in new_col:
    df_ori=data[[i,'DailyRate']].groupby(i).count()
    df_ori.reset_index(inplace=True, drop=False)
    
    df=data[data['Attrition']=='Yes']
    df=df[[i,'DailyRate']].groupby(i).count()
    df.reset_index(inplace=True, drop=False)
    
    df_fin=df.merge(df_ori, on=i, how='left')
    df_fin['Total Percentage']=df_fin['DailyRate_x']*100/df_fin['DailyRate_y']
    
    df_all.append(df_fin)

#Visulalizing categories by percentage of people left

def viz(cat):
    if cat=='Categorical':
        st.write('Categorical Variables')
        for i in range(len(df_all)):
        #fig=make_subplots(rows=len(new_col),cols=1)
    
            fig=px.bar(df_all[i], x=df_all[i].iloc[:,0], y=df_all[i].iloc[:,3], color_discrete_sequence =['pink']*len(df_all[i]), text=round(df_all[i].iloc[:,3],2))
            fig.update_layout(showlegend=False, autosize=True, title_text='Percentage of people left by {}'.format(new_col[i]))
    
            fig.update_yaxes(title_text='Percentage (%)')  
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            fig.update_xaxes(title_text='Category: {}'.format(new_col[i]), visible=True)
            fig.update_layout(yaxis_range=[0, 45])

            # Plot!
            st.plotly_chart(fig, use_container_width=True)
    else:
        # 2. Numerical variables
        st.write('Numerical Variables')
        num_cols=['Age','MonthlyIncome','DistanceFromHome','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','TotalWorkingYears','YearsWithCurrManager' ]
        import plotly.figure_factory as ff
        for i in num_cols:
            df1=data[data['Attrition']=='No']
            df2=data[data['Attrition']=='Yes']
    
            group_labels = ['Current Employees', 'Ex Employees']
            fig1=ff.create_distplot([df1[i],df2[i]],group_labels, show_hist=False,show_rug=False)
            fig1.update_yaxes(title_text='Density') 
            fig1.update_xaxes(title_text='Category: {}'.format(i), visible=True)
    
            #Plot
            st.plotly_chart(fig1, use_container_width=True)

viz(Select_viz)
##################################################################################################################################################################################

#Modeling
# Create a label encoder object
le = LabelEncoder()
df_HR=data.copy()

# Label Encoding will be used for columns with 2 or less unique values

for col in df_HR.columns[1:]:
    if df_HR[col].dtype == 'object':
        if len(list(df_HR[col].unique())) <= 2:
            le.fit(df_HR[col])
            df_HR[col] = le.transform(df_HR[col])

# convert rest of categorical variable into dummy
df_HR = pd.get_dummies(df_HR, drop_first=True)

#Feature Scaling
# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
HR_col = list(df_HR.columns)
HR_col.remove('Attrition')
for col in HR_col:
    df_HR[col] = df_HR[col].astype(float)
    df_HR[[col]] = scaler.fit_transform(df_HR[[col]])
df_HR['Attrition'] = pd.to_numeric(df_HR['Attrition'], downcast='float')
# assign the target to a new dataframe and convert it to a numerical feature
target = df_HR['Attrition'].copy()

# let's remove the target feature and redundant features from the dataset
df_HR.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber',
            'StandardHours', 'Over18'], axis=1, inplace=True)
print('Size of Full dataset is: {}'.format(df_HR.shape))

# Since we have class imbalance (i.e. more employees with turnover=0 than turnover=1)
# let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(df_HR,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=7,
                                                    stratify=target)  

#Logistic Regression

kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
modelCV = LogisticRegression(solver='liblinear',
                             class_weight="balanced", 
                             random_state=7)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(class_weight = "balanced",random_state=7)

#Display top ten features based on model selected

# Add a selectbox to the sidebar:
st.sidebar.markdown("# Machine Learning Models")
Select_method = st.sidebar.selectbox('Select a model',('Random Forest','Logistic Regression'))


def model(sel):
    if sel=='Logistic Regression':
        #fig, ax = plt.subplots(figsize=(20, 10))
        viz = FeatureImportances(modelCV,relative=False,stack=False, topn=10)
        viz.fit(X_train, y_train)
        #Plot
        st_yellowbrick(viz)
    else:
        #fig, ax = plt.subplots(figsize=(20, 10))
        viz1 = FeatureImportances(rf_classifier,relative=False,stack=False, topn=10)
        viz1.fit(X_train, y_train)
        #Plot
        st_yellowbrick(viz1)

st.title('Few Inference from the Visual Analysis')

st.write('1. Single employees show the largest proportion of leavers, compared to Married and Divorced counterparts.')
st.write('2. People who travel frequently show higher proportion of leavers compared to their counterparts. This also alligns well with high attrition in the Sales Department.')
st.write('3. People who have to work overtime show higher proportion of leavers compared to their counterparts.')
st.write('4. Lower the job satisfaction the wider the gap by attrition status in the levels of income.')
st.write('5. Higher the total working years the higher the monthly income of an employee.')
st.write('6. Higher the percent salary hike the higher the performance rating.')
st.write('7. Higher the years with current manager the higher the years since last promotion.')
st.write('8. Higher the age the higher the monthly income.')

st.title('Top 10 reasons contributing to Employees Attrition/Retention')
model(Select_method)

st.title('Recommendations')
st.write('As more data is generated about the employees (New Joiners/recent Leavers) the algorithm can be re-trained using the additional data and theoritically generate more accurate predictions to identify high-risk employees of leaving based on the probabilistic label assigned to each feature variable (i.e. employee) by the algorithm.')


st.write('Employees then can be assigned to a "Risk Category" based on the predicted label such that:')

st.write('1. Low-risk for employees with label < 0.6')
st.write('2. Medium-risk for employees with label between 0.6 and 0.8')
st.write('3. High-risk for employees with label > 0.8')


