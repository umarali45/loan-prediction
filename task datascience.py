#!/usr/bin/env python
# coding: utf-8

# # Loan Approval Prediction
# - Defining the problem statement
# - Collecting the data
# - Exploratory data analysis
# - Feature engineering
# - Modelling
# - Confusion Matrix

# # 1. Defining the problem statement
# - About Company:
# 
# Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.
# 
# - Problem:
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

# # 2. Collecting the data
# 
#  
# ### load train, test dataset using Pandas
# 
# ### data set description
# - Variable	Description
# - Loan_ID	Unique Loan ID
# - Gender	Male/ Female
# - Married	Applicant married (Y/N)
# - Dependents	Number of dependents
# - Education	Applicant Education (Graduate/ Under Graduate)
# - Self_Employed	Self employed (Y/N)
# - ApplicantIncome	Applicant income
# - CoapplicantIncome	Coapplicant income
# - LoanAmount	Loan amount in thousands
# - Loan_Amount_Term	Term of loan in months
# - Credit_History	credit history meets guidelines
# - Property_Area	Urban/ Semi Urban/ Rural
# - Loan_Status	Loan approved (Y/N)

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ### import python lib for visualization

# In[2]:


import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


cd D:\SEM 7\AI\AI project


# In[4]:


df = pd.read_csv("Loan Prediction Dataset.csv")


# # 3. Exploratory data analysis
# Printing first 5 rows of the train dataset.

# In[5]:


df.head()


# We can see there are total 13 columns including target variable, all of them are self explanatory.

# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


# find the null values
df.isnull().sum()


# We also see some missing values, lets take stock of missing columns and what are the possible values for categorical and numerical columns

# # Filling missing values

# In[10]:


# fill the missing values for numerical terms - mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[47]:


#  fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])


#  # categorical attributes visualization

# In[12]:


sns.countplot(df['Gender'])


# Sex: There are more Men than Women (approx. 3x)

# In[13]:


sns.countplot(df['Married'])


# Martial Status: 2/3rd of the population in the dataset is Marred; Married applicants are more likely to be granted loans.

# In[14]:


sns.countplot(df['Dependents'])


# Dependents: Majority of the population have zero dependents and are also likely to accepted for loan.

# In[15]:


sns.countplot(df['Education'])


# Education: About 5/6th of the population is Graduate and graduates have higher propotion of loan approval

# In[16]:


sns.countplot(df['Self_Employed'])


# Employment: 5/6th of population is not self employed.

# In[17]:


sns.countplot(df['Property_Area'])


# Property Area: More applicants from Semi-urban and also likely to be granted loans.

# In[18]:


sns.countplot(df['Loan_Status'])


# Loan Approval Status: About 2/3rd of applicants have been granted loan.

# In[19]:


categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=df,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# Applicant with credit history are far more likely to be accepted.
# Loan Amount Term: Majority of the loans taken are for 360 Months (30 years).

# # numerical attributes visualization & Normalization

# In[20]:


sns.distplot(df["ApplicantIncome"])


# In[21]:


sns.distplot(df["CoapplicantIncome"])


# In[22]:


sns.distplot(df["LoanAmount"])


# In[23]:


sns.distplot(df['Loan_Amount_Term'])


# In[24]:


sns.distplot(df['Credit_History'])


# # 4. Feature engineering
# 
# Feature engineering is the process of using domain knowledge of the data  
# to create features (**feature vectors**) that make machine learning algorithms work.  

# In[25]:


# total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# ## Log Transformation

# In[26]:


# apply log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
sns.distplot(df["ApplicantIncomeLog"])


# In[27]:


df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome'])
sns.distplot(df["ApplicantIncomeLog"])


# In[28]:


df['LoanAmountLog'] = np.log(df['LoanAmount'])
sns.distplot(df["LoanAmountLog"])


# In[29]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
sns.distplot(df["Loan_Amount_Term_Log"])


# In[30]:


df['Total_Income_Log'] = np.log(df['Total_Income'])
sns.distplot(df["Total_Income_Log"])


# ## Coorelation Matrix

# In[31]:


corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")


# In[32]:


df.head()


# In[33]:


# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
df.head()


# ## 4 Binning
# Binning/Converting Categorical Variable to Numerical 
# ### Label Encoding

# In[34]:



from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[35]:


df.head()


# ## Train-Test Split

# In[36]:


# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']


# In[37]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # 5. Modelling

# In[38]:


from sklearn.linear_model import LogisticRegression


# In[39]:


# classify function
from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    # cross validation - it is used for better validation of model
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)


# In[40]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[41]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# In[42]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[43]:


model = ExtraTreesClassifier()
classify(model, X, y)


# ## Confusion Matrix
# 
# A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

# In[44]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[45]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[46]:


sns.heatmap(cm, annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




