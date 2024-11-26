#!/usr/bin/env python
# coding: utf-8

# # 1. Data Exploration

# In[ ]:


import pandas as pd
df = pd.read_csv('heart_disease_uci.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# # 2. Handling Missing Data

# In[ ]:


df.replace('?', pd.NA, inplace=True)


# In[ ]:


df.notnull().sum()


# In[ ]:


for col in ['ca', 'thal']:
    df[col].fillna(df[col].mode()[0], inplace=True) # filling missing with mode of row


# In[ ]:


df.dropna(inplace=True)


# # 3. Feature Creation

# In[ ]:


bins = [0, 40, 60, 100]
labels = ['<40', '40-60', '>60']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)


# In[ ]:


def cholesterol_level(chol):
    if chol < 200:
        return 'Low'
    elif 200 <= chol <= 239:
        return 'Normal'
    else:
        return 'High'

df['CholesterolLevel'] = df['chol'].apply(cholesterol_level)


# In[ ]:


df['IsRisk'] = ((df['chol'] > 240) | (df['trestbps'] > 140) | (df['age'] > 60)).astype(int)


# # 4. Feature Transformation

# In[ ]:


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in ['sex', 'cp', 'thal', 'AgeGroup']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_cols = ['chol', 'trestbps', 'thalch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# # 5. Feature Interaction

# In[ ]:


df['BP_Chol_Interaction'] = df['trestbps'] * df['chol']


# In[ ]:


threshold = 100
df['ExerciseRisk'] = ((df['exang'] == 1) & (df['thalch'] < threshold)).astype(int)


# In[ ]:




