#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LIBRARIES 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
from datetime import datetime, timedelta


# In[2]:


from trino.dbapi import connect
from trino.auth import BasicAuthentication


# In[3]:


from collections import Counter
import pytz


# In[4]:


import pickle as p
import requests as req


# In[5]:


logreg_mod = p.load(open('lr_mod.p', 'rb'))


# In[6]:


conn = connect(host="tcp.smooth-bonefish.dataos.app",
      port="7432",
      auth=BasicAuthentication('balaji', 'dG9rZW5fc2Vjb25kbHlfZWFybHlfc2hhcnBfZmxlYS43YTU4NDNkOC02MzE4LTQ3MzQtYjZjOS05NTZlOWNlNGI1NWM='),
      http_scheme="https",
      http_headers={"cluster-name": "minervaa"}  # eg:minervaa
      )


# In[7]:


train_df = pd.read_csv("data/training_data.csv")


# In[8]:


train_df


# In[9]:


player_qr = "SELECT * FROM icebase.telco_ott.ott_player where date_diff('day', session_start, current_date) <= 1"


# In[10]:


player_df = pd.read_sql(player_qr, conn)


# In[13]:


def get_most_common_dev(devlist):
    c = Counter(devlist)
    return c.most_common(1)[0][0]


# In[14]:


agg_dict =  {
    'duration': 'mean',
    'device_id': len,
    'hardware': get_most_common_dev
}

player_agg = player_df.groupby('user_id').agg(agg_dict).reset_index()

rename_dict = {
    'duration': 'avg_dur',
    'device_id': 'num_devs',
    'hardware': 'freq_used_dev'
}

player_agg = player_agg.rename(columns = rename_dict)


# In[15]:


userids = list(player_df['user_id'].unique())


# In[16]:


userids = ["'"+u+"'" for u in userids]


# In[17]:


user_qr = """SELECT profile_id as user_id, dob, contact_type, gender, pack_id,
start_date, status, upgradable, validity_end, country, state, customer_id
FROM icebase.telco_ott.ott_userdataset
WHERE profile_id IN ({0})
""".format(",".join(userids))


# In[18]:


user_df = pd.read_sql(user_qr, conn)


# In[19]:


pred_df = user_df.merge(player_agg, on = 'user_id', how = 'left')


# In[20]:


pred_df


# In[21]:


cust_ids =  ["'"+u+"'" for u in user_df['customer_id']]


# In[22]:


billing_qr = """SELECT  customer_id, billing_type, customer_longevity,subscription_type
FROM icebase.telco_ott.ott_billing_01
WHERE customer_id IN ({0})""".format(",".join(cust_ids))


# In[23]:


billing_df = pd.read_sql(billing_qr, conn)


# In[24]:


pred_df = pred_df.merge(billing_df, on = 'customer_id', how = 'left')


# In[25]:


def get_age(dob):
    diff = datetime.now(tz = pytz.utc)-pd.Timestamp(dob).to_pydatetime()
    return diff.days/365


# In[26]:


pred_df["age"] = pred_df["dob"].apply(get_age)


# In[27]:


pred_df.drop(columns = ["dob"], inplace = True)


# In[28]:


def label_trans(lookup_dict, val):
    return lookup_dict[val]


# In[29]:


contact_lookup = {'Email': 0, 'Phone': 1, 'Telegram': 2, 'WhatsApp': 3}


# In[30]:


pred_df['contact_type'] = pred_df['contact_type'].apply(lambda x: label_trans(contact_lookup, x))


# In[31]:


gender_lookup = {'Agender': 0, 'Bigender': 1, 'Female': 2, 'Genderfluid': 3, 'Genderqueer': 4, 'Male': 5, 'Non-binary': 6, 'Polygender': 7}


# In[32]:


pred_df['gender'] = pred_df['gender'].apply(lambda x: label_trans(gender_lookup, x))


# In[33]:


pred_df['pack_id'] = pred_df['pack_id'].astype(int)


# In[34]:


status_lookup = {'ACTIVE': 0, 'INACTIVE': 1}


# In[35]:


pred_df['status'] = pred_df['status'].apply(lambda x: label_trans(status_lookup, x))


# In[36]:


state_lookup = {'Andhra Pradesh': 0, 'Arunachal Pradesh': 1, 'Assam': 2, 'Bihar': 3, 'Chhattisgarh': 4, 'Goa': 5, 'Gujarat': 6, 'Haryana': 7, 'Himachal Pradesh': 8, 'Jharkhand': 9, 'Karnataka': 10, 'Kerala': 11, 'Madhya Pradesh': 12, 'Maharashtra': 13, 'Manipur': 14, 'Meghalaya': 15, 'Mizoram': 16, 'Nagaland': 17, 'Odisha': 18, 'Punjab': 19, 'Rajasthan': 20, 'Sikkim': 21, 'Tamil Nadu': 22, 'Telangana': 23, 'Tripura': 24, 'Uttar Pradesh': 25, 'Uttarakhand': 26, 'West Bengal': 27}


# In[37]:


pred_df['state'] = pred_df['state'].apply(lambda x: label_trans(state_lookup, x))


# In[38]:


billing_type_lookup = {'NetBanking': 0, 'Paytm': 1}


# In[39]:


pred_df['billing_type'] = pred_df['billing_type'].apply(lambda x: label_trans(billing_type_lookup, x))


# In[40]:


upgradable_lookup = {'NO': 0, 'YES': 1}


# In[41]:


pred_df['upgradable'] = pred_df['upgradable'].apply(lambda x: label_trans(upgradable_lookup, x))


# In[42]:


freq_used_dev_lookup = {'Android Phone': 0, 'Android TV': 1, 'Apple TV': 2, 'FireTV': 3, 'PC/Laptop': 4, 'iPad': 5, 'iPhone': 6}
pred_df['freq_used_dev'] = pred_df['freq_used_dev'].apply(lambda x: label_trans(freq_used_dev_lookup, x))


# In[43]:


pred_df['subscription_type'] = pred_df['subscription_type'].str.replace(' Months', '')
pred_df['customer_longevity'] = pred_df['customer_longevity'].str.strip().str.replace(' Years', '')


# In[44]:


cols2train = ['Contact_type', 'Gender', 'Pack_ID',
       'Status', 'Upgradable', 'State', 'avg_dur', 'num_devs',
       'freq_used_dev', 'Subscription_Type', 'Billing_Type',
       'Customer_Longevity', 'age' 
]

cols2train = [c.lower() for c in cols2train]


# In[45]:


churn_pred = list(logreg_mod.predict(pred_df[cols2train]))


# In[46]:


neg_proba = [p[0] for p in list(logreg_mod.predict_proba(pred_df[cols2train]))]
pos_proba = [p[1] for p in list(logreg_mod.predict_proba(pred_df[cols2train]))]


# In[47]:


pred_df['churnpred'] = churn_pred
pred_df['pos_proba'] = pos_proba
pred_df['neg_proba'] = neg_proba
pred_df['timestamp'] = datetime.now()
pred_df.rename(columns = {'user_id':'customer_id'}, inplace = True)


# In[51]:


churn_dict = pred_df[['timestamp', 'customer_id', 'churnpred', 'pos_proba', 'neg_proba']].to_dict(orient = 'index')


# In[52]:


ret = req.post('https://smooth-bonefish.dataos.app/ottchurn/api/v1/churn_pred/', data = churn_dict)

