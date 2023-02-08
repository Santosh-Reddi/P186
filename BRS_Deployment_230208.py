import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


st.title('Book Recommendation System')
st.sidebar.header('User Input Parameters')

df1=pd.read_csv("Cleaned_Data_Actual.csv",
               dtype={"User-ID": int, "Book-Title": "string",'Book-Rating':int,'Age':int,
                     'Country':'string','Num of ratings':int,'Avg_Rating':float,'Books_Read':int})
dfu=pd.read_csv("User_USA.csv",
               dtype={"User-ID": int, "Book-Title": "string",'Book-Rating':int,'Age':int,
                     'Country':'string','Num of ratings':int,'Avg_Rating':float,'Books_Read':int})
dfo=pd.read_csv("User_Other.csv",
               dtype={"User-ID": int, "Book-Title": "string",'Book-Rating':int,'Age':int,
                     'Country':'string','Num of ratings':int,'Avg_Rating':float,'Books_Read':int})
df1.drop_duplicates(keep='first', inplace=True)
df2=df1[df1['Book-Rating']!=0]
df2=df2.drop_duplicates(keep='first')
dfu=dfu.drop_duplicates(keep='first')
dfo=dfo.drop_duplicates(keep='first')

min_rat1 = 50
min_rat2 = 10

book_ratings1 = dfu.groupby("Book-Title")["Book-Rating"].count()
df3 = dfu[dfu["Book-Title"].isin(book_ratings1[book_ratings1 >= min_rat1].index)]

book_ratings2 = dfo.groupby("Book-Title")["Book-Rating"].count()
df4 = dfo[dfo["Book-Title"].isin(book_ratings2[book_ratings2 >= min_rat2].index)]


def user_input_features():
    UID = st.sidebar.number_input("Provide User-ID",min_value=1,max_value=278858)   
    data = {'User-ID':UID}
    features = pd.DataFrame(data, index=[0])
    return features

df_ip = user_input_features()
st.subheader("Specified User-ID")
st.write(df_ip)

UID=int(df_ip['User-ID'][0])

if UID in df3['User-ID'].values:
    df_user=df3[df3['User-ID']==UID]
    df_user.sort_values('Book-Rating',ascending=False,inplace=True)
    # Import from Parquet(Compressed Dataframe)
    user_sim_df = pd.read_parquet('USA_sim1.parquet.gzip')
    user_sim_df.columns=[int(col) for col in user_sim_df.columns]
    #Select the row from similarity matrix as per the given User-ID
    user_simi = user_sim_df.loc[UID, :]
    #Sort the above variable and get top-5 users similar to our defined User
    top_5 = user_simi.sort_values(ascending=False)[1:6]
    #Add User-ID's to list
    simi_users = top_5.index.tolist()
    #Create empty dataframe
    df_reco=pd.DataFrame(columns=df3.columns)
    #Create Dataframe with User-ID similar to given User-ID
    for i in simi_users:
        entry = df3.loc[df3['User-ID'] == i]
        df_reco=pd.concat([df_reco,entry])
    df_reco.drop_duplicates(subset='Book-Title',keep='first',inplace=True)
    df_reco=df_reco.sort_values(by='Book-Rating',ascending=False)
    list1=list(df_user['Book-Title'])
    list2=list(df_reco['Book-Title'])
    reco=[x for x in list2 if x not in list1]
    
elif UID in df4['User-ID'].values:
    df_user=df4[df4['User-ID']==UID]
    df_user.sort_values('Book-Rating',ascending=False,inplace=True)
    user_sim_df = pd.read_parquet('Other_sim1.parquet.gzip')
    user_sim_df.columns=[int(col) for col in user_sim_df.columns]
    user_simi = user_sim_df.loc[UID, :]
    top_5 = user_simi.sort_values(ascending=False)[1:11]
    simi_users = top_5.index.tolist()
    df_reco=pd.DataFrame(columns=df4.columns)
    for i in simi_users:
        entry = df4.loc[df4['User-ID'] == i]
        df_reco=pd.concat([df_reco,entry])
    df_reco.drop_duplicates(subset='Book-Title',keep='first',inplace=True)
    df_reco=df_reco.sort_values(by='Book-Rating',ascending=False)
    list1=list(df_user['Book-Title'])
    list2=list(df_reco['Book-Title'])
    reco=[x for x in list2 if x not in list1]
elif UID in df2['User-ID'].values:
    df_user=df2[df2['User-ID']==UID]
    df_user.sort_values('Book-Rating',ascending=False,inplace=True)
    country=set(df_user['Country'])
    country_df = df2[df2['Country'] == country]
    country_df = country_df.sort_values(by=['Book-Rating','Num of ratings','Avg_Rating'],ascending=[False,False,False])
    list2=country_df['Book-Title'][:300]
    df_user=df2[df2['User-ID']==UID]
    list1=list(df_user['Book-Title'])
    reco_a=[x for x in list2 if x not in list1]
    reco=reco_a[:11]
else:
    df_temp=df1.sort_values(by=['Book-Rating','Num of ratings','Avg_Rating'],ascending=[False,False,False])
    df_temp.drop_duplicates(subset='Book-Title',keep='first',inplace=True)
    reco1=df_temp['Book-Title'].tolist()
    reco=reco1[:11]
    st.write('User read 1 book or rated no books')




st.subheader("Book Recommendations for User-ID:")
st.write(reco[:10])



