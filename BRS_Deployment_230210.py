import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


st.title('Book Recommendation System')
st.sidebar.header('User Input Parameters')

df1 = pd.read_parquet('Cleaned_Data_Actual.parquet.gzip')
df2 = pd.read_parquet('Zero_Rating.parquet.gzip')
dfu = pd.read_parquet('User_USA.parquet.gzip')
dfo = pd.read_parquet('User_Other.parquet.gzip')


min_rat1 = 50
min_rat2 = 10

book_ratings1 = dfu.groupby("Book_Title")["Book_Rating"].count()
df3 = dfu[dfu["Book_Title"].isin(book_ratings1[book_ratings1 >= min_rat1].index)]

book_ratings2 = dfo.groupby("Book_Title")["Book_Rating"].count()
df4 = dfo[dfo["Book_Title"].isin(book_ratings2[book_ratings2 >= min_rat2].index)]


def user_input_features():
    UID = st.sidebar.number_input("Provide User_ID",min_value=1,max_value=278858)   
    data = {'User_ID':UID}
    features = pd.DataFrame(data, index=[0])
    return features

df_ip = user_input_features()
st.subheader("Specified User_ID")
st.write(df_ip)

UID=int(df_ip['User_ID'][0])

if UID in df3['User_ID'].values:
    df_user=df3[df3['User_ID']==UID]
    df_user.sort_values('Book_Rating',ascending=False,inplace=True)
    # Import from Parquet(Compressed Dataframe)
    user_sim_df = pd.read_parquet('USA_sim1.parquet.gzip')
    user_sim_df.columns=[int(col) for col in user_sim_df.columns]
    #Select the row from similarity matrix as per the given User_ID
    user_simi = user_sim_df.loc[UID, :]
    #Sort the above variable and get top_5 users similar to our defined User
    top_5 = user_simi.sort_values(ascending=False)[1:6]
    #Add User_ID's to list
    simi_users = top_5.index.tolist()
    #Create empty dataframe
    df_reco=pd.DataFrame(columns=df3.columns)
    #Create Dataframe with User_ID similar to given User_ID
    for i in simi_users:
        entry = df3.loc[df3['User_ID'] == i]
        df_reco=pd.concat([df_reco,entry])
    df_reco.drop_duplicates(subset='Book_Title',keep='first',inplace=True)
    df_reco=df_reco.sort_values(by='Book_Rating',ascending=False)
    list1=list(df_user['Book_Title'])
    list2=list(df_reco['Book_Title'])
    reco=[x for x in list2 if x not in list1]
    
elif UID in df4['User_ID'].values:
    df_user=df4[df4['User_ID']==UID]
    df_user.sort_values('Book_Rating',ascending=False,inplace=True)
    user_sim_df = pd.read_parquet('Other_sim1.parquet.gzip')
    user_sim_df.columns=[int(col) for col in user_sim_df.columns]
    user_simi = user_sim_df.loc[UID, :]
    top_5 = user_simi.sort_values(ascending=False)[1:11]
    simi_users = top_5.index.tolist()
    df_reco=pd.DataFrame(columns=df4.columns)
    for i in simi_users:
        entry = df4.loc[df4['User_ID'] == i]
        df_reco=pd.concat([df_reco,entry])
    df_reco.drop_duplicates(subset='Book_Title',keep='first',inplace=True)
    df_reco=df_reco.sort_values(by='Book_Rating',ascending=False)
    list1=list(df_user['Book_Title'])
    list2=list(df_reco['Book_Title'])
    reco=[x for x in list2 if x not in list1]
elif UID in df2['User_ID'].values:
    df_user=df2[df2['User_ID']==UID]
    df_user.sort_values('Book_Rating',ascending=False,inplace=True)
    country=set(df_user['Country'])
    country_df = df2[df2['Country'] == country]
    country_df = country_df.sort_values(by=['Book_Rating','Num_of_ratings','Avg_Rating'],ascending=[False,False,False])
    list2=country_df['Book_Title'][:300]
    df_user=df2[df2['User_ID']==UID]
    list1=list(df_user['Book_Title'])
    reco_a=[x for x in list2 if x not in list1]
    reco=reco_a[:11]
else:
    df_temp=df1.sort_values(by=['Book_Rating','Num_of_ratings','Avg_Rating'],ascending=[False,False,False])
    df_temp.drop_duplicates(subset='Book_Title',keep='first',inplace=True)
    reco1=df_temp['Book_Title'].tolist()
    reco=reco1[:11]
    st.write('User read 1 book or rated no books')




st.subheader("Book Recommendations for User_ID:")
st.write(reco[:10])



