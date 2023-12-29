import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.cm as cm
   
import polars as pol 


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.title("PHIDO demo for Jan 19")
st.write(
    """"""
)

def filter_dataframe(df, name_of_chkbox, select_cols = None ) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox( name_of_chkbox )

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    if select_cols is None:
        select_cols = df.columns
    
        
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", select_cols )
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
    return df

@st.cache_data
def load_data(): # private_repository requires URL of resources in relative form
    geo_df = pd.read_csv( './data/GeoReferenceTableBC_city2lha.csv' )
    df = pol.read_csv("./data/sample_2019_repeated_10yr_weekly.csv")
    return df, geo_df

input_pol, geo_df = load_data()

    
from datetime import datetime
# convert from string to datetime field
try:
    input_pol = input_pol.with_columns( pol.col("date").str.to_datetime("%m-%d-%Y").alias('date_dt') )    
except:    
    input_pol = input_pol.with_columns( pol.col("date").str.to_datetime("%Y-%m-%d").alias('date_dt') )

# we can now do math on last field 
enddate   = input_pol[:,-1].max()  

# Getting the min and max date 
endDate = pd.to_datetime( input_pol["date"]).max()
try:
    startDate = endDate - pd.DateOffset( months = 12 )
    print('There exists sufficient data to examine last 12 months')
except:
    startDate = endDate - pd.DateOffset( months = 1 )
    print('There exists sufficient data to examine last 30 days')

    
# ================== setup the maps ==================
hsda_codes = geo_df.copy()
hsda_codes.drop_duplicates('HSDA_NAME',inplace=True )
hsda_codes.set_index( 'HSDA_NAME', inplace=True)

lha_codes = geo_df.copy()
lha_codes.drop_duplicates('LHA_NAME',inplace=True )
lha_codes.set_index( 'LHA_NAME', inplace=True)


# ================== setup the layout ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([ 'Table', 'Time-series', 'LHA map', 'HSDA map', 'Counts by disease' ])
  

with tab1:
    # plot subset as time series
    st.header("Time-series")

    # ======================================== Ctrl for Element 1 ========================================
    col1, col2 = st.columns((2))
    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    sub_pol = input_pol.filter( pol.col("date_dt") > date1 ).filter( pol.col("date_dt") < date2 )   

    st.scatter_chart( data=sub_pol.to_pandas(), 
                     x='date', 
                     y=['observedCounts', 'fittedCounts' ],
                     color=['#FF0000', '#0000FF'],
                     size = 2 
                     )     
     

with tab2:    
    #col1, col2 = st.columns((2))    
    st.header("Table")    
    sub_df = filter_dataframe( input_pol.to_pandas(), 'Add filters' ) # filter entire df
    st.dataframe( sub_df ) # display subset


with tab3:       
    st.header("LHA")
    st.map( 
        geo_df,     
        latitude  = 'LATITUDE',
        longitude = 'LONGITUDE',
        size=20, 
        color = '#daa'
    )          
    st.dataframe( geo_df )


with tab5:       
    diseases = np.unique( sub_df['surveillance_condition'] )
    st.text(diseases)
   
    columns = [ 'surveillance_reported_hsda_abbr', 'date' ]

    for d in diseases: 
        D = sub_pol.filter( pol.col('surveillance_condition') == d )
        val_df = D.to_pandas().loc[:, [ 'surveillance_reported_hsda_abbr', 'observedCounts' ] ].groupby('surveillance_reported_hsda_abbr', group_keys = False ).sum()    
        st.text( d )
        st.dataframe( val_df )     
        

with tab4:
    S = ['status','surveillance_condition', 'surveillance_reported_hsda_abbr', 'observedCounts']
    sub_df = filter_dataframe( input_pol.select(S).to_pandas(), 'Add select filters:',  ) # filter entire df
    st.dataframe( sub_df )

    diseases = np.unique( sub_df['surveillance_condition'] )       
    st.text(diseases)

    '''
    for d in diseases: 
        D = sub_pol.filter( pol.col('surveillance_condition') == d )
        val_df = D.to_pandas().loc[:, [ 'surveillance_reported_hsda_abbr', 'observedCounts' ] ].groupby('surveillance_reported_hsda_abbr', group_keys = False ).sum()    
        
        val_df['lat']  = 0
        val_df['long'] = 0
        
        try:
            val_df['lat']  = hsda_codes.loc[ val_df[k] ].LATITUDE         
            val_df['long'] = hsda_codes.loc[ val_df[k] ].LONGITUDE               
        except:
            pass 
            
        #st.text( caption )
        st.dataframe( val_df ) 

        mx = val_df['observedCounts'].max() + 1e-10 
        norm = mpl.colors.Normalize(vmin=0, vmax=mx, clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)

        val_df['hex_color'] = val_df['observedCounts'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
        
        st.header("Total counts per location by HSDA classification")
        st.map( 
            val_df,     
            latitude  = 'lat',
            longitude = 'long',
            size=100,
            color='hex_color' )              
    ''';