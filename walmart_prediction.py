# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:29:57 2024

@author: Dell
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
def main():
    html_temp="""
        <div style="background-color:lightblue;padding:16px">
        <h2 style="color:black;text-align:center;"> Walmart unemployment prediction
        </div>"""
    model=xgb.XGBRegressor()
    model.load_model('xgb_model.json')
    
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('')
    st.write('')
    
    
    st.markdown("##### Are you willing to know the unemployment rate of wallmart!? \n##### So let's try to find out")
    p1= st.number_input("what is temperature?(In degree celcius)",format="%.2f")
    p2=st.number_input("What is weekly sales??",format="%.2f")
    p3=st.number_input("What is CPI",format="%.2f")
    p4=st.number_input("What is fuel price",format="%.2f")
    
    data_new=pd.DataFrame({'Temperature':p1,'Weekly_Sales':p2,'CPI':p3,'Fuel_Price':p4},index=[0])
    if st.button('Predict'):
        pred=model.predict(data_new)
        st.balloons()
        st.success("Unemployment rate is : {:.2f} ".format(pred[0]))
        
   
   
    
    
    
    
    
    
    
if __name__=='__main__':
    main()