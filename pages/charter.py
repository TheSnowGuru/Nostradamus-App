import pandas as pd #import streamlit as st 
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from tempfile import NamedTemporaryFile
import statistics
import plotly 
import plotly.express as px
from sklearn.linear_model import LinearRegression
import random, string
from pathlib import Path
import os
import sys #df = pd.read_csv(df)
from streamlit_folium import folium_static
import altair_viewer
 #fig = "I am here"
df = pd.read_csv(r"pages/dataset.csv") #df = pd.read_csv(r"writable_files\dataset.csv")
# Need to Assign the df2 to df because that is the default dataframe that GPT deals with 
#Can use a downloaded file to obtain the CSV  # Tries to import the uploaded file if possible, if it is possible it is overwritten
try:
    from a_robot_charter import df_export
    df = pd.read_csv(df_export)
finally:
    pass
    