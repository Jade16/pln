#!/usr/bin/env python3
import io

import streamlit as st
import wandb
import pandas as pd
#from datasets import Dataset
#from sklearn.model_selection import train_test_split

import exploratory_analysis
import extracting_dataset  

#if not dataset:
dataset = extracting_dataset.extracting_dataset() #rodar esse apenas uma vez caso o arquivo nao tenha sido extraido 

st.set_page_config(layout="wide")
st.title("Polaridade de Comentários")

st.header("Exploração dos Dados")
exploratory_analysis.streamlit_show(dataset)
