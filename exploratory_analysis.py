from collections.abc import Reversible
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wordcloud
import altair as alt
from collections import Counter

def plot_token_histogram(df): #o unico que acho que deu certo
    st.header("Número de Tokens por Comentário")
    token_counts = df["review_text_tokenized"].map(len) 
    st.bar_chart(token_counts.value_counts().sort_index())

def plot_token_distribution(df):
    df['token_count'] = df['review_text_tokenized'].apply(len)

    # Conta quantos comentários têm cada número de tokens
    token_summary = df['token_count'].value_counts().sort_index().reset_index()
    token_summary.columns = ['Nº de Tokens','Nº de Comentários']

    # Exibe como tabela no Streamlit
    st.subheader("Distribuição de Tokens por Comentário")
    st.dataframe(token_summary, use_container_width=True, hide_index=True)  
#-------------------------------------------- 

def plot_class_distribution(df):
    st.subheader("Distribuição de Comentários por Avaliação")

    # Conta quantos comentários existem por polaridade
    polarity_counts = df['rating'].value_counts().reset_index()
    polarity_counts.columns = ['Avaliação', 'Quantidade']

    # Converte os valores de polaridade em string para forçar o eixo X a ser categórico
    polarity_counts['Avaliação'] = polarity_counts['Avaliação'].astype(str)

    # Ordena os valores do eixo X (caso apareça fora de ordem)
    polarity_counts = polarity_counts.sort_values(by='Avaliação') 

    # Gráfico de barras com Altair
    chart = alt.Chart(polarity_counts).mark_bar().encode(
            x=alt.X('Avaliação:N', title='Avaliação (1 a 5)'),
            y=alt.Y('Quantidade', title='Quantidade'),
        color='Avaliação:N'
    )

    st.altair_chart(chart, use_container_width=True)   
#--------------------------------------------

def show_most_common_tokens(df):
    st.subheader("Palavras Mais Comuns")

    # Permitir ao usuário escolher quantos tokens quer visualizar
    n = st.slider("Quantidade de palavras mais comuns a exibir", min_value=5, max_value=150, value=10)
    
    df["review_text_tokenized"] = df["review_text_tokenized"].apply(lambda x: x.split() if isinstance(x, str) else x)

    all_tokens = [token for sublist in df["review_text_tokenized"] for token in sublist]
     
    # contar tokens (palavras)
    token_counts = Counter(all_tokens)

    # Obter os N mais comuns
    most_common_tokens = token_counts.most_common(n)

    # Transformar em DataFrame para exibir
    common_df = pd.DataFrame(most_common_tokens, columns=["Token", "Frequência"])

    # Exibir tabela sem índice
    st.dataframe(common_df, hide_index=True)

#--------------------------------------------

def plot_wordcloud(df):
    st.subheader("Word Cloud do Dataset")
    
#-------------------------------------------- 

def streamlit_show(df):
    st.title("Análise Exploratória do Dataset de Polaridade")

    with st.expander("Análises Básicas", expanded=True):
        plot_token_histogram(df)
        plot_token_distribution(df) 
        plot_class_distribution(df)
        show_most_common_tokens(df) 
         
            #plot_all_tokens(df) 

    #with st.expander("Análise de Texto"):
        #if st.checkbox("Palavras Mais Comuns", value=True):
            #plot_top_words(df)

    #with st.expander("Word Cloud"):
        #if st.checkbox("Exibir Word Cloud", value=False):
            #plot_wordcloud(df)
