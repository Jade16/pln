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
    #st.table(common_df.style.set_properties(**{'text-align':'left'}))
    st.dataframe(common_df, use_container_width=True, hide_index=True)  

#--------------------------------------------
def analyze_rating_length_correlation(df):
    st.subheader("Correlação entre Avaliação e Comprimento do Texto")
    df = df.copy()  # Isso evita modificar o DataFrame original 
    # Calcula comprimento do texto (versão segura para strings ou listas)
    df['text_length'] = df['review_text_tokenized'].apply(
        lambda x: len(x.split()) if isinstance(x, str) else len(x))
    
    # Filtro opcional para outliers (com checkbox)
    if st.checkbox("Remover outliers extremos (5% superiores)"):
        max_length = df['text_length'].quantile(0.95)
        df = df[df['text_length'] <= max_length]
   
    # 2. Cálculo da Correlação
    correlation = df[['rating', 'text_length']].corr().iloc[0, 1]
    # explicar o que é a correlação (pearson)
    
    # Exibe métrica com cor condicional
    metric_color = "red" if correlation < -0.1 else "green" if correlation > 0.1 else "gray"
    st.markdown(f"""
    <div style="border-left: 5px solid {metric_color}; padding-left: 10px;">
        <h4 style="margin-top: 0;">Correlação (Pearson)</h4>
        <h2 style="color: {metric_color}; margin-top: -10px;">{correlation:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Gráfico de Dispersão Aprimorado
    st.markdown("Relação Quantitativa")
    scatter = alt.Chart(df).mark_circle(
        size=60,
        opacity=0.6,
        stroke='#333',
        strokeWidth=1
    ).encode(
        x=alt.X('rating:O', 
                title='Avaliação (1-5)',
                axis=alt.Axis(labelAngle=0, grid=False),
                scale=alt.Scale(domain=[1, 2, 3, 4, 5])),
        y=alt.Y('text_length:Q', 
                title='Número de Palavras',
                scale=alt.Scale(zero=False)),
        color=alt.Color('rating:N', 
                       scale=alt.Scale(scheme='redyellowgreen'),
                       legend=None),
        tooltip=['rating', 'text_length']
    ).properties(
        width=700,
        height=400
    )
    
    # Adiciona linha de tendência
    trend_line = scatter.transform_regression(
        'rating', 'text_length'
    ).mark_line(color='red', size=2)
    
    st.altair_chart(scatter + trend_line, use_container_width=True)
    
    # 4. Boxplot Aprimorado
    st.markdown("Distribuição Detalhada por Avaliação")
    boxplot = alt.Chart(df).mark_boxplot(
        extent='min-max',
        size=30,
        color='rating:N'
    ).encode(
        x=alt.X('rating:O', 
                title='Avaliação',
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('text_length:Q', 
                title='Comprimento do Texto (palavras)',
                scale=alt.Scale(zero=False)),
        color=alt.Color('rating:N', 
                       scale=alt.Scale(scheme='redyellowgreen'),
                       legend=None)
    ).properties(
        width=700
    )
    
    st.altair_chart(boxplot, use_container_width=True)
    
    # 5. Análise Textual Complementar
    st.markdown("Insights Explicativos")
    st.write(f"""
        - **Correlação negativa ({correlation:.2f})** sugere que avaliações mais baixas (1-2 estrelas) 
        tendem a ser mais longas, possivelmente porque usuários explicam motivos para insatisfação.
        - Avaliações neutras (3 estrelas) geralmente são as mais curtas.
        - A dispersão mostra grande variação, indicando que outros fatores influenciam o comprimento.
        """)
#-------------------------------------------- 

def streamlit_show(df):
    st.title("Análise Exploratória do Dataset de Polaridade")

    with st.expander("Análises Básicas", expanded=True):
        plot_token_histogram(df)
        plot_token_distribution(df) 
        plot_class_distribution(df)
        show_most_common_tokens(df)
        analyze_rating_length_correlation(df)  
         
            #plot_all_tokens(df) 

    #with st.expander("Análise de Texto"):
        #if st.checkbox("Palavras Mais Comuns", value=True):
            #plot_top_words(df)

    #with st.expander("Word Cloud"):
        #if st.checkbox("Exibir Word Cloud", value=False):
            #plot_wordcloud(df)
