# === BIBLIOTECAS ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import io
import ast
import re
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud

# === CONFIGURAÇÃO DA PÁGINA ===
st.set_page_config(page_title="Análise Exploratória", layout="wide")
st.markdown("<h1 style='text-align: center; color: #264653;'>Análise Exploratória de Dados</h1>", unsafe_allow_html=True)
st.divider()

# === FUNÇÕES DE CACHE ===
@st.cache_data(show_spinner=False)
def load_base():
    df = pd.read_csv("data/b2w.csv")
    df = df.dropna(subset=["polarity"]).reset_index(drop=True)
    # Seleção de colunas
    if "review_text_processed" in df.columns:
        df["review_length"] = df["review_text_processed"].str.split().apply(len)
        df["text_len"] = df["review_text_processed"].str.len()
    else:
        df["review_length"] = np.nan
        df["text_len"] = np.nan
    
    if "review_text_tokenized" in df.columns:
        df["tokens"] = df["review_text_tokenized"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
        df["num_tokens"] = df["tokens"].apply(len)
    else:
        df["tokens"] = None
        df["num_tokens"] = np.nan
    return df

df = load_base()

# === TABS ===
abas = st.tabs([
    "📄  Base de Dados",
    "📊 Estatísticas Gerais",
    "💬 Frequência de Palavras",
    "☁️ Wordcloud",
    "⭐ Rating/Polaridade",
    "🔠 N-Gramas"
])

# === TAB 1: Base de Dados ===
with abas[0]:
    st.subheader("Visualização da Base")
    st.dataframe(df.head(500), use_container_width=True)  # Seleção de amostra
    st.divider()

    st.subheader("Informações da Tabela")
    info_df = pd.DataFrame({
        "Coluna": df.columns,
        "Tipo de Dado": df.dtypes.values,
        "Não Nulos": df.notnull().sum().values,
        "Nulos": df.isnull().sum().values,
        "% Nulos": (df.isnull().mean().values * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

# === TAB 2: Estatísticas Gerais ===
with abas[1]:
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe().round(2), use_container_width=True)
    st.divider()

    st.subheader("Polarity e Rating")
    st.dataframe(df[['polarity', 'rating']].describe().round(2), use_container_width=True)
    st.divider()

    st.subheader("Distribuição de Polaridade")
    pol_df = df['polarity'].value_counts().sort_index().reset_index()
    pol_df.columns = ['Polaridade', 'Frequência']
    fig = px.bar(pol_df, x='Polaridade', y='Frequência', text='Frequência', color_discrete_sequence=['#7C3AED'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Distribuição de Notas")
    rating_counts = df['rating'].value_counts().sort_index()
    rating_percent = (rating_counts / rating_counts.sum() * 100).round(1)
    rating_df = pd.DataFrame({
        'Rating': rating_counts.index,
        'Contagem': rating_counts.values,
        'Porcentagem (%)': rating_percent.astype(str) + '%'
    })
    fig = px.bar(rating_df, x='Rating', y='Contagem', text='Porcentagem (%)', color_discrete_sequence=['#FFA24D'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    if "review_length" in df.columns and df["review_length"].notnull().any():
        st.subheader("Distribuição de Tokens por Comentário")
        fig = px.histogram(df, x='review_length', nbins=20, labels={'review_length': 'Número de Tokens'}, color_discrete_sequence=['#2596FF'])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# === TAB 3: Frequência de Palavras ===
with abas[2]:
    st.subheader("Palavras mais Frequentes")
    sample_text = " ".join(df['review_text'].astype(str).sample(min(len(df), 5000), random_state=0)).lower()
    tokens = re.findall(r'\b\w+\b', sample_text)
    top_words = Counter(tokens).most_common(20)
    top_df = pd.DataFrame(top_words, columns=['Palavra', 'Frequência'])

    fig = px.bar(top_df, x='Palavra', y='Frequência', text='Frequência', color_discrete_sequence=['#5B6DCD'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Palavras mais Frequentes (com 3+ caracteres)")
    tokens3 = re.findall(r'\b\w{3,}\b', sample_text)
    top_df = pd.DataFrame(Counter(tokens3).most_common(20), columns=['Palavra', 'Frequência'])
    fig = px.bar(top_df, x='Palavra', y='Frequência', text='Frequência', color_discrete_sequence=['#FFB86B'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Palavras Menos Frequentes")
    least_common = sorted(Counter(tokens3).items(), key=lambda x: x[1])[:20]
    st.dataframe(pd.DataFrame(least_common, columns=["Palavra", "Frequência"]), use_container_width=True)

# === TAB 4: Nuvens de Palavras ===
with abas[3]:
    st.subheader("Nuvens de Palavras por Sentimento")

    def generate_wordcloud(text):
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        return fig

    # Seleção de amostra
    n_wc = 4000
    pos_reviews = df[df['rating'] >= 4]['review_text_processed'].dropna().astype(str)
    neg_reviews = df[df['rating'] <= 2]['review_text_processed'].dropna().astype(str)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Reviews Positivas")
        st.pyplot(generate_wordcloud(" ".join(pos_reviews.sample(min(n_wc, len(pos_reviews)), random_state=1))))
    with col2:
        st.markdown("##### Reviews Negativas")
        st.pyplot(generate_wordcloud(" ".join(neg_reviews.sample(min(n_wc, len(neg_reviews)), random_state=1))))

# === TAB 5: Análises por Rating/Polaridade ===
with abas[4]:
    st.subheader("Distribuição de Tokens por Rating")
    if "tokens" in df.columns and "num_tokens" in df.columns:
        fig = px.box(df, x='rating', y='num_tokens', color='rating', points="all", color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    st.subheader("Palavras Mais Frequentes por Rating")
    col1, col2 = st.columns(2)
    for idx, rating in enumerate(sorted(df['rating'].dropna().unique())):
        if "tokens" in df.columns:
            top = Counter(word for tokens in df[df['rating'] == rating]['tokens'] for word in tokens).most_common(15)
            top_df = pd.DataFrame(top, columns=['Palavra', 'Frequência'])
            fig = px.bar(top_df, x='Palavra', y='Frequência', text='Frequência', title=f'Rating {rating}', color_discrete_sequence=['#2A9D8F'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', title_x=0.5)
            if idx % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Palavras por Polaridade")
    col1, col2 = st.columns(2)
    for idx, pol in enumerate(sorted(df['polarity'].dropna().unique())):
        if "tokens" in df.columns:
            top = Counter(word for tokens in df[df['polarity'] == pol]['tokens'] for word in tokens).most_common(15)
            top_df = pd.DataFrame(top, columns=['Palavra', 'Frequência'])
            fig = px.bar(top_df, x='Palavra', y='Frequência', text='Frequência', title=f'Polaridade {pol}', color_discrete_sequence=['#7C3AED'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', title_x=0.5)
            if idx % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Distribuição de Sentimentos")
    df['sentiment'] = pd.cut(df['polarity'], [-1.01, 0, 1.01], labels=['Negativo', 'Positivo'])
    sent_counts = df['sentiment'].value_counts().reset_index()
    sent_counts.columns = ['Sentimento', 'Contagem']
    fig = px.pie(sent_counts, names='Sentimento', values='Contagem', color='Sentimento', hole=0.4, color_discrete_map={'Negativo': '#FFB86B', 'Positivo': '#28C7A7'})
    fig.update_layout(title='Proporção de Sentimentos', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Comprimento do Texto vs. Sentimento")
    if "text_len" in df.columns:
        fig = px.scatter(df, x='text_len', y='rating', color='polarity', color_continuous_scale='viridis', opacity=0.6)
        fig.update_layout(title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# === TAB 6: N-Gramas ===
with abas[5]:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    def has_letters_and_length(token):
        return bool(re.search(r'[a-zA-Z]', token)) and len(token) > 2

    def get_top_ngrams(text_series, n=10, ngram_n=2):
        words = [
            word for text in text_series
            for word in word_tokenize(str(text).lower())
            if has_letters_and_length(word)
        ]
        if len(words) < ngram_n:
            return []
        ngram_list = list(ngrams(words, ngram_n))
        return Counter(ngram_list).most_common(n)

    def plot_ngrams(sentiment_label, ngram_n=2, top_n=10):
        subset = df[df['sentiment'] == sentiment_label]['review_text_processed']
        # Seleção de amostra
        if len(subset) > 2000:
            subset = subset.sample(2000, random_state=1)
        top_ngrams = get_top_ngrams(subset, n=top_n, ngram_n=ngram_n)
        if not top_ngrams:
            return None
        labels = [" ".join(gram) for gram, _ in top_ngrams]
        counts = [count for _, count in top_ngrams]
        fig = px.bar(
            x=counts, y=labels, orientation='h', text=counts, color_discrete_sequence=['#00426A']
        )
        fig.update_layout(
            title=f'Top {ngram_n}-Gramas - {sentiment_label}',
            xaxis_title='Contagem',
            yaxis_title=f'{ngram_n}-Gramas',
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    st.subheader("Top N-Gramas por Sentimento")

    ngram_options = {'1-grama': 1, '2-grama': 2, '3-grama': 3, '4-grama': 4}
    ngram_choice_label = st.selectbox("Selecione o tipo de N-Grama", list(ngram_options.keys()))
    ngram_n = ngram_options[ngram_choice_label]

    col1, col2 = st.columns(2)
    with col1:
        fig_pos = plot_ngrams('Positivo', ngram_n=ngram_n)
        if fig_pos:
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info(f"Nenhum {ngram_choice_label} encontrado para avaliações positivas.")
    with col2:
        fig_neg = plot_ngrams('Negativo', ngram_n=ngram_n)
        if fig_neg:
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info(f"Nenhum {ngram_choice_label} encontrado para avaliações negativas.")
