# === BIBLIOTECAS ===
import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px
import spacy_streamlit

# === CONFIGURAÇÃO DA PÁGINA ===
st.set_page_config(page_title="PLN Clássica", layout="wide")
st.markdown("<h1 style='text-align: center; color: #264653;'>PLN Clássica </h1>", unsafe_allow_html=True)
st.divider()

# === LEITURA DOS ARQUIVOS PRÉ-PROCESSADOS ===
@st.cache_data
def load_data():
    df = pd.read_csv("weights/dataset_all.csv")
    return df

df = load_data()

@st.cache_data
def load_pos_tags():
    return pd.read_csv("weights/pos_tags.csv")

@st.cache_data
def load_dep_parse():
    return pd.read_csv("weights/dep_parse.csv")

@st.cache_data
def load_sentilex():
    return pd.read_csv("weights/sentilex.csv")

sentilex_df = load_sentilex()
sentilex = dict(zip(sentilex_df["Palavra"].str.lower(), sentilex_df["Polaridade"]))

# === MODELO SPACY ===
@st.cache_resource
def load_spacy_model():
    return spacy.load("pt_core_news_sm")

nlp = load_spacy_model()

# === FUNÇÃO DE ANÁLISE SEMÂNTICA ===
def semantic_sentiment(text):
    doc = nlp(text)
    intensifiers = [token.text.lower() for token in doc if token.pos_ == "ADV" and token.dep_ in ("advmod", "intj")]
    negators = [token.text.lower() for token in doc if token.dep_ == 'neg']
    total_score = 0
    negation = False
    current_intensity = 1.0
    for token in doc:
        if token.text.lower() in negators:
            negation = not negation
            continue
        if token.text.lower() in intensifiers:
            current_intensity += 1
            continue
        word_polarity = sentilex.get(token.text.lower(), 0)
        if negation:
            word_polarity *= -1
        word_polarity *= current_intensity
        if token.dep_ in ("amod", "advmod"):
            word_polarity *= 1.5
        elif token.dep_ == "nsubj":
            word_polarity *= 1.2
        total_score += word_polarity
        current_intensity = 1.0
        negation = False if token.dep_ in ("punct", "cc") else negation
    return total_score / len(doc) if len(doc) > 0 else 0

# === ABAS ===
abas = st.tabs([
    "📄 Base de Dados",
    "🔤 Análise Morfológica",
    "🌐 Análise Sintática",
    "💬 Sentimento",
    "✍️ Análise Personalizada"
])

# === ABA: BASE DE DADOS ===
with abas[0]:
    st.subheader("Base de Dados")
    st.dataframe(df[["review_text", "review_text_clean", "review_text_tokenized", "hybrid_sentiment"]].sample(10), use_container_width=True)
    st.divider()

# === ABA: POS-TAGS ===
with abas[1]:
    st.subheader("Frequência de POS-Tags")
    pos_df = load_pos_tags()
    fig = px.bar(pos_df, x="POS", y="Freq", text="Percent", color_discrete_sequence=["#2596FF"])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Exemplos")
    for i, row in enumerate(df.sample(3, random_state=10).itertuples(), 1):
        st.write(f"**Review {i}:** {row.review_text}")
        doc = nlp(row.review_text)
        data = [(token.text, token.pos_) for token in doc]
        st.dataframe(pd.DataFrame(data, columns=["Token", "Classe Gramatical"]), use_container_width=True)
        st.markdown("---")

# === ABA: DEPENDÊNCIAS ===
with abas[2]:
    st.subheader("Frequência de Dependências Sintáticas")
    dep_df = load_dep_parse()
    fig = px.bar(dep_df.head(20), x="Dep", y="Freq", text="Percent", color_discrete_sequence=["#2596FF"])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Exemplos")
    for i, row in enumerate(df.sample(3, random_state=2).itertuples(), 1):
        st.write(f"**Review {i}:** {row.review_text}")
        doc = nlp(row.review_text)
        with st.expander("Árvore de Dependência"):
            spacy_streamlit.visualize_parser(doc, title="Parser", key=f"dep_{i}")
        data = [(token.text, token.dep_, token.head.text) for token in doc]
        st.dataframe(pd.DataFrame(data, columns=["Token", "Dependência", "Palavra Raiz"]), use_container_width=True)
        st.markdown("---")

# === ABA: SENTIMENTO ===
with abas[3]:
    st.subheader("Distribuição de Sentimento")
    sent_counts = df["hybrid_sentiment"].value_counts().rename_axis("Sentimento").reset_index(name="Contagem")
    fig = px.bar(sent_counts, x="Sentimento", y="Contagem", text="Contagem", color_discrete_sequence=["#2596FF"])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Exemplos Classificados")
    for _, row in df.sample(3, random_state=42).iterrows():
        st.write(f"**Review:** {row['review_text']}")
        st.write(f"**Sentimento:** {row['hybrid_sentiment']} | **Score:** {row['score']:.2f}")
        st.markdown("---")

# === ABA: ANÁLISE PERSONALIZADA ===
with abas[4]:
    st.subheader("Análise de Frase do Usuário")
    user_input = st.text_area("Digite uma frase:")
    if user_input:
        doc = nlp(user_input)
        with st.expander("Árvore de Dependência da Frase"):
            spacy_streamlit.visualize_parser(doc, title="Parser", key="parser_input")
        score = semantic_sentiment(user_input)
        st.write(f"**Score:** {score:.2f}")
        if score > 0.1:
            sentiment = "Positivo"; stars = "⭐⭐⭐⭐⭐"
        elif score < -0.1:
            sentiment = "Negativo"; stars = "⭐"
        else:
            sentiment = "Neutro"; stars = "⭐⭐⭐"
        st.markdown(f"**Classificação:** {sentiment} {stars}")

