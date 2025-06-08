import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px
import spacy_streamlit

# === Configurações da Página ===
st.set_page_config(page_title="PLN Clássica", layout="wide", page_icon=":books:")
st.markdown("<h1 style='text-align: center; color: #264653;'>PLN Clássica </h1>", unsafe_allow_html=True)
st.divider()

# === Leitura dos arquivos ===
@st.cache_data
def load_dataframe():
    return pd.read_csv("models_results/part_1/dataset_all.csv")

df = load_dataframe()

@st.cache_data
def load_pos_tags():
    return pd.read_csv("models_results/part_1/pos_tags.csv")

@st.cache_data
def load_dep_parse():
    return pd.read_csv("models_results/part_1/dep_parse.csv")

@st.cache_data
def load_sentilex():
    return pd.read_csv("models_results/part_1/sentilex.csv")

sentilex_df = load_sentilex()
sentilex = dict(zip(sentilex_df["Palavra"].str.lower(), sentilex_df["Polaridade"]))

# === Modelo SpaCy ===
@st.cache_resource
def load_spacy_model():
    return spacy.load("pt_core_news_sm")

nlp = load_spacy_model()

# === Análise Semântica ===
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

# === TABS ===
tabs = st.tabs([
    "📄 Base de Dados",
    "🔤 Análise Morfológica",
    "🌐 Análise Sintática",
    "💬 Sentimento",
    "✍️ Análise Personalizada"
])

# === Base de Dados ===
with tabs[0]:
    st.subheader("Base de Dados")
    st.dataframe(df[["review_text", "review_text_clean", "review_text_tokenized", "hybrid_sentiment"]].sample(10), use_container_width=True)
    st.divider()

# === Pos-Tags ===
with tabs[1]:
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

# === Dependência Sintática ===
with tabs[2]:
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

# === Distribuição de Sentimentos  ===
with tabs[3]:
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

# === Análise Personalizada ===
with tabs[4]:
    st.subheader("Análise Personalizada de Sentimento")
    user_input = st.text_area("Digite uma frase para análise:", placeholder="Exemplo: O serviço foi excelente!")

    if user_input.strip():
        doc = nlp(user_input)
        with st.expander("Visualize a Árvore de Dependência"):
            spacy_streamlit.visualize_parser(doc, title="Parser", key="parser_input")

        score = semantic_sentiment(user_input)
        st.divider()

        if score > 0:
            sentiment = "Positivo"
            stars = "👍"
            color = "#43aa8b"  
        else:
            sentiment = "Negativo"
            stars = "👎"
            color = "#e63946"  

        st.markdown(
            f"<span style='font-size:1.1em;color:{color};font-weight:600'>{sentiment} {stars}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<span style='font-size:0.98em;color: #444;'>Score: {score:.2f}</span>",
            unsafe_allow_html=True
        )
