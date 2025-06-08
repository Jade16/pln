import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from pathlib import Path
import os
import spacy
import pandas as pd
import json


# --- Configurações da Página ---
st.set_page_config(page_title="PLN Moderna", layout="centered", page_icon=":books:")
st.markdown("<h1 style='text-align: center; color: #264653;'> PLN Moderna</h1>", unsafe_allow_html=True)
st.divider()


# --- Funções ---
@st.cache_resource
def load_model(checkpoint_path):
    path = str(checkpoint_path)
    if not os.path.isdir(path):
        st.error(f"Diretório do modelo não encontrado em '{path}'.")
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    if predicted_class_id == 0:
        return "NEGATIVA"
    else:
        return "POSITIVA"

model_path = 'models_results/part_2/final_model_weights'

with st.spinner("Carregando o modelo treinado..."):
    tokenizer, model = load_model(model_path)

tabs = st.tabs(["🎮 Jogo", "✍️ Análise Personalizada"])

# === ABA: Jogo ===
with tabs[0]:
    st.subheader("Jogo de Análise de Sentimentos")
    with open("data/frases.json", "r", encoding="utf-8") as f:
        EXAMPLE_PHRASES = json.load(f)

    if tokenizer and model:
        if 'game' not in st.session_state:
            st.session_state.game = {
                'phrases': random.sample(EXAMPLE_PHRASES, 5),
                'current_index': 0,
                'results': []
            }

        game = st.session_state.game
        idx = game['current_index']

        if idx >= 5:
            st.success("🎉 Fim de Jogo!")
            score = sum(res['your_answer'] == res['model_answer'] for res in game['results'])
            st.header(f"Sua pontuação final: {score}/5")
            st.markdown({
                5: "### Parabéns, você é expert em sentimentos!",
                3: "### Muito bem! Você tem boa intuição."
            }.get(score, "### Continue praticando para melhorar!"))

            # ===== Balões =====
            if score == 5 and not game.get('victory_celebrated', False):
                st.balloons()
                game['victory_celebrated'] = True 
            
            st.divider()
            st.subheader("Confira seu desempenho:")
            for i, res in enumerate(game['results']):
                phrase, yours, model_ans = res['phrase'], res['your_answer'], res['model_answer']
                icon = "✅" if yours == model_ans else "❌"
                color = "#43aa8b" if yours == model_ans else "#e63946"
                st.markdown(
                    f"<div style='padding:8px 0'><b>Frase {i+1}:</b> <i>\"{phrase}\"</i><br>"
                    f"{icon} Sua resposta: <b style='color:{color};'>{yours}</b> | "
                    f"Modelo: <b style='color:#264653'>{model_ans}</b></div>",
                    unsafe_allow_html=True
                )
            st.divider()
            if st.button("Jogar Novamente", type="primary"):
                del st.session_state.game
                st.rerun()
        else:
            current_phrase = game['phrases'][idx]
            st.progress(idx/5, text=f"Frase {idx+1}/5")
            st.info(f'"{current_phrase}"', icon="💬")
            col1, col2 = st.columns(2)
            def process_answer(choice):
                game['results'].append({
                    'phrase': current_phrase,
                    'your_answer': choice,
                    'model_answer': predict_sentiment(current_phrase, tokenizer, model)
                })
                game['current_index'] += 1
                st.rerun()
            with col1:
                if st.button("Positiva 👍", use_container_width=True):
                    process_answer("POSITIVA")
            with col2:
                if st.button("Negativa 👎", use_container_width=True):
                    process_answer("NEGATIVA")
    else:
        st.warning("O modelo não pôde ser carregado. Verifique o diretório.")

# === ABA: ANÁLISE PERSONALIZADA ===
@st.cache_resource
def load_sentilex():
    sentilex_df = pd.read_csv("models_results/part_1/sentilex.csv")
    return dict(zip(sentilex_df["Palavra"].str.lower(), sentilex_df["Polaridade"]))
sentilex = load_sentilex()

@st.cache_resource
def load_spacy_model():
    return spacy.load("pt_core_news_sm")
nlp = load_spacy_model()

def semantic_sentiment(text):
    doc = nlp(text)
    intensifiers = [t.text.lower() for t in doc if t.pos_ == "ADV" and t.dep_ in ("advmod", "intj")]
    negators = [t.text.lower() for t in doc if t.dep_ == 'neg']
    score, negation, intensity = 0, False, 1.0
    for token in doc:
        word = token.text.lower()
        if word in negators:
            negation = not negation
            continue
        if word in intensifiers:
            intensity += 1
            continue
        pol = sentilex.get(word, 0)
        if negation: pol *= -1
        pol *= intensity
        if token.dep_ in ("amod", "advmod"):
            pol *= 1.5
        elif token.dep_ == "nsubj":
            pol *= 1.2
        score += pol
        intensity = 1.0
        if token.dep_ in ("punct", "cc"): negation = False
    return score / max(1, len(doc))

with tabs[1]:
    st.subheader("Análise Personalizada de Sentimento")
    user_input = st.text_area("Digite uma frase para análise:", placeholder="Exemplo: O serviço foi excelente!")

    if user_input.strip():
        score = semantic_sentiment(user_input)
        sent_classic = "Positivo" if score > 0 else "Negativo"
        classic_color = "#43aa8b" if score > 0 else "#e63946"
        classic_icon = "👍" if score > 0 else "👎"

        if tokenizer and model:
            bert_sentiment = predict_sentiment(user_input, tokenizer, model)
            bert_color = "#43aa8b" if bert_sentiment == "POSITIVA" else "#e63946"
            bert_icon = "👍" if bert_sentiment == "POSITIVA" else "👎"
        else:
            bert_sentiment = "Modelo não carregado"
            bert_color, bert_icon = "#bcbcbc", "❓"

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<b>Método Clássico</b>", unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-size:1.15em;color:{classic_color};font-weight:600'>{sent_classic} {classic_icon}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span style='font-size:1em;color:#6b6b6b;'>Score:<br><b>{score:.2f}</b></span>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown("<b>Modelo BERTimbau</b>", unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-size:1.15em;color:{bert_color};font-weight:600'>{bert_sentiment} {bert_icon}</span>",
                unsafe_allow_html=True
            )
