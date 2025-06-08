import streamlit as st

st.set_page_config(
    page_title="PLN",
    page_icon=":books:",
    layout="wide"
)

st.markdown("""
<style>
html, body, .main, .block-container {
    color-scheme: light dark;
    background: var(--background-color, #f7f9fa) !important;
}
.pln-mainbox {
    margin: 8vh auto 0 3vw;
    background: var(--background-color, #fff);
    border-radius: 20px;
    max-width: 480px;
    padding: 2.5em 2.2em 2em 2.2em;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: none;
}
.pln-title {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 2.15em;
    font-weight: 600;
    color: var(--text-color, #23343d);
    letter-spacing: 0.2px;
    margin-bottom: 0.18em;
    text-align: center;
}
.pln-inst, .pln-auth {
    color: var(--secondary-text-color, #477187);
    font-size: 1.04em;
    font-weight: 400;
    text-align: center;
    margin-bottom: 0.12em;
}
.pln-auth {
    color: var(--tertiary-text-color, #7e97a6);
    font-size: 0.97em;
    margin-bottom: 1.12em;
}
.divider-slim {
    border: none;
    border-top: 1.1px solid var(--divider-color, #d3dbe3);
    margin: 0.8em 0 1.15em 0;
    width: 78px;
    align-self: center;
}
.link-btns {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.8em;
    margin-bottom: 1.05em;
    width: 100%;
}
.btn-outline {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 1.04em;
    font-weight: 500;
    color: var(--primary-button-color, #20505c);
    background: transparent;
    border: 1.5px solid var(--primary-button-border, #29798e);
    border-radius: 1.5em;
    text-decoration: none;
    transition: all 0.14s cubic-bezier(.34,1.56,.64,1);
    outline: none;
    cursor: pointer;
    text-align: center;
    width: 97%;
    min-width: 130px;
    max-width: 320px;
    padding: 0.85em 1.1em;
    box-sizing: border-box;
    user-select: none;
    display: block;
    box-shadow: none;
}
.btn-outline:hover, .btn-outline:active {
    border-color: var(--primary-button-hover, #49dbbb);
    color: var(--primary-button-hover-color, #227774);
    background: var(--primary-button-hover-bg, #f0fdfa);
    text-decoration: none;
    filter: brightness(0.98);
}
.pln-footer {
    color: var(--footer-color, #b1bbc2);
    font-size: 0.91em;
    text-align: center;
    margin-top: 2em;
    letter-spacing: 0.11px;
    font-weight: 300;
}

@media (prefers-color-scheme: dark) {
    html, body, .main, .block-container {
        --background-color: #111418;
        --text-color: #e6e6e7;
        --secondary-text-color: #94b5d2;
        --tertiary-text-color: #b2bec6;
        --divider-color: #223241;
        --primary-button-color: #e8f6f7;
        --primary-button-border: #33e1d0;
        --primary-button-hover: #7be6da;
        --primary-button-hover-color: #24b1a2;
        --primary-button-hover-bg: #12222d;
        --footer-color: #586675;
    }
}
@media (max-width: 650px) {
    .pln-mainbox {padding: 1.2em 0.2em;}
    .btn-outline {max-width: 99vw;}
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="pln-mainbox">
        <div class="pln-title">Polar Minds</div>
        <div class="pln-inst">
            Universidade de São Paulo<br>
            Instituto de Ciências Matemáticas e de Computação
        </div>
        <div class="pln-auth">
            Desenvolvido por alunos de graduação e pós-graduação<br>
            Disciplina de Processamento de Linguagem Natural
        </div>
        <hr class="divider-slim">
        <div class="link-btns">
            <a class="btn-outline" href="http://localhost:8501/AED">📊 Análise Exploratória</a>
            <a class="btn-outline" href="http://localhost:8501/PLN_Classica">🔡 PLN Clássica</a>
            <a class="btn-outline" href="http://localhost:8501/PLN_Moderna">🤖 PLN Moderna</a>
        </div>
        <div class="pln-footer">
            © 2025 — ICMC USP
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
