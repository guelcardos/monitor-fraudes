"""
Monitor de Fraudes — aplicativo Streamlit.

Interface interativa sobre o detector de anomalias (Isolation Forest)
do projeto deteccao-anomalias-financeiras: KPIs, sensibilidade
ajustável, gráficos e fila de revisão com download.

Execução local:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CAMINHO_BASE = Path(__file__).parent / "dados" / "transacoes_classificadas.csv.gz"

COR_PRINCIPAL = "#2a78d6"
COR_ALERTA = "#eb6834"
COR_NEUTRA = "#9aa0a6"

st.set_page_config(
    page_title="Monitor de Fraudes",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_base() -> pd.DataFrame:
    df = pd.read_csv(CAMINHO_BASE, parse_dates=["data_hora"])
    df["data"] = df["data_hora"].dt.date
    return df


df = carregar_base()

# ---------------------------------------------------------------------------
# Barra lateral — filtros
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 Monitor de Fraudes")
st.sidebar.caption(
    "Detecção de anomalias com Isolation Forest sobre transações "
    "bancárias **100% fictícias**."
)

sensibilidade = st.sidebar.slider(
    "Sensibilidade do alerta (% mais anômalo)",
    min_value=0.2, max_value=5.0, value=1.0, step=0.2,
    help="Percentual das transações mais anômalas que entram na fila "
         "de revisão. Mais sensível = mais alertas = mais falsos positivos.",
)

datas = st.sidebar.date_input(
    "Período",
    value=(df["data"].min(), df["data"].max()),
    min_value=df["data"].min(),
    max_value=df["data"].max(),
)

tipos = st.sidebar.multiselect(
    "Tipo de transação", sorted(df["tipo"].unique()),
    default=sorted(df["tipo"].unique()),
)
canais = st.sidebar.multiselect(
    "Canal", sorted(df["canal"].unique()),
    default=sorted(df["canal"].unique()),
)

# corte de score definido na base COMPLETA (política global do banco),
# depois aplicado ao recorte filtrado
corte = float(np.quantile(df["score_anomalia"], 1 - sensibilidade / 100))
df["alerta"] = (df["score_anomalia"] >= corte).astype(int)

if isinstance(datas, tuple) and len(datas) == 2:
    inicio, fim = datas
else:
    inicio = fim = datas if not isinstance(datas, tuple) else datas[0]

visao = df[
    df["data"].between(inicio, fim)
    & df["tipo"].isin(tipos)
    & df["canal"].isin(canais)
]

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

st.title("Monitor de Fraudes")
st.caption(
    f"{len(visao):,} transações no recorte · corte de score em "
    f"{corte:.3f} (top {sensibilidade:.1f}% da base)".replace(",", ".")
)

alertas = visao[visao["alerta"] == 1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transações", f"{len(visao):,}".replace(",", "."))
col2.metric("Alertas", f"{len(alertas):,}".replace(",", "."))
col3.metric(
    "Taxa de alerta",
    f"{len(alertas) / len(visao) * 100:.2f}%" if len(visao) else "—",
)
col4.metric(
    "Valor em alerta",
    f"R$ {alertas['valor'].sum():,.0f}".replace(",", "."),
)

st.divider()

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

esquerda, direita = st.columns(2)

with esquerda:
    st.subheader("Alertas por dia")
    por_dia = (
        visao.groupby("data")
        .agg(transacoes=("transacao_id", "count"), alertas=("alerta", "sum"))
        .reset_index()
    )
    figura = px.bar(
        por_dia, x="data", y="alertas",
        color_discrete_sequence=[COR_ALERTA],
    )
    figura.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        xaxis_title=None, yaxis_title="Alertas",
    )
    st.plotly_chart(figura, width='stretch')

with direita:
    st.subheader("Distribuição do score")
    figura = go.Figure()
    figura.add_histogram(
        x=visao["score_anomalia"], nbinsx=60,
        marker_color=COR_PRINCIPAL, name="Transações",
    )
    figura.add_vline(
        x=corte, line_color=COR_ALERTA, line_width=2, line_dash="dash",
        annotation_text="corte", annotation_font_color=COR_ALERTA,
    )
    figura.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        yaxis_type="log", showlegend=False,
        xaxis_title="Score de anomalia", yaxis_title="Transações (log)",
    )
    st.plotly_chart(figura, width='stretch')

# ---------------------------------------------------------------------------
# Validação contra as fraudes plantadas
# ---------------------------------------------------------------------------

with st.expander("🎯 Validação contra as fraudes plantadas"):
    st.markdown(
        "Os dados sintéticos incluem **613 fraudes rotuladas** que o "
        "modelo nunca viu no treino. Mova o slider de sensibilidade e "
        "veja o trade-off: mais sensibilidade aumenta o recall "
        "(fraudes capturadas) ao custo da precisão (falsos positivos)."
    )
    fraudes = df[df["fraude"] == 1]
    capturadas = fraudes[fraudes["alerta"] == 1]
    alertas_total = df[df["alerta"] == 1]

    precisao = len(capturadas) / len(alertas_total) if len(alertas_total) else 0
    recall = len(capturadas) / len(fraudes) if len(fraudes) else 0

    v1, v2, v3 = st.columns(3)
    v1.metric("Precisão", f"{precisao:.1%}")
    v2.metric("Recall", f"{recall:.1%}")
    v3.metric(
        "Valor de fraude capturado",
        f"R$ {capturadas['valor'].sum():,.0f}".replace(",", "."),
    )

    recall_padrao = (
        capturadas.groupby("padrao_fraude").size()
        / fraudes.groupby("padrao_fraude").size()
    ).fillna(0).rename("recall").reset_index()
    figura = px.bar(
        recall_padrao, x="padrao_fraude", y="recall",
        color_discrete_sequence=[COR_PRINCIPAL], text_auto=".0%",
    )
    figura.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=280,
        xaxis_title=None, yaxis_title="Recall",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(figura, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Fila de revisão
# ---------------------------------------------------------------------------

st.subheader("Fila de revisão — transações mais suspeitas")

fila = (
    alertas.sort_values("score_anomalia", ascending=False)
    .loc[:, ["transacao_id", "cliente_id", "data_hora", "valor",
             "tipo", "canal", "score_anomalia"]]
    .head(200)
)

st.dataframe(
    fila,
    width='stretch',
    hide_index=True,
    column_config={
        "transacao_id": "Transação",
        "cliente_id": "Cliente",
        "data_hora": st.column_config.DatetimeColumn(
            "Data/hora", format="DD/MM/YYYY HH:mm"
        ),
        "valor": st.column_config.NumberColumn("Valor", format="R$ %.2f"),
        "tipo": "Tipo",
        "canal": "Canal",
        "score_anomalia": st.column_config.ProgressColumn(
            "Score", format="%.3f",
            min_value=float(df["score_anomalia"].min()),
            max_value=float(df["score_anomalia"].max()),
        ),
    },
)

st.download_button(
    "⬇️ Baixar alertas do recorte (CSV)",
    alertas.to_csv(index=False, sep=";").encode("utf-8"),
    file_name="alertas_fraude.csv",
    mime="text/csv",
)

st.caption(
    "Dados 100% fictícios, gerados pelo projeto "
    "[deteccao-anomalias-financeiras](https://github.com/guelcardos/deteccao-anomalias-financeiras). "
    "Desenvolvido por Miguel Cardoso · "
    "[LinkedIn](https://www.linkedin.com/in/guelcardoso/) · "
    "[GitHub](https://github.com/guelcardos)"
)
