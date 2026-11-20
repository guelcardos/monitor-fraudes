# 🔍 Monitor de Fraudes — Aplicativo Web (Streamlit)

Aplicativo web interativo para **monitoramento de transações suspeitas**, construído em Streamlit sobre o detector de anomalias do projeto [deteccao-anomalias-financeiras](https://github.com/guelcardos/deteccao-anomalias-financeiras).

> ⚠️ Todos os dados são **100% fictícios** (118 mil transações sintéticas de um banco fictício). Nenhum dado real foi utilizado.

## ✨ O que o app oferece

- **KPIs em tempo real**: transações, alertas, taxa de alerta e valor em risco no recorte selecionado
- **Sensibilidade ajustável**: um slider controla o percentual de transações mais anômalas que entram na fila de revisão — e os números reagem na hora
- **Filtros** por período, tipo de transação (PIX, TED, débito, boleto) e canal
- **Validação interativa**: painel que mostra precisão e recall contra as 613 fraudes plantadas conforme a sensibilidade muda — o trade-off entre capturar mais fraude e gerar mais falsos positivos fica visível
- **Fila de revisão**: as transações mais suspeitas ranqueadas por score, com barra de progresso visual e download em CSV

## 🚀 Como executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501` lendo a base pré-processada de `dados/transacoes_classificadas.csv.gz`.

Para regenerar a base do zero (novas transações + novo treinamento do modelo):

```bash
python src/gerar_base.py
```

## ☁️ Deploy no Streamlit Community Cloud (gratuito)

1. Faça o fork ou publique este repositório no seu GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte sua conta do GitHub
3. Clique em **New app**, selecione o repositório, branch `main` e arquivo `app.py`
4. Pronto — o app ganha uma URL pública para compartilhar

## ⚙️ Como funciona por trás

A base é pré-processada pelo pipeline do projeto principal:

1. **Geração**: 118 mil transações sintéticas (90 dias, 800 clientes com perfis próprios de comportamento) + 613 fraudes plantadas em 4 padrões (valor atípico, madrugada, rajada, canal novo)
2. **Features por cliente**: o desvio é medido contra o padrão histórico do próprio cliente (razão sobre a mediana, desvio robusto, velocidade entre transações, familiaridade do canal)
3. **Isolation Forest** (não supervisionado): atribui a cada transação um score de anomalia — o modelo nunca viu os rótulos de fraude
4. **O app**: aplica o corte de score escolhido no slider e monta a visão operacional

## 📁 Estrutura do projeto

```
monitor-fraudes/
├── app.py                   # aplicativo Streamlit
├── dados/
│   └── transacoes_classificadas.csv.gz   # base pré-processada (2,5 MB)
├── src/
│   ├── gerar_base.py        # regenera a base completa do zero
│   ├── gerar_transacoes.py  # transações sintéticas + fraudes plantadas
│   ├── features.py          # engenharia de features por cliente
│   └── modelo.py            # Isolation Forest + regra baseline
├── requirements.txt
└── README.md
```

## 🛠️ Tecnologias

- **Streamlit** — interface web
- **Plotly** — gráficos interativos
- **scikit-learn** — Isolation Forest
- **pandas / numpy** — dados e features

---

Desenvolvido por **Miguel Cardoso** · [LinkedIn](https://www.linkedin.com/in/guelcardoso/) · [GitHub](https://github.com/guelcardos)
