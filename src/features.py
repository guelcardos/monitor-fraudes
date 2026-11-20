"""
Engenharia de features para detecção de anomalias.

O princípio central: uma transação não é suspeita pelo valor absoluto,
e sim pelo desvio em relação ao comportamento histórico DO PRÓPRIO
cliente. R$ 5.000 pode ser rotina para um cliente e um alerta grave
para outro.

Todas as features usam apenas dados observáveis (as transações em si);
o rótulo de fraude nunca entra no cálculo.
"""

import numpy as np
import pandas as pd


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve o DataFrame com as colunas de features adicionadas."""
    df = df.sort_values(["cliente_id", "data_hora"]).copy()
    df["data_hora"] = pd.to_datetime(df["data_hora"])

    # --- perfil histórico do cliente (mediana é robusta a outliers) ---
    estatisticas = df.groupby("cliente_id")["valor"].agg(
        mediana_cliente="median",
        iqr_cliente=lambda v: np.subtract(*np.percentile(v, [75, 25])) or 1.0,
    )
    df = df.merge(estatisticas, on="cliente_id", how="left")

    # 1. desvio do valor em relação ao padrão do cliente
    df["razao_valor"] = df["valor"] / df["mediana_cliente"]
    df["desvio_robusto"] = (df["valor"] - df["mediana_cliente"]) / df["iqr_cliente"]

    # 2. escala logarítmica do valor (distribuição é assimétrica)
    df["log_valor"] = np.log1p(df["valor"])

    # 3. horário
    df["hora"] = df["data_hora"].dt.hour
    df["eh_madrugada"] = df["hora"].between(0, 5).astype(int)

    # 4. velocidade: minutos desde a transação anterior do cliente
    df["minutos_desde_ultima"] = (
        df.groupby("cliente_id")["data_hora"].diff().dt.total_seconds() / 60
    )
    df["minutos_desde_ultima"] = df["minutos_desde_ultima"].fillna(60 * 24)
    # recorte em 24h e escala log (interessa o extremo inferior: rajadas)
    df["log_minutos_desde_ultima"] = np.log1p(
        df["minutos_desde_ultima"].clip(upper=60 * 24)
    )

    # 5. frequência: transações do cliente na última hora
    df["qtd_ultima_hora"] = (
        df.set_index("data_hora")
        .groupby("cliente_id")["valor"]
        .transform(lambda s: s.rolling("1h").count())
        .to_numpy()
    )

    # 6. familiaridade do canal: proporção histórica de uso pelo cliente
    freq_canal = (
        df.groupby(["cliente_id", "canal"]).size()
        / df.groupby("cliente_id").size()
    ).rename("freq_canal_cliente")
    df = df.merge(freq_canal, on=["cliente_id", "canal"], how="left")

    return df.sort_values("data_hora").reset_index(drop=True)


COLUNAS_MODELO = [
    "razao_valor",
    "desvio_robusto",
    "log_valor",
    "hora",
    "eh_madrugada",
    "log_minutos_desde_ultima",
    "qtd_ultima_hora",
    "freq_canal_cliente",
]
