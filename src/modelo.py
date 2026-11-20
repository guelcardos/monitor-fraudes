"""
Modelos de detecção: regra de negócio (baseline) e Isolation Forest.

O baseline representa o que um time de prevenção faria sem machine
learning: regras fixas sobre valor e horário. O Isolation Forest é
treinado de forma NÃO supervisionada — nunca vê o rótulo de fraude —
e aprende o que é "normal" para isolar o que destoa.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import COLUNAS_MODELO

SEMENTE = 42

# capacidade operacional: fração das transações que o time de
# prevenção consegue revisar por período
TAXA_REVISAO = 0.01


def baseline_regra(df: pd.DataFrame) -> np.ndarray:
    """Regra de negócio clássica:

    - valor acima de 6x a mediana do cliente; OU
    - madrugada (0h-5h) com valor acima de 2x a mediana.
    """
    regra_valor = df["razao_valor"] > 6
    regra_madrugada = (df["eh_madrugada"] == 1) & (df["razao_valor"] > 2)
    return (regra_valor | regra_madrugada).astype(int).to_numpy()


def isolation_forest(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Treina o Isolation Forest e devolve (score, flag).

    O score é invertido para que MAIOR = MAIS anômalo. A flag marca o
    1% mais anômalo (mesma capacidade de revisão do baseline ter).
    """
    X = StandardScaler().fit_transform(df[COLUNAS_MODELO])

    modelo = IsolationForest(
        n_estimators=300,
        contamination=TAXA_REVISAO,
        random_state=SEMENTE,
        n_jobs=-1,
    )
    modelo.fit(X)

    score = -modelo.score_samples(X)  # maior = mais anômalo
    corte = np.quantile(score, 1 - TAXA_REVISAO)
    flag = (score >= corte).astype(int)
    return score, flag
