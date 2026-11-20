"""
Regenera a base de transações classificadas usada pelo aplicativo.

Executa o ciclo completo: gera as transações sintéticas, constrói as
features, roda o Isolation Forest e grava o resultado comprimido em
dados/transacoes_classificadas.csv.gz.

Uso:
    python src/gerar_base.py
"""

from pathlib import Path

import pandas as pd

import gerar_transacoes
from features import construir_features
from modelo import baseline_regra, isolation_forest

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"


def main() -> None:
    gerar_transacoes.main()

    df = pd.read_csv(PASTA_DADOS / "transacoes.csv")
    df = construir_features(df)

    df["alerta_regra"] = baseline_regra(df)
    score, flag = isolation_forest(df)
    df["score_anomalia"] = score
    df["alerta_modelo"] = flag

    colunas = [
        "transacao_id", "cliente_id", "data_hora", "valor", "tipo", "canal",
        "score_anomalia", "alerta_regra", "alerta_modelo",
        "fraude", "padrao_fraude",
    ]
    destino = PASTA_DADOS / "transacoes_classificadas.csv.gz"
    df[colunas].to_csv(destino, index=False, compression="gzip")
    print(f"Base do aplicativo gravada em {destino}")


if __name__ == "__main__":
    main()
