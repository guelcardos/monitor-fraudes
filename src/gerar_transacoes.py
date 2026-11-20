"""
Gerador de transações bancárias sintéticas com fraudes plantadas.

Simula 90 dias de transações de um banco fictício. Cada cliente tem um
perfil próprio de comportamento (ticket médio, horários habituais,
canais preferidos); as transações legítimas seguem o perfil e as
fraudulentas o violam, segundo quatro padrões clássicos:

1. VALOR_ATIPICO   — valor muito acima do padrão do cliente;
2. MADRUGADA       — transação de valor alto entre 1h e 5h;
3. RAJADA          — várias transações em poucos minutos;
4. CANAL_NOVO      — canal que o cliente nunca usou + valor elevado.

As fraudes são rotuladas (coluna `fraude`) apenas para avaliação do
modelo — o treinamento é não supervisionado e não vê o rótulo.

Uso:
    python src/gerar_transacoes.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
SEMENTE = 42

QTD_CLIENTES = 800
DIAS = 90
DATA_INICIO = pd.Timestamp("2026-01-05")
TIPOS = ["PIX", "TED", "CARTAO_DEBITO", "BOLETO"]
CANAIS = ["APP", "INTERNET_BANKING", "ATM", "AGENCIA"]

# fraudes plantadas por padrão
QTD_FRAUDES = {
    "VALOR_ATIPICO": 180,
    "MADRUGADA": 120,
    "RAJADA": 40,      # clientes atacados (cada um gera 4-7 transações)
    "CANAL_NOVO": 110,
}


def criar_perfis(rng: np.random.Generator) -> pd.DataFrame:
    """Sorteia o perfil de comportamento de cada cliente."""
    clientes = []
    for i in range(QTD_CLIENTES):
        ticket = rng.lognormal(mean=4.6, sigma=0.7)  # mediana ~R$ 100
        clientes.append(
            {
                "cliente_id": f"C{i + 1:04d}",
                "ticket_medio": round(ticket, 2),
                "desvio_ticket": round(ticket * rng.uniform(0.25, 0.6), 2),
                "hora_pico": int(rng.integers(9, 21)),
                "transacoes_dia": rng.uniform(0.3, 3.0),
                # cada cliente usa 2 canais habituais
                "canais": rng.choice(CANAIS, size=2, replace=False).tolist(),
            }
        )
    return pd.DataFrame(clientes)


def gerar_legitimas(perfis: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Gera as transações legítimas seguindo o perfil de cada cliente."""
    transacoes = []
    for perfil in perfis.itertuples():
        qtd = rng.poisson(perfil.transacoes_dia * DIAS)
        for _ in range(qtd):
            dia = int(rng.integers(0, DIAS))
            # horário concentrado em torno do pico do cliente (desvio 3h),
            # limitado ao intervalo 6h-23h
            hora = int(np.clip(rng.normal(perfil.hora_pico, 3), 6, 23))
            minuto = int(rng.integers(0, 60))
            valor = max(1.0, rng.normal(perfil.ticket_medio, perfil.desvio_ticket))
            transacoes.append(
                {
                    "cliente_id": perfil.cliente_id,
                    "data_hora": DATA_INICIO
                    + pd.Timedelta(days=dia, hours=hora, minutes=minuto),
                    "valor": round(valor, 2),
                    "tipo": rng.choice(TIPOS),
                    "canal": rng.choice(perfil.canais),
                    "fraude": 0,
                    "padrao_fraude": "",
                }
            )
    return transacoes


def gerar_fraudes(perfis: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Planta os quatro padrões de fraude em clientes sorteados."""
    fraudes = []

    def transacao(perfil, data_hora, valor, canal, padrao):
        return {
            "cliente_id": perfil.cliente_id,
            "data_hora": data_hora,
            "valor": round(valor, 2),
            "tipo": rng.choice(["PIX", "TED"]),  # fraude prefere liquidez
            "canal": canal,
            "fraude": 1,
            "padrao_fraude": padrao,
        }

    # 1. Valor muito acima do padrão do cliente (8x a 25x o ticket)
    for perfil in perfis.sample(QTD_FRAUDES["VALOR_ATIPICO"], random_state=1).itertuples():
        dia = int(rng.integers(0, DIAS))
        hora = int(rng.integers(8, 22))
        valor = perfil.ticket_medio * rng.uniform(8, 25)
        fraudes.append(
            transacao(
                perfil,
                DATA_INICIO + pd.Timedelta(days=dia, hours=hora,
                                           minutes=int(rng.integers(0, 60))),
                valor, rng.choice(perfil.canais), "VALOR_ATIPICO",
            )
        )

    # 2. Madrugada: valor alto entre 1h e 5h
    for perfil in perfis.sample(QTD_FRAUDES["MADRUGADA"], random_state=2).itertuples():
        dia = int(rng.integers(0, DIAS))
        hora = int(rng.integers(1, 5))
        valor = perfil.ticket_medio * rng.uniform(3, 10)
        fraudes.append(
            transacao(
                perfil,
                DATA_INICIO + pd.Timedelta(days=dia, hours=hora,
                                           minutes=int(rng.integers(0, 60))),
                valor, rng.choice(perfil.canais), "MADRUGADA",
            )
        )

    # 3. Rajada: 4 a 7 transações em poucos minutos no mesmo cliente
    for perfil in perfis.sample(QTD_FRAUDES["RAJADA"], random_state=3).itertuples():
        dia = int(rng.integers(0, DIAS))
        hora = int(rng.integers(0, 24))
        inicio = DATA_INICIO + pd.Timedelta(days=dia, hours=hora,
                                            minutes=int(rng.integers(0, 50)))
        for sequencia in range(int(rng.integers(4, 8))):
            valor = perfil.ticket_medio * rng.uniform(1.5, 5)
            fraudes.append(
                transacao(
                    perfil,
                    inicio + pd.Timedelta(minutes=int(sequencia * rng.integers(1, 3))),
                    valor, rng.choice(perfil.canais), "RAJADA",
                )
            )

    # 4. Canal que o cliente nunca usa + valor elevado
    for perfil in perfis.sample(QTD_FRAUDES["CANAL_NOVO"], random_state=4).itertuples():
        canais_novos = [c for c in CANAIS if c not in perfil.canais]
        dia = int(rng.integers(0, DIAS))
        hora = int(rng.integers(7, 23))
        valor = perfil.ticket_medio * rng.uniform(4, 12)
        fraudes.append(
            transacao(
                perfil,
                DATA_INICIO + pd.Timedelta(days=dia, hours=hora,
                                           minutes=int(rng.integers(0, 60))),
                valor, rng.choice(canais_novos), "CANAL_NOVO",
            )
        )

    return fraudes


def main() -> None:
    rng = np.random.default_rng(SEMENTE)
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)

    perfis = criar_perfis(rng)
    transacoes = gerar_legitimas(perfis, rng) + gerar_fraudes(perfis, rng)

    df = (
        pd.DataFrame(transacoes)
        .sort_values("data_hora")
        .reset_index(drop=True)
    )
    df.insert(0, "transacao_id", [f"T{i + 1:06d}" for i in range(len(df))])

    df.to_csv(PASTA_DADOS / "transacoes.csv", index=False)
    perfis[["cliente_id", "ticket_medio", "canais"]].to_csv(
        PASTA_DADOS / "clientes.csv", index=False
    )

    qtd_fraudes = int(df["fraude"].sum())
    print(f"{len(df)} transacoes gravadas em {PASTA_DADOS / 'transacoes.csv'}")
    print(f"Clientes: {QTD_CLIENTES} | Periodo: {DIAS} dias")
    print(f"Fraudes plantadas: {qtd_fraudes} "
          f"({qtd_fraudes / len(df) * 100:.2f}% do total)")
    print(df[df.fraude == 1]["padrao_fraude"].value_counts().to_string())


if __name__ == "__main__":
    main()
