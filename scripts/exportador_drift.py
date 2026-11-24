"""Exportador de drift: KS, PSI e percentis exatos ao vivo, para o Grafana.

Acompanha o log do serviço (`logs/inferencias.jsonl`) e, a cada
`BLU_DRIFT_INTERVALO` segundos, compara a **janela** dos últimos
`BLU_DRIFT_JANELA` pedidos bem-sucedidos com o **conjunto de referência**:

- variável de entrada `nitidez` (contínua): distância de Kolmogorov-Smirnov,
  p-valor e o D crítico para alfa = 0,05;
- variável de saída `ocupadas` (0 a 7, discreta): Population Stability Index,
  uma faixa por valor;
- tempo de resposta do servidor (`latencia_total_ms`) na janela, por endpoint:
  mínimo, quartis, p90, p95, p99, máximo e bigodes do boxplot. São percentis
  exatos, calculados pedido a pedido, e não estimados pelas faixas do
  histograma do /metrics.

Expõe tudo no formato Prometheus em http://localhost:9101/metrics. O
Prometheus coleta e o Grafana mostra no painel "BLU - Estatísticas (AV2)".

A referência é `estatisticas/respostas_referencia.csv` (as 245 fotos de treino
submetidas ao serviço pelo `just av2`). Se ele não existir, usa
`referencia/referencia.jsonl`, que tem as mesmas variáveis.

Uso fora do contêiner: uv run python scripts/exportador_drift.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from _estatisticas import (
    CATEGORIAS_OCUPADAS,
    ks,
    ks_critico,
    percentis,
    psi_categorico,
)
from prometheus_client import Gauge, start_http_server

LOG = Path(os.environ.get("BLU_LOG_PATH", "logs/inferencias.jsonl"))
REFERENCIA_CSV = Path(os.environ.get("BLU_REFERENCIA_CSV", "estatisticas/respostas_referencia.csv"))
REFERENCIA_JSONL = Path(os.environ.get("BLU_REFERENCIA_JSONL", "referencia/referencia.jsonl"))
JANELA = int(os.environ.get("BLU_DRIFT_JANELA", "100"))
INTERVALO = float(os.environ.get("BLU_DRIFT_INTERVALO", "10"))
PORTA = int(os.environ.get("BLU_DRIFT_PORTA", "9101"))
# Abaixo disso o KS e os percentis da janela não querem dizer muito.
MINIMO_AMOSTRAS = int(os.environ.get("BLU_DRIFT_MINIMO", "30"))

ENDPOINTS = ("/prever_ocupacao", "/prever_ocupacao_imagem")
QUANTIS_NITIDEZ = (0.25, 0.5, 0.75)
ESTATISTICAS_LATENCIA = (
    "min", "bigode_inferior", "p25", "p50", "p75", "p90", "p95", "p99", "bigode_superior", "max",
)

# --- Métricas -------------------------------------------------------------------

JANELA_AMOSTRAS = Gauge(
    "blu_drift_janela_amostras", "Pedidos bem-sucedidos na janela comparada com a referência."
)
REFERENCIA_AMOSTRAS = Gauge(
    "blu_drift_referencia_amostras", "Imagens no conjunto de referência."
)
KS_NITIDEZ = Gauge(
    "blu_drift_ks_nitidez",
    "Distância D de Kolmogorov-Smirnov da nitidez: janela x referência (0 = iguais, 1 = disjuntas).",
)
KS_CRITICO = Gauge(
    "blu_drift_ks_nitidez_critico",
    "D crítico do KS para alfa = 0,05 com os tamanhos atuais; D acima dele indica drift.",
)
KS_PVALOR = Gauge("blu_drift_ks_nitidez_pvalor", "p-valor do KS da nitidez.")
PSI_OCUPADAS = Gauge(
    "blu_drift_psi_ocupadas",
    "PSI de vagas ocupadas por imagem: janela x referência (< 0,1 estável; > 0,25 drift).",
)
OCUPADAS_PROPORCAO = Gauge(
    "blu_drift_ocupadas_proporcao",
    "Proporção das imagens com cada número de vagas ocupadas.",
    ["conjunto", "valor"],
)
NITIDEZ_QUANTIL = Gauge(
    "blu_drift_nitidez_quantil", "Quartis da nitidez.", ["conjunto", "quantil"]
)
LATENCIA = Gauge(
    "blu_janela_latencia_ms",
    "Tempo de resposta do servidor nos últimos pedidos da janela, por endpoint (percentis exatos).",
    ["endpoint", "estatistica"],
)
LATENCIA_AMOSTRAS = Gauge(
    "blu_janela_latencia_amostras", "Pedidos na janela de latência, por endpoint.", ["endpoint"]
)
ULTIMO_PEDIDO = Gauge(
    "blu_drift_ultimo_pedido_timestamp_segundos", "Horário (Unix) do pedido mais recente da janela."
)
ATUALIZACAO = Gauge(
    "blu_drift_atualizacao_timestamp_segundos", "Horário (Unix) do último cálculo."
)


# --- Referência -----------------------------------------------------------------


def carregar_referencia() -> tuple[np.ndarray, np.ndarray, str]:
    nitidez, ocupadas = [], []
    if REFERENCIA_CSV.exists():
        with REFERENCIA_CSV.open(encoding="utf-8") as arquivo:
            for linha in csv.DictReader(arquivo):
                if linha.get("status_http") == "200" and linha.get("nitidez"):
                    nitidez.append(float(linha["nitidez"]))
                    ocupadas.append(int(float(linha["ocupadas"])))
        origem = str(REFERENCIA_CSV)
    else:
        with REFERENCIA_JSONL.open(encoding="utf-8") as arquivo:
            for texto in arquivo:
                registro = json.loads(texto)
                entrada, saida = registro.get("entrada") or {}, registro.get("saida") or {}
                if entrada.get("nitidez") is not None and saida.get("ocupadas") is not None:
                    nitidez.append(float(entrada["nitidez"]))
                    ocupadas.append(int(saida["ocupadas"]))
        origem = str(REFERENCIA_JSONL)
    return np.array(nitidez), np.array(ocupadas), origem


# --- Leitura do log, só o que é novo ---------------------------------------------


class LeitorLog:
    """Lê o log aos poucos, do ponto onde parou. Se o arquivo encolher (foi
    rotacionado para `.1` e começou de novo), volta ao início."""

    def __init__(self, caminho: Path):
        self.caminho = caminho
        self.posicao = 0

    def novas_linhas(self) -> list[dict]:
        if not self.caminho.exists():
            return []
        if self.caminho.stat().st_size < self.posicao:
            self.posicao = 0
        registros = []
        with self.caminho.open("rb") as arquivo:
            arquivo.seek(self.posicao)
            for bruta in arquivo:
                if not bruta.endswith(b"\n"):
                    break  # linha ainda sendo escrita; lê na próxima vez
                self.posicao += len(bruta)
                try:
                    registros.append(json.loads(bruta))
                except json.JSONDecodeError:
                    continue
        return registros


# --- Cálculo --------------------------------------------------------------------


def atualizar(janela: deque, latencias: dict[str, deque], ref_nit, ref_ocu) -> None:
    for endpoint, valores in latencias.items():
        LATENCIA_AMOSTRAS.labels(endpoint=endpoint).set(len(valores))
        if len(valores) >= MINIMO_AMOSTRAS:
            resumo = percentis(np.array(valores))
            for estatistica in ESTATISTICAS_LATENCIA:
                LATENCIA.labels(endpoint=endpoint, estatistica=estatistica).set(resumo[estatistica])

    JANELA_AMOSTRAS.set(len(janela))
    if janela:
        ULTIMO_PEDIDO.set(janela[-1][2])
    if len(janela) >= MINIMO_AMOSTRAS:
        nitidez = np.array([n for n, _, _ in janela])
        ocupadas = np.array([o for _, o, _ in janela])

        d, pvalor = ks(ref_nit, nitidez)
        KS_NITIDEZ.set(d)
        KS_PVALOR.set(pvalor)
        KS_CRITICO.set(ks_critico(ref_nit.size, nitidez.size))
        for q, valor in zip(QUANTIS_NITIDEZ, np.quantile(nitidez, QUANTIS_NITIDEZ), strict=True):
            NITIDEZ_QUANTIL.labels(conjunto="janela", quantil=str(q)).set(valor)

        psi, por_categoria = psi_categorico(ref_ocu, ocupadas, CATEGORIAS_OCUPADAS)
        PSI_OCUPADAS.set(psi)
        for valor, proporcao in zip(por_categoria["valor"], por_categoria["p_prod"], strict=True):
            OCUPADAS_PROPORCAO.labels(conjunto="janela", valor=str(valor)).set(proporcao)
    ATUALIZACAO.set(time.time())


def main() -> int:
    ref_nit, ref_ocu, origem = carregar_referencia()
    if ref_nit.size < MINIMO_AMOSTRAS:
        print(f"referência com só {ref_nit.size} imagens ({origem})", file=sys.stderr)
        return 1
    REFERENCIA_AMOSTRAS.set(ref_nit.size)
    for q, valor in zip(QUANTIS_NITIDEZ, np.quantile(ref_nit, QUANTIS_NITIDEZ), strict=True):
        NITIDEZ_QUANTIL.labels(conjunto="referencia", quantil=str(q)).set(valor)
    for valor in CATEGORIAS_OCUPADAS:
        OCUPADAS_PROPORCAO.labels(conjunto="referencia", valor=str(valor)).set(
            float((ref_ocu == valor).mean())
        )

    start_http_server(PORTA)
    print(
        f"exportador de drift em :{PORTA}/metrics | referência: {ref_nit.size} imagens "
        f"({origem}) | log: {LOG} | janela: {JANELA} pedidos | a cada {INTERVALO:.0f} s",
        flush=True,
    )

    leitor = LeitorLog(LOG)
    janela: deque = deque(maxlen=JANELA)  # (nitidez, ocupadas, ts_unix)
    latencias = {endpoint: deque(maxlen=JANELA) for endpoint in ENDPOINTS}
    while True:
        for registro in leitor.novas_linhas():
            if registro.get("status_http") != 200 or registro.get("endpoint") not in ENDPOINTS:
                continue
            if registro.get("latencia_total_ms") is not None:
                latencias[registro["endpoint"]].append(float(registro["latencia_total_ms"]))
            nitidez = (registro.get("entrada") or {}).get("nitidez")
            ocupadas = (registro.get("saida") or {}).get("ocupadas")
            if nitidez is not None and ocupadas is not None:
                janela.append((float(nitidez), int(ocupadas), float(registro.get("ts_unix", 0))))
        atualizar(janela, latencias, ref_nit, ref_ocu)
        time.sleep(INTERVALO)


if __name__ == "__main__":
    sys.exit(main())
