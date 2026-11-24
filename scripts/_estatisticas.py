"""Cálculos de monitoramento usados pelo relatório e pelo exportador de drift.

Só depende de numpy e scipy, para rodar também no contêiner do exportador
(`scripts/exportador_drift.py`), que não tem pandas nem matplotlib.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

PERCENTIS = (25, 50, 75, 90, 95, 99)
CATEGORIAS_OCUPADAS = list(range(8))  # 0 a 7 vagas ocupadas (a maquete tem 7)
EPS = 1e-6
ALFA = 0.05
# Coeficiente do valor crítico do KS de duas amostras para alfa = 0,05:
# D_crítico = 1,358 * sqrt((n + m) / (n * m)).
_C_ALFA_005 = 1.358


def percentis(valores) -> dict:
    """Resumo não paramétrico: não supõe nenhuma distribuição.

    Usa a interpolação linear do numpy (padrão), a mesma do Excel
    (PERCENTIL.INC) e do pandas.
    """
    valores = np.asarray(valores, dtype=float)
    resumo = {"n": int(valores.size), "min": float(valores.min()), "max": float(valores.max())}
    for q in PERCENTIS:
        resumo[f"p{q}"] = float(np.percentile(valores, q))
    resumo["iqr"] = resumo["p75"] - resumo["p25"]
    # Bigodes do boxplot (regra de Tukey): o valor mais extremo dentro de
    # 1,5 IQR além dos quartis. O que passa disso é ponto atípico.
    dentro = valores[
        (valores >= resumo["p25"] - 1.5 * resumo["iqr"])
        & (valores <= resumo["p75"] + 1.5 * resumo["iqr"])
    ]
    resumo["bigode_inferior"] = float(dentro.min())
    resumo["bigode_superior"] = float(dentro.max())
    resumo["atipicos"] = int(valores.size - dentro.size)
    return resumo


def ic_bootstrap(valores, q: float = 95, reamostragens: int = 2000, semente: int = 0):
    """Intervalo de 95% do percentil q por bootstrap (percentis da reamostragem)."""
    valores = np.asarray(valores, dtype=float)
    gerador = np.random.default_rng(semente)
    amostras = gerador.choice(valores, size=(reamostragens, valores.size), replace=True)
    estimativas = np.percentile(amostras, q, axis=1)
    return float(np.percentile(estimativas, 2.5)), float(np.percentile(estimativas, 97.5))


def ks(referencia, producao) -> tuple[float, float]:
    """Distância D de Kolmogorov-Smirnov (maior diferença entre as CDFs) e p-valor."""
    resultado = ks_2samp(referencia, producao)
    return float(resultado.statistic), float(resultado.pvalue)


def ks_critico(n: int, m: int) -> float:
    """D acima do qual o KS rejeita "mesma distribuição" com alfa = 0,05."""
    return _C_ALFA_005 * float(np.sqrt((n + m) / (n * m)))


def psi_categorico(referencia, producao, categorias=CATEGORIAS_OCUPADAS):
    """PSI com uma faixa por categoria.

    PSI = soma (p_prod - p_ref) * ln(p_prod / p_ref). Categorias sem nenhum
    caso recebem 1e-6 para o logaritmo existir. Devolve o PSI e, por
    categoria, as proporções de cada lado e a contribuição ao total.
    """
    referencia = np.asarray(referencia)
    producao = np.asarray(producao)
    p_ref = np.array([(referencia == c).mean() for c in categorias])
    p_prod = np.array([(producao == c).mean() for c in categorias])
    a, b = np.maximum(p_ref, EPS), np.maximum(p_prod, EPS)
    contribuicao = (b - a) * np.log(b / a)
    por_categoria = {
        "valor": list(categorias),
        "p_ref": p_ref,
        "p_prod": p_prod,
        "contribuicao": contribuicao,
    }
    return float(contribuicao.sum()), por_categoria


def leitura_psi(valor_psi: float) -> str:
    if valor_psi < 0.1:
        return "estável"
    if valor_psi < 0.25:
        return "mudança moderada"
    return "drift"
