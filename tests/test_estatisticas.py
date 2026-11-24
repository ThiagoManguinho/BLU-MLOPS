"""Testes das funções de cálculo do relatório de estatísticas (percentis, KS, PSI)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from relatorio_estatisticas import ic_bootstrap, ks, percentis, psi_categorico


def test_percentis_de_1_a_100():
    resumo = percentis(np.arange(1, 101))

    assert resumo["n"] == 100
    assert resumo["min"] == 1 and resumo["max"] == 100
    assert resumo["p50"] == pytest.approx(50.5)
    assert resumo["p90"] == pytest.approx(90.1)
    assert resumo["p95"] == pytest.approx(95.05)
    assert resumo["p99"] == pytest.approx(99.01)
    assert resumo["iqr"] == pytest.approx(resumo["p75"] - resumo["p25"])


def test_ic_bootstrap_contem_o_p95():
    valores = np.random.default_rng(1).exponential(100, size=300)

    baixo, alto = ic_bootstrap(valores, 95)

    assert baixo <= np.percentile(valores, 95) <= alto


def test_ks_amostras_iguais_e_disjuntas():
    a = np.arange(100.0)

    assert ks(a, a)[0] == pytest.approx(0)
    assert ks(a, a + 1000)[0] == pytest.approx(1)


def test_psi_distribuicoes_iguais_da_zero():
    valores = np.array([0, 1, 2, 3, 4, 5, 6, 7] * 10)

    psi, tabela = psi_categorico(valores, valores, list(range(8)))

    assert psi == pytest.approx(0)
    assert tabela["p_ref"].sum() == pytest.approx(1)


def test_psi_distribuicao_deslocada_indica_drift():
    referencia = np.array([0] * 50 + [1] * 50)
    producao = np.array([6] * 50 + [7] * 50)

    psi, _ = psi_categorico(referencia, producao, list(range(8)))

    assert psi > 0.25
