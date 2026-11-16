"""Confere se o log tem o que precisa para histogramas, boxplots e KS/PSI.

Lê o log de produção (logs/inferencias.jsonl e os arquivos rotacionados
.1, .2, ...) e o conjunto de referência (referencia/referencia.jsonl), que têm
o mesmo formato, e para cada variável numérica:
- conta quantas amostras existem de cada lado;
- calcula o KS de duas amostras (scipy.stats.ks_2samp) e o PSI com 10 faixas
  pelos quantis da referência (mesma construção do exemplo 008 da disciplina);
- desenha histograma e boxplot lado a lado (referência × produção) em analise/.

Também calcula o KS sobre os histogramas de luminância por pixel agregados,
como no exemplo 003 da disciplina.

Uso: uv run python scripts/analisar_log.py [--log logs/inferencias.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _variaveis import VARIAVEIS, valor
from scipy.stats import ks_2samp

_GRAFICOS = (
    "nitidez",
    "luminancia_media",
    "luminancia_desvio",
    "total_vagas",
    "taxa_ocupacao",
    "confianca_media",
    "latencia_modelo_ms",
)
_EPS = 1e-6
_MIN_AMOSTRAS = 30


def ler_jsonl(caminhos: list[Path]) -> list[dict]:
    linhas = []
    for caminho in caminhos:
        with caminho.open(encoding="utf-8") as arquivo:
            linhas.extend(json.loads(linha) for linha in arquivo if linha.strip())
    return linhas


def arquivos_de_log(base: Path) -> list[Path]:
    """O arquivo atual e os rotacionados (.1 é o mais recente dos antigos)."""
    rotacionados = sorted(
        base.parent.glob(base.name + ".*"), key=lambda p: -int(p.suffix[1:] or 0)
    )
    return [p for p in rotacionados if p.suffix[1:].isdigit()] + ([base] if base.exists() else [])


def psi(referencia: np.ndarray, producao: np.ndarray, faixas: int = 10) -> float:
    """PSI = soma (p_prod - p_ref) * ln(p_prod / p_ref), faixas pelos quantis
    da referência. Variáveis discretas (ex.: total de vagas) geram quantis
    repetidos; nesse caso as faixas são os próprios valores distintos."""
    limites = np.unique(np.quantile(referencia, np.linspace(0, 1, faixas + 1)))
    if limites.size < 3:
        limites = np.unique(np.concatenate([referencia, producao]))
        limites = np.append(limites, limites[-1] + 1)
    limites[0], limites[-1] = -np.inf, np.inf
    p_ref = np.histogram(referencia, limites)[0] / referencia.size
    p_prod = np.histogram(producao, limites)[0] / producao.size
    p_ref, p_prod = np.maximum(p_ref, _EPS), np.maximum(p_prod, _EPS)
    return float(np.sum((p_prod - p_ref) * np.log(p_prod / p_ref)))


def leitura_psi(valor_psi: float) -> str:
    if valor_psi < 0.1:
        return "estável"
    if valor_psi < 0.25:
        return "mudança moderada"
    return "drift"


def ks_pixels(referencia: list[dict], producao: list[dict]) -> float | None:
    """D do KS entre os histogramas de luminância por pixel agregados."""
    def cdf(linhas):
        total = np.zeros(256)
        for linha in linhas:
            histograma = (linha.get("entrada") or {}).get("histograma_l")
            if histograma:
                total += np.asarray(histograma, dtype=float)
        return np.cumsum(total) / total.sum() if total.sum() else None

    a, b = cdf(referencia), cdf(producao)
    return None if a is None or b is None else float(np.max(np.abs(a - b)))


def grafico(nome: str, referencia: np.ndarray, producao: np.ndarray, pasta: Path) -> Path:
    figura, (eixo_h, eixo_b) = plt.subplots(1, 2, figsize=(11, 4))
    faixas = np.histogram_bin_edges(np.concatenate([referencia, producao]), bins=20)
    eixo_h.hist(referencia, bins=faixas, alpha=0.6, density=True, label=f"referência (n={referencia.size})")
    eixo_h.hist(producao, bins=faixas, alpha=0.6, density=True, label=f"produção (n={producao.size})")
    eixo_h.set_title(f"Histograma: {nome}")
    eixo_h.legend()
    eixo_b.boxplot([referencia, producao], tick_labels=["referência", "produção"])
    eixo_b.set_title(f"Boxplot: {nome}")
    figura.tight_layout()
    caminho = pasta / f"{nome}.png"
    figura.savefig(caminho, dpi=110)
    plt.close(figura)
    return caminho


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=Path("logs/inferencias.jsonl"))
    parser.add_argument("--referencia", type=Path, default=Path("referencia/referencia.jsonl"))
    parser.add_argument("--saida", type=Path, default=Path("analise"))
    args = parser.parse_args()

    arquivos = arquivos_de_log(args.log)
    if not arquivos:
        print(f"log não encontrado: {args.log}. Gere carga antes (just carga).", file=sys.stderr)
        return 1
    log = ler_jsonl(arquivos)
    referencia = ler_jsonl([args.referencia])
    sucesso = [linha for linha in log if linha.get("status_http") == 200]
    erros = len(log) - len(sucesso)

    print(f"Log: {len(log)} pedidos em {len(arquivos)} arquivo(s); {len(sucesso)} com 200, {erros} com erro")
    print(f"Referência: {len(referencia)} imagens ({args.referencia})\n")

    args.saida.mkdir(parents=True, exist_ok=True)
    tabela = []
    for nome, secao, chave in VARIAVEIS:
        ref = np.array([v for linha in referencia if (v := valor(linha, secao, chave)) is not None])
        prod = np.array([v for linha in sucesso if (v := valor(linha, secao, chave)) is not None])
        if ref.size == 0 or prod.size == 0:
            continue
        ks = ks_2samp(ref, prod)
        valor_psi = psi(ref, prod)
        tabela.append(
            {
                "variavel": nome,
                "n_ref": ref.size,
                "n_prod": prod.size,
                "media_ref": round(float(ref.mean()), 3),
                "media_prod": round(float(prod.mean()), 3),
                "ks_D": round(float(ks.statistic), 3),
                "ks_p": float(f"{ks.pvalue:.3g}"),
                "psi": round(valor_psi, 3),
                "leitura_psi": leitura_psi(valor_psi),
                "amostra_suficiente": prod.size >= _MIN_AMOSTRAS,
            }
        )
        if nome in _GRAFICOS:
            grafico(nome, ref, prod, args.saida)

    resultado = pd.DataFrame(tabela)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(resultado.to_string(index=False))
    resultado.to_csv(args.saida / "drift.csv", index=False)

    d_pixels = ks_pixels(referencia, sucesso)
    if d_pixels is not None:
        print(f"\nKS nos histogramas de luminância por pixel (agregados): D = {d_pixels:.3f}")

    motivos = {}
    for linha in sucesso:
        for motivo in linha.get("alertas", []):
            motivos[motivo] = motivos.get(motivo, 0) + 1
    print("Imagens suspeitas no log, por motivo:", motivos or "nenhuma")

    if len(sucesso) < _MIN_AMOSTRAS:
        print(
            f"\nAtenção: só {len(sucesso)} pedidos com sucesso. KS e PSI ficam instáveis com "
            f"menos de {_MIN_AMOSTRAS} amostras; gere mais carga."
        )
    print(f"\nGráficos e tabela em {args.saida}/ (drift.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
