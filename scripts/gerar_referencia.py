"""Gera o conjunto de referência para comparar com o log de produção (KS/PSI).

Passa cada imagem de uma pasta (padrão: as 245 fotos de treino do modelo,
`train-maquete/images`) pelas **mesmas** funções que o serviço usa para
montar o log (`monitoramento.caracteristicas_imagem` e
`caracteristicas_saida`) e pelo próprio modelo. Assim referência e log têm o
mesmo formato e podem ser comparados variável a variável.

Saídas:
- referencia/referencia.jsonl: uma linha por imagem, no formato do log;
- referencia/resumo.json: média, desvio e percentis de cada variável. Os
  limites de "imagem suspeita" e as faixas dos histogramas Prometheus do
  serviço foram escolhidos a partir desse resumo.

Uso: uv run python scripts/gerar_referencia.py [--pasta DIR] [--saida DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from _variaveis import VARIAVEIS, valor
from PIL import Image

from blu_service import model, monitoramento
from blu_service.logic import classificar_ocupacao

_EXTENSOES = {".jpg", ".jpeg", ".png"}
_PERCENTIS = (1, 5, 25, 50, 75, 95, 99)


def _linha(caminho: Path, limiar: float) -> dict:
    with Image.open(caminho) as imagem:
        imagem.load()
        entrada = monitoramento.caracteristicas_imagem(imagem)
        entrada["content_length"] = caminho.stat().st_size
        inicio = time.perf_counter()
        deteccoes = model.prever(imagem)
        latencia = round((time.perf_counter() - inicio) * 1000, 3)
        resultado = classificar_ocupacao(deteccoes, limiar)
        saida = monitoramento.caracteristicas_saida(resultado, imagem.width, imagem.height)
    return {
        "versao_log": monitoramento.VERSAO_LOG,
        "origem": "referencia",
        "arquivo": caminho.name,
        "latencia_modelo_ms": latencia,
        "limiar_confianca": limiar,
        "entrada": entrada,
        "saida": saida,
        "alertas": monitoramento.motivos_suspeita(entrada, saida),
    }


def resumir(linhas: list[dict]) -> dict:
    resumo = {"n_imagens": len(linhas), "variaveis": {}}
    for nome, secao, chave in VARIAVEIS:
        valores = np.array(
            [v for linha in linhas if (v := valor(linha, secao, chave)) is not None]
        )
        if valores.size == 0:
            continue
        resumo["variaveis"][nome] = {
            "n": int(valores.size),
            "media": round(float(valores.mean()), 4),
            "desvio": round(float(valores.std()), 4),
            "min": round(float(valores.min()), 4),
            "max": round(float(valores.max()), 4),
            **{
                f"p{p}": round(float(np.percentile(valores, p)), 4)
                for p in _PERCENTIS
            },
        }
    totais = [linha["saida"]["total_vagas"] for linha in linhas]
    resumo["total_vagas_frequencia"] = {
        str(n): totais.count(n) for n in sorted(set(totais))
    }
    return resumo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pasta", type=Path, default=Path("train-maquete/images"))
    parser.add_argument("--saida", type=Path, default=Path("referencia"))
    args = parser.parse_args()

    imagens = sorted(p for p in args.pasta.glob("*") if p.suffix.lower() in _EXTENSOES)
    if not imagens:
        print(f"nenhuma imagem em {args.pasta}", file=sys.stderr)
        return 1

    limiar = float(os.environ.get("BLU_CONF_THRESHOLD", "0.25"))
    args.saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    with (args.saida / "referencia.jsonl").open("w", encoding="utf-8") as arquivo:
        for i, caminho in enumerate(imagens, start=1):
            linha = _linha(caminho, limiar)
            linhas.append(linha)
            arquivo.write(json.dumps(linha, ensure_ascii=False, separators=(",", ":")) + "\n")
            if i % 25 == 0 or i == len(imagens):
                print(f"{i}/{len(imagens)} imagens processadas")

    resumo = resumir(linhas)
    resumo["pasta"] = args.pasta.as_posix()
    (args.saida / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"referência gravada em {args.saida}/ ({len(linhas)} imagens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
