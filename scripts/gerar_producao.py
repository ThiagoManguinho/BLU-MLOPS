"""Gera o conjunto de "produção": variações realistas das fotos de samples/.

As 3 fotos de `samples/` mostram a maquete inteira vista de cima, como a câmera
fixa de produção veria. Para simular um dia de operação dessa câmera, cada foto
vira `--por-imagem` variações (padrão 40, ou seja, 120 imagens), com:
- luz do dia mudando: brilho de 0,75 a 1,2 e contraste de 0,85 a 1,15;
- câmera levemente mexida: rotação de até ±3° e deslocamento de até ±2%,
  sem zoom (as quinas vazias recebem a cor da borda da foto);
- foco/lente: desfoque gaussiano leve (raio 0 a 1,2) e, em ~10% das imagens,
  mais forte (raio 2 a 3), como lente suja;
- ruído do sensor: ruído gaussiano com desvio de 0 a 6 níveis de cinza;
- compressão: JPEG com qualidade de 60 a 95.

A semente é fixa, então rodar de novo gera exatamente as mesmas imagens. Por
isso `dados/` não é versionado. O `manifesto.csv` registra a origem e os
parâmetros de cada imagem.

Uso: uv run python scripts/gerar_producao.py [--por-imagem 40] [--semente 42]
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

_EXTENSOES = {".jpg", ".jpeg", ".png"}
_FRACAO_LENTE_SUJA = 0.10


def sortear_parametros(aleatorio: random.Random) -> dict:
    lente_suja = aleatorio.random() < _FRACAO_LENTE_SUJA
    return {
        "brilho": round(aleatorio.uniform(0.75, 1.2), 3),
        "contraste": round(aleatorio.uniform(0.85, 1.15), 3),
        "rotacao_graus": round(aleatorio.uniform(-3, 3), 2),
        "desloc_x": round(aleatorio.uniform(-0.02, 0.02), 4),
        "desloc_y": round(aleatorio.uniform(-0.02, 0.02), 4),
        "desfoque_raio": round(
            aleatorio.uniform(2, 3) if lente_suja else aleatorio.uniform(0, 1.2), 2
        ),
        "ruido_sigma": round(aleatorio.uniform(0, 6), 2),
        "qualidade_jpeg": aleatorio.randint(60, 95),
    }


def aplicar(imagem: Image.Image, p: dict, gerador: np.random.Generator) -> Image.Image:
    largura, altura = imagem.size

    # Câmera mexida: gira e desloca sem zoom (as vagas encostam na borda das
    # fotos, um recorte as cortaria). As quinas que ficam vazias recebem a cor
    # média da borda da foto, em vez de preto.
    pixels = np.asarray(imagem)
    borda = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    imagem = imagem.rotate(
        p["rotacao_graus"],
        resample=Image.Resampling.BICUBIC,
        translate=(p["desloc_x"] * largura, p["desloc_y"] * altura),
        fillcolor=tuple(int(c) for c in borda.mean(axis=0)),
    )

    imagem = ImageEnhance.Brightness(imagem).enhance(p["brilho"])
    imagem = ImageEnhance.Contrast(imagem).enhance(p["contraste"])
    if p["desfoque_raio"] > 0:
        imagem = imagem.filter(ImageFilter.GaussianBlur(p["desfoque_raio"]))
    if p["ruido_sigma"] > 0:
        pixels = np.asarray(imagem, dtype=np.float32)
        pixels += gerador.normal(0, p["ruido_sigma"], pixels.shape)
        imagem = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return imagem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--origem", type=Path, default=Path("samples"))
    parser.add_argument("--saida", type=Path, default=Path("dados/producao"))
    parser.add_argument("--por-imagem", type=int, default=40)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

    fotos = sorted(p for p in args.origem.glob("*") if p.suffix.lower() in _EXTENSOES)
    if not fotos:
        print(f"nenhuma imagem em {args.origem}", file=sys.stderr)
        return 1

    args.saida.mkdir(parents=True, exist_ok=True)
    for antiga in args.saida.glob("*.jpg"):
        antiga.unlink()

    aleatorio = random.Random(args.semente)
    gerador = np.random.default_rng(args.semente)
    linhas = []
    for foto in fotos:
        # exif_transpose: o exemplo_livre.jpg guarda a rotação só no EXIF
        # (orientação 6); sem isso as variações sairiam deitadas.
        original = ImageOps.exif_transpose(Image.open(foto)).convert("RGB")
        for i in range(1, args.por_imagem + 1):
            parametros = sortear_parametros(aleatorio)
            nome = f"{foto.stem}_{i:03d}.jpg"
            aplicar(original, parametros, gerador).save(
                args.saida / nome, format="JPEG", quality=parametros["qualidade_jpeg"]
            )
            linhas.append({"arquivo": nome, "origem": foto.name, **parametros})

    with (args.saida / "manifesto.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    lente_suja = sum(1 for linha in linhas if linha["desfoque_raio"] >= 2)
    print(
        f"{len(linhas)} imagens em {args.saida} ({len(fotos)} fotos x {args.por_imagem}; "
        f"{lente_suja} com desfoque forte de lente suja)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
