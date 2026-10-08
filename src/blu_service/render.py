from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from .logic import ResultadoOcupacao

_COR_LIVRE = (46, 204, 113)  # verde
_COR_OCUPADA = (231, 76, 60)  # vermelho
_ESPESSURA_LINHA = 4


def _fonte(tamanho: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", tamanho)
    except OSError:
        return ImageFont.load_default()


def anotar_imagem(imagem: Image.Image, resultado: ResultadoOcupacao) -> bytes:
    
    imagem_anotada = imagem.convert("RGB").copy()
    desenho = ImageDraw.Draw(imagem_anotada)
    fonte = _fonte(max(16, imagem_anotada.width // 60))

    for vaga in resultado.vagas:
        cor = _COR_LIVRE if vaga.estado == "LIVRE" else _COR_OCUPADA
        pontos = [tuple(ponto) for ponto in vaga.obb]
        desenho.polygon(pontos, outline=cor, width=_ESPESSURA_LINHA)

        rotulo = f"{vaga.estado} {vaga.confianca:.2f}"
        x, y = pontos[0]
        caixa_texto = desenho.textbbox((x, y), rotulo, font=fonte)
        desenho.rectangle(caixa_texto, fill=cor)
        desenho.text((x, y), rotulo, fill=(255, 255, 255), font=fonte)

    buffer = io.BytesIO()
    imagem_anotada.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
