from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from pathlib import Path

import bentoml
from PIL import Image, UnidentifiedImageError

from . import model
from .logic import ResultadoOcupacao, classificar_ocupacao
from .render import anotar_imagem

_LIMIAR_CONFIANCA = float(os.environ.get("BLU_CONF_THRESHOLD", "0.25"))


class _ImagemInvalida(Exception):
    def __init__(self, detalhe: str):
        self.detalhe = detalhe


class _ModeloIndisponivel(Exception):
    def __init__(self, detalhe: str):
        self.detalhe = detalhe


def _avaliar(imagem: Path) -> tuple[Image.Image, ResultadoOcupacao]:
   
    try:
        imagem_pil = Image.open(imagem)
        imagem_pil.load()
    except (UnidentifiedImageError, OSError) as erro:
        raise _ImagemInvalida(str(erro)) from erro

    try:
        deteccoes = model.prever(imagem_pil)
    except FileNotFoundError as erro:
        raise _ModeloIndisponivel(str(erro)) from erro

    resultado = classificar_ocupacao(deteccoes, _LIMIAR_CONFIANCA)
    return imagem_pil, resultado


@bentoml.service(name="blu_service")
class BluService:
    

    @bentoml.api(route="/prever_ocupacao")
    def prever_ocupacao(self, imagem: Path, ctx: bentoml.Context) -> dict:
        
        try:
            _imagem_pil, resultado = _avaliar(imagem)
        except _ImagemInvalida as erro:
            ctx.response.status_code = 400
            return {"erro": "arquivo_invalido", "detalhe": erro.detalhe}
        except _ModeloIndisponivel as erro:
            ctx.response.status_code = 503
            return {"erro": "modelo_indisponivel", "detalhe": erro.detalhe}

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "modelo": model.nome_modelo(),
            "total_vagas": resultado.total_vagas,
            "livres": resultado.livres,
            "ocupadas": resultado.ocupadas,
            "taxa_ocupacao": resultado.taxa_ocupacao,
            "vagas": [
                {
                    "id": vaga.id,
                    "estado": vaga.estado,
                    "confianca": vaga.confianca,
                    "obb": vaga.obb,
                }
                for vaga in resultado.vagas
            ],
            "vagas_descartadas_baixa_confianca": resultado.vagas_descartadas_baixa_confianca,
        }

    @bentoml.api(route="/prever_ocupacao_imagem")
    def prever_ocupacao_imagem(self, imagem: Path, ctx: bentoml.Context) -> Image.Image:
        
        try:
            imagem_pil, resultado = _avaliar(imagem)
        except (_ImagemInvalida, _ModeloIndisponivel):
            
            ctx.response.status_code = 400
            return Image.new("RGB", (1, 1))

        return Image.open(io.BytesIO(anotar_imagem(imagem_pil, resultado)))

    @bentoml.api(route="/health")
    def health(self) -> dict:
        return {
            "status": "ok",
            "modelo_carregado": model.modelo_carregado(),
            "limiar_confianca": _LIMIAR_CONFIANCA,
        }
