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

    """Carrega e classifica a imagem. Levanta as exceções acima em caso de
    erro, para que cada endpoint decida como respondê-las (JSON de erro no
    endpoint de JSON, imagem de erro não faz sentido no endpoint de imagem)."""
   
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

    """Serviço de inferência para classificação de vagas de estacionamento.

    Recebe imagens e usa o modelo YOLOv8-OBB treinado pela equipe BLU para
    identificar cada vaga como `LIVRE` ou `OCUPADA`.
    """
    

    @bentoml.api(route="/prever_ocupacao")
    def prever_ocupacao(self, imagem: Path, ctx: bentoml.Context) -> dict:

        """Classifica a ocupação das vagas e retorna os resultados em JSON.

        Para cada vaga detectada, a resposta informa o estado
        (`LIVRE` ou `OCUPADA`), a confiança e os quatro pontos da caixa
        orientada (`obb`). Também são retornados o total de vagas, as
        quantidades livres e ocupadas, a taxa de ocupação e as detecções
        descartadas por baixa confiança.

        Retorna HTTP 400 quando o arquivo não é uma imagem válida e HTTP 503
        quando o arquivo do modelo não está disponível.
        """
        
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

        """Classifica a ocupação e devolve uma imagem anotada.

        Recebe uma imagem como `multipart/form-data` no campo `imagem` e
        retorna a imagem original com as vagas detectadas desenhadas sobre
        ela. Vagas livres recebem contorno verde e o rótulo `LIVRE`; vagas
        ocupadas recebem contorno vermelho e o rótulo `OCUPADA`. Cada rótulo
        inclui a confiança do modelo.

        A classificação e o limiar de confiança são os mesmos utilizados por
        `/prever_ocupacao`, permitindo comparar a imagem anotada com a
        resposta JSON. Em caso de arquivo inválido ou modelo indisponível,
        retorna HTTP 400 e uma imagem RGB mínima de 1x1 pixel.
        """
        
        try:
            imagem_pil, resultado = _avaliar(imagem)
        except (_ImagemInvalida, _ModeloIndisponivel):
            
            ctx.response.status_code = 400
            return Image.new("RGB", (1, 1))

        return Image.open(io.BytesIO(anotar_imagem(imagem_pil, resultado)))
