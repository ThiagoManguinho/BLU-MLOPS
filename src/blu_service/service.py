from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from typing import Annotated

import bentoml
from PIL import Image, UnidentifiedImageError
from pydantic import WrapValidator

from . import model
from .logic import ResultadoOcupacao, classificar_ocupacao
from .render import anotar_imagem

_LIMIAR_CONFIANCA = float(os.environ.get("BLU_CONF_THRESHOLD", "0.25"))


def _abrir_imagem(valor, handler):
    """Converte a falha do BentoML ao abrir o upload como imagem em erro de
    validação. Sem isso, um arquivo que não é imagem vira HTTP 500; como erro
    de validação, o BentoML responde HTTP 400."""

    try:
        return handler(valor)
    except (UnidentifiedImageError, OSError) as erro:
        raise ValueError(
            "arquivo_invalido: o arquivo enviado não é uma imagem válida (jpeg/png)"
        ) from erro


# Entrada dos endpoints: imagem PIL recebida por upload (multipart, campo `imagem`).
ImagemUpload = Annotated[Image.Image, WrapValidator(_abrir_imagem)]


class _ImagemInvalida(Exception):
    def __init__(self, detalhe: str):
        self.detalhe = detalhe


class _ModeloIndisponivel(Exception):
    def __init__(self, detalhe: str):
        self.detalhe = detalhe


def _avaliar(imagem: Image.Image) -> ResultadoOcupacao:

    """Classifica a imagem já recebida pelo BentoML. Levanta as exceções acima
    em caso de erro, para que cada endpoint decida como respondê-las (JSON de
    erro no endpoint de JSON, imagem de erro não faz sentido no endpoint de
    imagem)."""

    # O BentoML já abriu o arquivo como imagem PIL, mas o PIL só decodifica os
    # pixels sob demanda: um JPEG truncado passa pela abertura e só falha aqui.
    try:
        imagem.load()
    except (UnidentifiedImageError, OSError) as erro:
        raise _ImagemInvalida(str(erro)) from erro

    try:
        deteccoes = model.prever(imagem)
    except FileNotFoundError as erro:
        raise _ModeloIndisponivel(str(erro)) from erro

    return classificar_ocupacao(deteccoes, _LIMIAR_CONFIANCA)


@bentoml.service(name="blu_service")
class BluService:

    """Serviço de inferência para classificação de vagas de estacionamento.

    Recebe imagens e usa o modelo YOLOv8-OBB treinado pela equipe BLU para
    identificar cada vaga como `LIVRE` ou `OCUPADA`.
    """
    

    @bentoml.api(route="/prever_ocupacao")
    def prever_ocupacao(self, imagem: ImagemUpload, ctx: bentoml.Context) -> dict:

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
            resultado = _avaliar(imagem)
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
    def prever_ocupacao_imagem(self, imagem: ImagemUpload, ctx: bentoml.Context) -> Image.Image:

        """Classifica a ocupação e devolve uma imagem anotada.

        Recebe uma imagem como `multipart/form-data` no campo `imagem` e
        retorna a imagem original com as vagas detectadas desenhadas sobre
        ela. Vagas livres recebem contorno verde e o rótulo `LIVRE`; vagas
        ocupadas recebem contorno vermelho e o rótulo `OCUPADA`. Cada rótulo
        inclui a confiança do modelo.

        A classificação e o limiar de confiança são os mesmos utilizados por
        `/prever_ocupacao`, permitindo comparar a imagem anotada com a
        resposta JSON. Um arquivo que não é imagem é recusado pelo BentoML
        com HTTP 400 e erro de validação em JSON; imagem corrompida ou modelo
        indisponível retornam HTTP 400 e uma imagem RGB mínima de 1x1 pixel.
        """
        
        try:
            resultado = _avaliar(imagem)
        except (_ImagemInvalida, _ModeloIndisponivel):
            
            ctx.response.status_code = 400
            return Image.new("RGB", (1, 1))

        return Image.open(io.BytesIO(anotar_imagem(imagem, resultado)))
