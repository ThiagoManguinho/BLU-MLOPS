"""Regra de negócio do BLU: mapeia detecções do modelo para o estado_vaga do
Project Charter.

O modelo foi treinado com duas classes que representam o *estado* de uma
vaga, não dois tipos de objeto a serem cruzados geometricamente:

- ``vaga``  -> a vaga está LIVRE
- ``carro`` -> a vaga está OCUPADA

Cada detecção do modelo já É uma vaga com o seu estado. Não há sobreposição
de caixas a calcular.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CLASSE_PARA_ESTADO = {
    "vaga": "LIVRE",
    "carro": "OCUPADA",
}


@dataclass(frozen=True)
class Deteccao:
    """Uma detecção bruta do modelo, antes do filtro de confiança."""

    classe: str
    confianca: float
    obb: list[list[float]]  # 4 cantos [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


@dataclass(frozen=True)
class Vaga:
    """Uma vaga já classificada, pronta para o contrato de saída da API."""

    id: int
    estado: str
    confianca: float
    obb: list[list[float]]


@dataclass(frozen=True)
class ResultadoOcupacao:
    total_vagas: int
    livres: int
    ocupadas: int
    taxa_ocupacao: float
    vagas: list[Vaga] = field(default_factory=list)
    vagas_descartadas_baixa_confianca: int = 0


def classificar_ocupacao(
    deteccoes: list[Deteccao], limiar_confianca: float
) -> ResultadoOcupacao:
    """Aplica o filtro de confiança e agrega as detecções em um resultado.

    Detecções abaixo de ``limiar_confianca`` não são classificadas como
    LIVRE nem OCUPADA — ficam de fora da contagem e aparecem em
    ``vagas_descartadas_baixa_confianca``, alinhado ao item "fora do escopo"
    do Charter: não estimar estado quando a confiança da detecção é baixa.
    """
    vagas: list[Vaga] = []
    descartadas = 0

    for deteccao in deteccoes:
        if deteccao.confianca < limiar_confianca:
            descartadas += 1
            continue

        estado = CLASSE_PARA_ESTADO.get(deteccao.classe)
        if estado is None:
            # Classe desconhecida (não deveria ocorrer com este modelo) —
            # trata como descarte em vez de quebrar a resposta.
            descartadas += 1
            continue

        vagas.append(
            Vaga(
                id=len(vagas),
                estado=estado,
                confianca=deteccao.confianca,
                obb=deteccao.obb,
            )
        )

    total = len(vagas)
    livres = sum(1 for v in vagas if v.estado == "LIVRE")
    ocupadas = sum(1 for v in vagas if v.estado == "OCUPADA")
    taxa_ocupacao = round(ocupadas / total, 4) if total else 0.0

    return ResultadoOcupacao(
        total_vagas=total,
        livres=livres,
        ocupadas=ocupadas,
        taxa_ocupacao=taxa_ocupacao,
        vagas=vagas,
        vagas_descartadas_baixa_confianca=descartadas,
    )
