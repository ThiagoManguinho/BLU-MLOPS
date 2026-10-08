from __future__ import annotations

from dataclasses import dataclass, field

CLASSE_PARA_ESTADO = {
    "vaga": "LIVRE",
    "carro": "OCUPADA",
}


@dataclass(frozen=True)
class Deteccao:
    
    classe: str
    confianca: float
    obb: list[list[float]]  


@dataclass(frozen=True)
class Vaga:

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
   
    vagas: list[Vaga] = []
    descartadas = 0

    for deteccao in deteccoes:
        if deteccao.confianca < limiar_confianca:
            descartadas += 1
            continue

        estado = CLASSE_PARA_ESTADO.get(deteccao.classe)
        if estado is None:
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
