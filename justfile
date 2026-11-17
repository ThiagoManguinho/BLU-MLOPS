# Comandos do projeto BLU. Instale `just` (https://github.com/casey/just) e `uv`
# (https://docs.astral.sh/uv/) antes de usar.

# No Windows, roda as receitas via PowerShell por padrão (não depende de ter
# Git Bash no PATH). A receita `demo` é a exceção: ela força bash via shebang
# própria, porque usa sintaxe de shell (`&`, `kill %1`) que só bash entende.
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Saída dos scripts em UTF-8 também no Windows (acentos no terminal).
export PYTHONUTF8 := "1"

# Cada pedido de inferência vira uma linha em logs/inferencias.jsonl.
#
# Sobe o serviço BentoML (Swagger em http://localhost:3000).
serve:
    uv run python -m bentoml serve src.blu_service.service:BluService --port 3000

# Sobe o serviço em segundo plano, roda os 3 curls de exemplo e derruba tudo.
# Requer bash (Git Bash no Windows) e `curl` no PATH.
demo:
    #!/usr/bin/env bash
    uv run python -m bentoml serve src.blu_service.service:BluService --port 3000 &
    sleep 5
    curl -s -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_livre.jpg"
    echo ""
    curl -s -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_ocupado.jpg"
    echo ""
    curl -s -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_misto.jpeg"
    echo ""
    curl -s -X POST http://localhost:3000/prever_ocupacao_imagem -F "imagem=@samples/exemplo_misto.jpeg" -o resultado_anotado.jpg
    echo "imagem anotada salva em resultado_anotado.jpg"
    kill %1

# Roda o lint (ruff).
lint:
    uv run python -m ruff check .

# Roda os testes.
test:
    uv run python -m pytest -q

# Gera a imagem de container do serviço (requer Docker instalado).
# --image-tag fixa a tag da imagem Docker como blu_service:latest; sem isso,
# o bentoml gera uma tag com hash aleatório a cada build.
docker:
    uv run python -m bentoml build
    uv run python -m bentoml containerize blu_service:latest --image-tag blu_service:latest

# Roda o container gerado por `just docker`.
docker-run:
    docker run --rm -p 3000:3000 blu_service:latest

# --- Monitoramento ---------------------------------------------------------

# Exemplos (com o `just serve` rodando):
#   just carga                         100 pedidos com as imagens de samples/
#   just carga --n 300 --degradar      30% das imagens borradas/escuras/estouradas
#
# Gera carga no serviço para ver o log e as métricas evoluírem.
carga *args:
    uv run python scripts/gerar_carga.py {{args}}

# Requer a imagem gerada por `just docker` (rode de novo se o código mudar).
#
# Sobe tudo em contêiner: serviço, Prometheus, Grafana, Alertmanager e Mailpit.
up:
    docker compose up -d

# Sobe o monitoramento e os alertas, para o serviço rodando na máquina (`just serve`).
monitor-up:
    docker compose up -d prometheus grafana alertmanager mailpit

# Derruba os contêineres (o histórico do Prometheus fica no volume).
down:
    docker compose down

# Tira uma foto do /metrics e monta docs/analise_metrics.xlsx.
planilha:
    uv run python scripts/exportar_metrics.py

# Compara o log com a referência: KS, PSI, histogramas e boxplots em analise/.
analisar:
    uv run python scripts/analisar_log.py

# Refaz o conjunto de referência (precisa de train-maquete/, que não é versionado).
referencia *args:
    uv run python scripts/gerar_referencia.py {{args}}
