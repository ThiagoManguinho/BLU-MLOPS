# Comandos do projeto BLU. Instale `just` (https://github.com/casey/just) e `uv`
# (https://docs.astral.sh/uv/) antes de usar.

# Instala as dependências fixadas em uv.lock.
setup:
    uv sync

# Sobe o serviço BentoML (Swagger em http://localhost:3000).
serve:
    uv run bentoml serve src.blu_service.service:BluService --port 3000

# Sobe o serviço em segundo plano, roda os 3 curls de exemplo e derruba tudo.
# Requer `curl` no PATH. Em Windows, rode via Git Bash (o mesmo shell usado
# para os comandos deste justfile).
demo:
    uv run bentoml serve src.blu_service.service:BluService --port 3000 &
    sleep 5
    curl -s -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_livre.jpeg"
    echo ""
    curl -s -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_ocupado.jpeg"
    echo ""
    curl -s -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_misto.jpeg"
    echo ""
    curl -s -X POST http://localhost:3000/prever_ocupacao_imagem -F "imagem=@samples/exemplo_misto.jpeg" -o resultado_anotado.jpg
    echo "imagem anotada salva em resultado_anotado.jpg"
    kill %1

# Roda o lint (ruff).
lint:
    uv run ruff check .

# Roda os testes.
test:
    uv run pytest -q

# Gera a imagem de container do serviço (requer Docker instalado).
docker:
    uv run bentoml build
    uv run bentoml containerize blu_service:latest
