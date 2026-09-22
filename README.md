# BLU — Serviço de detecção de ocupação de vagas

Vertente MLOps do projeto BLU: uma API que recebe uma imagem de estacionamento
e devolve o estado de cada vaga (LIVRE ou OCUPADA), servida com BentoML sobre
um modelo YOLOv8-OBB treinado pela equipe.

## 1. O que este serviço apoia

O Project Charter do BLU (Entendimento do Negócio, fase de Otimização)
descreve duas decisões apoiadas por este sistema:

- **Gestor de mobilidade** (foco principal): consulta o padrão de ocupação de
  uma via ou região para decidir se ela precisa de ampliação, reorganização
  de vagas ou fiscalização mais intensa.
- **Motorista** (foco secundário): decide se vale a pena se deslocar até uma
  vaga sinalizada como livre.

Em nenhum dos dois casos o serviço age sozinho. Ele produz o `estado_vaga`
(LIVRE/OCUPADA) por leitura de imagem — o insumo de dados sobre o qual a
decisão humana é tomada, não a decisão em si. Cobrança automática, aplicação
de multa e identificação de motorista/veículo estão fora do escopo.

A unidade de análise é a vaga, não o veículo: o modelo não reconhece placas
nem identifica carros individualmente, apenas classifica o estado de cada
vaga na imagem.

## 2. Do clone à primeira predição

**Pré-requisitos:** [uv](https://docs.astral.sh/uv/) instalado (gerencia o
Python 3.11 e as dependências sozinho — não é preciso instalar Python 3.11 à
parte). Opcionalmente, [just](https://github.com/casey/just) para os atalhos
abaixo.

```bash
git clone <url-do-repositorio>
cd BLU-MLOPS

# uv baixa o Python 3.11 se necessário e instala as dependências fixadas em uv.lock
uv sync
```

Tempo esperado: **~2–4 minutos** na primeira vez (download de torch e
ultralytics), quase instantâneo nas seguintes.

Suba o serviço:

```bash
uv run bentoml serve src.blu_service.service:BluService --port 3000
# ou, com just instalado: just serve
```

O Swagger fica em `http://localhost:3000` assim que o log mostrar
`Service blu_service initialized` (leva alguns segundos após o servidor
subir, enquanto o modelo é carregado). Ele é útil para explorar a API, mas
não substitui os exemplos de `curl` abaixo — no dia da apresentação, o
Swagger pode não subir.

Rode a primeira predição (usa uma imagem de exemplo do repositório):

```bash
curl -X POST http://localhost:3000/prever_ocupacao \
  -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg"
```

Tempo esperado por predição: **~0.3–1 s em CPU**.

## 3. Contrato da API

### `POST /prever_ocupacao`

BentoML expõe APIs de serviço como `POST`; não há rota `GET` neste estilo de
serviço.

**Entrada:** `multipart/form-data` com um campo `imagem` contendo o arquivo
de imagem (jpeg/png).

**Saída (200):**

```json
{
  "timestamp": "2026-09-21T23:41:00.699576+00:00",
  "modelo": "best.pt (yolov8n-obb, ultralytics)",
  "total_vagas": 7,
  "livres": 3,
  "ocupadas": 4,
  "taxa_ocupacao": 0.5714,
  "vagas": [
    {
      "id": 0,
      "estado": "LIVRE",
      "confianca": 0.9281,
      "obb": [[677.47, 206.92], [682.91, 21.94], [211.26, 8.06], [205.82, 193.05]]
    }
  ],
  "vagas_descartadas_baixa_confianca": 0
}
```

- `estado`: `"LIVRE"` ou `"OCUPADA"`, mapeado diretamente das classes do
  modelo (`vaga` → LIVRE, `carro` → OCUPADA — ver seção 5).
- `obb`: os 4 cantos da caixa orientada, em pixels da imagem original.
- `vagas_descartadas_baixa_confianca`: detecções abaixo do limiar de
  confiança (`BLU_CONF_THRESHOLD`, padrão `0.25`), que **não** entram na
  contagem — alinhado à decisão de não estimar estado sob confiança baixa.

**Saída (400) — imagem inválida:**

```json
{"erro": "arquivo_invalido", "detalhe": "cannot identify image file '...'"}
```

### `POST /prever_ocupacao_imagem`

Mesma entrada de `/prever_ocupacao` (`multipart/form-data`, campo `imagem`).
Em vez de JSON, devolve a própria imagem com as vagas desenhadas: contorno
**verde + "LIVRE"** para vagas livres, contorno **vermelho + "OCUPADA"** para
vagas ocupadas, cada rótulo com a confiança do modelo. Usa exatamente o
mesmo `ResultadoOcupacao` do endpoint JSON (mesmo limiar de confiança, mesmos
rótulos), então as duas respostas são sempre consistentes entre si — não é
o `result.plot()` bruto da ultralytics, que mostraria "vaga"/"carro" e
ignoraria o filtro de confiança.

```bash
curl -X POST http://localhost:3000/prever_ocupacao_imagem \
  -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg" \
  -o resultado_anotado.jpg
```

Abra `resultado_anotado.jpg` para ver o resultado.

### `POST /health`

Sem corpo obrigatório (`{}` funciona). Retorna se o serviço está de pé e se
o modelo já foi carregado.

```json
{"status": "ok", "modelo_carregado": true, "limiar_confianca": 0.25}
```

## 4. Três casos de sucesso e um caso de erro

Com o serviço rodando (`uv run bentoml serve ...`), execute (Git Bash, WSL ou
Linux/macOS):

```bash
# Caso 1 — maioria das vagas livres
curl -X POST http://localhost:3000/prever_ocupacao \
  -F "imagem=@samples/exemplo_livre.jpeg;type=image/jpeg"

# Caso 2 — maioria das vagas ocupadas
curl -X POST http://localhost:3000/prever_ocupacao \
  -F "imagem=@samples/exemplo_ocupado.jpeg;type=image/jpeg"

# Caso 3 — vagas livres e ocupadas na mesma imagem
curl -X POST http://localhost:3000/prever_ocupacao \
  -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg"

# Caso de erro — arquivo que não é imagem
curl -i -X POST http://localhost:3000/prever_ocupacao \
  -F "imagem=@samples/arquivo_invalido.txt;type=text/plain"
```

Os três primeiros devolvem `200` com o JSON descrito acima; o quarto devolve
`400` com `{"erro": "arquivo_invalido", ...}`. Todos os quatro foram testados
manualmente contra o serviço real antes desta entrega.

Ou rode tudo de uma vez: `just demo`.

## 5. De onde veio o modelo servido

- Arquitetura: **YOLOv8n-OBB** (caixas orientadas — mais adequado que caixa
  alinhada para vagas fotografadas em ângulo), via
  [ultralytics](https://github.com/ultralytics/ultralytics) `8.4.60`.
- Ponto de partida: pesos pré-treinados `yolov8n-obb.pt` da ultralytics.
- Fine-tuning: pela equipe BLU, em CPU, sobre 245 imagens de uma **maquete**
  de estacionamento (`train-maquete/`), com 2 classes:
  - `0: vaga` → mapeada para o estado **LIVRE**
  - `1: carro` → mapeada para o estado **OCUPADA**

  Ou seja: o modelo já foi treinado para classificar o *estado* da vaga, não
  para detectar dois tipos de objeto a serem cruzados geometricamente. Cada
  detecção é uma vaga com seu estado; o serviço apenas traduz a classe
  predita para o rótulo `estado_vaga` do Charter (`src/blu_service/logic.py`).
- Pesos servidos: `best.pt`, na raiz do repositório, treinado em
  02/06/2026 (metadado do checkpoint).
- Os datasets de treino/validação (`train-maquete/`, `valid-maquete/`) não
  são versionados (ver `.gitignore`) por tamanho; ficam disponíveis
  localmente após o clone original ou sob pedido à equipe.

## 6. Uso de IA

Este repositório — código do serviço (`src/blu_service/`), testes, arquivos
de configuração (`pyproject.toml`, `bentofile.yaml`, `justfile`, CI) e este
README — foi produzido com apoio de assistentes de IA (Claude, via Claude
Code), a partir do modelo `best.pt` e dos datasets já existentes, fornecidos
pela equipe. A equipe revisou e validou manualmente: a semântica das classes
do modelo (`vaga`→LIVRE, `carro`→OCUPADA), o contrato da API, e os testes
end-to-end descritos na seção 4, todos rodados contra o serviço real antes
desta entrega. O modelo em si (`best.pt`) **não** foi gerado por IA generativa
de texto — é um checkpoint de visão computacional treinado pela equipe com
ultralytics/YOLO sobre dados próprios.

## 7. Limitações conhecidas

- **Dataset de maquete, não de via pública real.** O modelo nunca viu uma
  vaga de rua de verdade; a generalização para o ambiente real (iluminação,
  ângulo de câmera, oclusão) não foi validada.
- **Vazamento entre treino e validação.** `valid-maquete/` contém as mesmas
  imagens e rótulos de `train-maquete/`. As métricas de detecção registradas
  no checkpoint (precisão, recall, mAP50, mAP50-95) estão infladas por esse
  vazamento e não devem ser lidas como desempenho real. A equipe optou por
  não retreinar nesta entrega; refazer o split é o próximo passo natural.
- **Nenhum limiar de aceite para produção foi acordado** com o órgão gestor
  (pendência já registrada no Project Charter).
- **LGPD e retenção de imagens em via pública** seguem pendentes de definição
  com o órgão gestor — este serviço processa apenas imagens estáticas
  fornecidas via upload, sem captura, retenção ou identificação de pessoas.

## 8. Licença

Este projeto é distribuído sob a **GNU Affero General Public License v3.0**
(ver [LICENSE](LICENSE)). A escolha não é arbitrária: `best.pt` deriva dos
pesos pré-treinados `yolov8n-obb.pt` da ultralytics, distribuídos sob
AGPL-3.0, e o código que os carrega herda essa obrigação de licenciamento.

## Estrutura do repositório

```
src/blu_service/
  model.py      - carrega best.pt e roda a inferência (ultralytics YOLO)
  logic.py      - regra de negócio: classe → estado_vaga, agregação, limiar
  render.py     - desenha o ResultadoOcupacao sobre a imagem (endpoint de imagem)
  service.py    - endpoints BentoML (/prever_ocupacao, /prever_ocupacao_imagem, /health)
tests/
  test_logic.py  - testes da regra de negócio (rápidos, sem carregar o modelo)
  test_render.py - testes do desenho das anotações (rápidos, sem carregar o modelo)
samples/         - 3 imagens de exemplo + 1 arquivo inválido, para os testes da seção 4
best.pt           - pesos do modelo (versionado; ver seção 5)
bentofile.yaml    - manifesto de build/containerização do BentoML
justfile          - atalhos: setup, serve, demo, lint, test, docker
```

## Diferenciais implementados

- **Container Docker**: `just docker` empacota o Bento e gera a imagem via
  `bentoml build` + `bentoml containerize ... --image-tag blu_service:latest`
  (o `--image-tag` fixa o nome da imagem; sem ele, cada build gera uma tag
  com hash aleatório). Depois, `just docker-run` (ou
  `docker run --rm -p 3000:3000 blu_service:latest`) sobe o container.
  Testado de ponta a ponta.
- Comando único do clone à predição: `uv sync && just serve`.
- Proveniência do modelo documentada (seção 5).
- `ruff` rodando em CI (`.github/workflows/ci.yml`).
- `/health` responde ao status do serviço e do modelo.
