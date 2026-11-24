# BLU — Serviço de detecção de ocupação de vagas

Vertente MLOps do projeto BLU: uma API que recebe uma imagem de estacionamento
e devolve o estado de cada vaga (LIVRE ou OCUPADA), servida com BentoML sobre
um modelo YOLOv8-OBB treinado pela equipe.

## Visão geral: arquitetura e fluxo

```mermaid
flowchart LR
    subgraph entrada["1. Quem envia imagens"]
        direction TB
        cliente["curl / Swagger<br/>(seções 3 e 4)"]
        carga["just carga<br/>pedidos de teste"]
    end

    subgraph servico["2. Serviço BentoML: contêiner blu, porta 3000"]
        direction TB
        middleware["MiddlewareRegistro<br/>request_id + latência"]
        endpoints["Endpoints de inferência<br/>prever_ocupacao<br/>prever_ocupacao_imagem"]
        modelo["Modelo YOLOv8n-OBB<br/>best.pt"]
        metrics["GET /metrics<br/>bentoml_* e blu_*"]
        middleware --> endpoints
        endpoints --> modelo
        endpoints -->|métricas blu_*| metrics
    end

    subgraph monitor["3. Monitoramento e alertas: contêineres"]
        direction TB
        prometheus["Prometheus, porta 9090<br/>coleta a cada 5 s, guarda 15 dias<br/>avalia as regras de alerta"]
        grafana["Grafana, porta 3001<br/>painéis Monitoramento e Estatísticas"]
        alertmanager["Alertmanager, porta 9093<br/>agrupa e envia"]
        mailpit["E-mail<br/>Mailpit, porta 8025, ou SMTP real"]
        drift["Exportador de drift, porta 9101<br/>KS, PSI e percentis exatos"]
        prometheus -->|consultado pelo| grafana
        drift -->|lido a cada 10 s| prometheus
        prometheus -->|alerta de câmera| alertmanager
        alertmanager -->|e-mail| mailpit
    end

    subgraph arquivos["4. Arquivos e análise: sua máquina"]
        direction TB
        ref[("referencia/<br/>245 fotos de treino")]
        log[("logs/inferencias.jsonl<br/>1 linha por pedido")]
        analisar["just analisar<br/>KS, PSI, histogramas, boxplots"]
        planilha["just planilha<br/>docs/analise_metrics.xlsx"]
        ref --> analisar
        log --> analisar
    end

    cliente -->|POST com a imagem| middleware
    carga -->|POST com a imagem| middleware
    middleware -->|grava a linha do pedido| log
    log -->|últimos 100 pedidos| drift
    metrics -->|lido a cada 5 s| prometheus
    metrics -->|foto do momento| planilha
```

### O caminho de um pedido

1. O cliente (curl, Swagger ou `just carga`) envia a foto em `POST
   /prever_ocupacao` ou `/prever_ocupacao_imagem` (`multipart/form-data`,
   campo `imagem`) para a porta 3000.
2. O **`MiddlewareRegistro`** (`src/blu_service/monitoramento.py`) recebe o
   pedido primeiro: cria um `request_id` e começa a medir o tempo.
3. O **BentoML** abre o arquivo como imagem PIL. Se não for imagem, responde
   **400** ali mesmo; o pedido ainda assim vai para o log (passo 7).
4. O **endpoint** (`src/blu_service/service.py`) faz a análise:
   - mede a imagem: luminância, contraste, nitidez e histograma;
   - roda o **modelo** (`model.py`): caixas `vaga` e `carro`;
   - aplica a **regra de negócio** (`logic.py`): `vaga` vira LIVRE e `carro`
     vira OCUPADA, com contagem e taxa de ocupação;
   - resume a saída (vagas, confiança) e marca a imagem como suspeita se
     estiver borrada, escura, estourada ou sem as 7 vagas;
   - atualiza as métricas `blu_*` do `/metrics`.
5. A **resposta** volta ao cliente: JSON ou imagem anotada (`render.py`), com o
   cabeçalho `X-Request-ID`.
6. O **BentoML** atualiza as próprias métricas no `/metrics`: contagem por
   código HTTP e histograma de latência.
7. O **middleware** grava **uma linha** em `logs/inferencias.jsonl` com tudo o
   que entrou e saiu, o status e a latência total.
8. A cada 5 s, o **Prometheus** lê o `/metrics` e guarda o histórico. O
   **Grafana** consulta o Prometheus e atualiza o painel.
9. O **Prometheus** também avalia as regras de alerta
   (`monitoramento/alertas.yml`). Se mais de 30% das imagens dos últimos 5
   minutos estiverem borradas, escuras ou estouradas, o que indica um provável
   problema na câmera, ele avisa o **Alertmanager**, que manda um **e-mail**
   para a equipe.

Depois, fora do caminho do pedido:
- `just analisar` compara o log com a referência (fotos de treino) por KS e
  PSI;
- `just planilha` transforma uma foto do `/metrics` em histograma, boxplot e
  taxa de erros.

### Onde cada informação fica

| Onde | O que guarda | Para que serve | Por quanto tempo |
|---|---|---|---|
| `logs/inferencias.jsonl` | cada pedido, com o que entrou e saiu | investigar um pedido; histogramas, boxplots, KS/PSI | até 6 arquivos de 10 MB |
| `GET /metrics` | totais desde que o serviço subiu | foto do momento; planilha | até o serviço reiniciar |
| Prometheus | o `/metrics` a cada 5 s | evolução no tempo, taxas por minuto | 15 dias |
| Grafana | nada (consulta o Prometheus) | painel ao vivo | — |
| Alertmanager | alertas ativos e silêncios | avisar a equipe por e-mail | enquanto o contêiner existir |
| Mailpit | e-mails de alerta (caixa local) | ver os e-mails sem conta de e-mail | até o contêiner ser recriado |

### Dois modos de rodar

| | Tudo em contêiner (recomendado) | Serviço na máquina |
|---|---|---|
| Subir | `just docker` (1ª vez) + `just up` | `just serve` + `just monitor-up` |
| Serviço | contêiner `blu`, porta 3000 | processo local, porta 3000 |
| Prometheus, Grafana, Alertmanager, Mailpit, exportador de drift | contêineres, portas 9090, 3001, 9093, 8025 e 9101 | contêineres, portas 9090, 3001, 9093, 8025 e 9101 |
| Log | `./logs/inferencias.jsonl` (pasta montada) | `./logs/inferencias.jsonl` |
| Parar | `just down` | Ctrl+C no `serve` + `just down` |

Os dois usam a porta 3000, então não rode os dois ao mesmo tempo. O passo a
passo completo do monitoramento está na [seção 6](#6-monitoramento-log-em-arquivo-metrics-prometheus-e-grafana),
os alertas por e-mail na [seção 6.6](#66-alertas-por-e-mail-problema-na-câmera),
e a lista de arquivos, em [Estrutura do repositório](#estrutura-do-repositório).

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

**Pré-requisitos:** `git` e **Python 3.11 ou superior** já instalado na
máquina. 

Clone o repositório e instale o `uv` pelo
[requirements.txt](requirements.txt):

```bash
git clone <url-do-repositorio>
cd BLU-MLOPS

# Linux / macOS
python3 -m pip install --user -r requirements.txt

# Windows (PowerShell)
python -m pip install -r requirements.txt
```

Confira se ficou no PATH:

```bash
uv --version
```

> Se aparecer "comando não encontrado" ou "não é reconhecido", **feche e
> reabra o terminal**. No Linux, se continuar, adicione `~/.local/bin` ao PATH
> com `export PATH="$HOME/.local/bin:$PATH"`.
>
> Em Debian 12+/Ubuntu 23.04+ o pip pode recusar com
> `externally-managed-environment`. Nesse caso use
> `python3 -m pip install --user --break-system-packages -r requirements.txt`.
> O comando instala só para o seu usuário e não mexe nos pacotes do sistema.
> Outra opção é o instalador oficial, que dispensa o Python:
> `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Instale as dependências do projeto:

```bash
uv sync
```

Esse único comando cria o ambiente virtual em **`.venv/`** (usando o Python que
já está na sua máquina) e instala tudo que está fixado no `uv.lock` — incluindo
o [just](https://github.com/casey/just), via o pacote `rust-just`. **Não é
preciso criar a venv antes**: o `uv sync` faz isso por você, e criar uma à mão
pode acabar apontando para uma versão de Python diferente da que o projeto
espera.

Tempo esperado: **~2–5 minutos** na primeira vez, quase instantâneo nas
seguintes. O volume vem das dependências de visão computacional — o ambiente
final ocupa cerca de **935 MB**, sendo `torch` (~450 MB), o runtime do `polars`
(~170 MB) e o `opencv` (~110 MB) os três maiores.

### Ativar a venv (opcional, mas recomendado)

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Você saberá que funcionou porque o terminal passa a mostrar `(blu-service)` no
início da linha. Para sair, basta `deactivate`.

### Duas formas de rodar os comandos

Tudo neste README pode ser feito de duas maneiras equivalentes — escolha a que
preferir:

| | Sem ativar a venv | Com a venv ativada |
|---|---|---|
| Subir o serviço | `uv run just serve` | `just serve` |
| Rodar os testes | `uv run just test` | `just test` |
| Lint | `uv run just lint` | `just lint` |
| Ver todas as receitas | `uv run just --list` | `just --list` |

O prefixo `uv run` significa "execute isto dentro do `.venv` do projeto". Se a
venv já está ativada, o prefixo é dispensável — é exatamente o mesmo comando.

E o `just` é sempre apenas um atalho: cada receita do
[justfile](justfile) é um comando `uv` por baixo. Por exemplo, `just serve`
executa:

```bash
uv run python -m bentoml serve src.blu_service.service:BluService --port 3000
```

Ou seja, você nunca depende do `just` — se preferir, pode abrir o `justfile` e
copiar o comando direto. Os exemplos a seguir mostram as duas formas.

Suba o serviço:

```bash
# pelo atalho
uv run just serve

# ou, o comando completo por trás dele
uv run python -m bentoml serve src.blu_service.service:BluService --port 3000
```


O Swagger fica em `http://localhost:3000` assim que o log mostrar
`Service blu_service initialized` (leva alguns segundos após o servidor
subir, enquanto o modelo é carregado). Ele é útil para explorar a API. 

Rode a primeira predição (usa uma imagem de exemplo do repositório):

```powershell
# Windows (PowerShell) 
curl.exe -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg"
```

```bash
# Linux / macOS / Git Bash / WSL
curl -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg"
```

Tempo esperado por predição: **~0.3–1 s em CPU**.

## 3. Contrato da API

### `POST /prever_ocupacao`

BentoML expõe APIs de serviço como `POST`; não há rota `GET` neste estilo de
serviço.

**Entrada:** `multipart/form-data` com um campo `imagem` contendo o arquivo
de imagem (jpeg/png). No serviço, o parâmetro é tipado como imagem PIL
(`PIL.Image.Image`): o BentoML recebe o upload e entrega ao endpoint a imagem
já aberta, sem passar por caminho de arquivo.

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

**Saída (400) — arquivo que não é imagem:** o BentoML recusa o upload ao
tentar abri-lo como imagem e responde com o seu formato padrão de erro de
validação:

```json
{
  "error": "1 validation error for Input",
  "detail": [
    {
      "type": "value_error",
      "loc": ["imagem"],
      "msg": "Value error, arquivo_invalido: o arquivo enviado não é uma imagem válida (jpeg/png)"
    }
  ]
}
```

**Saída (400) — imagem corrompida** (por exemplo, um JPEG truncado, que abre
mas não decodifica):

```json
{"erro": "arquivo_invalido", "detalhe": "image file is truncated (5 bytes not processed)"}
```

**Saída (503) — modelo indisponível:** `{"erro": "modelo_indisponivel", ...}`.

### `POST /prever_ocupacao_imagem`

Mesma entrada de `/prever_ocupacao` (`multipart/form-data`, campo `imagem`).
Em vez de JSON, devolve a própria imagem com as vagas desenhadas: contorno
**verde + "LIVRE"** para vagas livres, contorno **vermelho + "OCUPADA"** para
vagas ocupadas, cada rótulo com a confiança do modelo. Usa exatamente o
mesmo `ResultadoOcupacao` do endpoint JSON (mesmo limiar de confiança, mesmos
rótulos), então as duas respostas são sempre consistentes entre si — não é
o `result.plot()` bruto da ultralytics, que mostraria "vaga"/"carro" e
ignoraria o filtro de confiança.

```powershell
# Windows (PowerShell)
curl.exe -X POST http://localhost:3000/prever_ocupacao_imagem -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg" -o resultado_anotado.jpg
```

```bash
# Linux / macOS / Git Bash / WSL
curl -X POST http://localhost:3000/prever_ocupacao_imagem -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg" -o resultado_anotado.jpg
```

Abra `resultado_anotado.jpg` para ver o resultado.

## 4. Três casos de sucesso e um caso de erro

Com o serviço rodando, execute os comandos abaixo a partir da raiz do
repositório.

As três fotos de exemplo têm o mesmo enquadramento: a maquete inteira, vista
de cima, como a câmera fixa de produção veria. Assim, a diferença entre os
casos fica só na ocupação das vagas, e não no ângulo da foto.

### Windows (PowerShell)

```powershell
# Caso 1 — todas as vagas livres
curl.exe -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_livre.jpg;type=image/jpeg"

# Caso 2 — todas as vagas ocupadas
curl.exe -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_ocupado.jpg;type=image/jpeg"

# Caso 3 — vagas livres e ocupadas na mesma imagem
curl.exe -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg"

# Caso de erro — arquivo que não é imagem (-i mostra o status HTTP)
curl.exe -i -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/arquivo_invalido.txt;type=text/plain"
```

> Use **`curl.exe`**, não `curl`. No PowerShell, `curl` é apelido de
> `Invoke-WebRequest`, que não entende a flag `-F` e falha.

### Linux / macOS / Git Bash / WSL

```bash
# Caso 1 — todas as vagas livres
curl -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_livre.jpg;type=image/jpeg"

# Caso 2 — todas as vagas ocupadas
curl -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_ocupado.jpg;type=image/jpeg"

# Caso 3 — vagas livres e ocupadas na mesma imagem
curl -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/exemplo_misto.jpeg;type=image/jpeg"

# Caso de erro — arquivo que não é imagem (-i mostra o status HTTP)
curl -i -X POST http://localhost:3000/prever_ocupacao -F "imagem=@samples/arquivo_invalido.txt;type=text/plain"
```

Os três primeiros devolvem `200` com o JSON descrito acima; o quarto devolve
`400` com o erro de validação `arquivo_invalido` (ver seção 3). Todos os
quatro foram testados manualmente contra o serviço real antes desta entrega,
com este resultado:

| Imagem | `total_vagas` | `livres` | `ocupadas` | Confiança |
|---|---:|---:|---:|---|
| `exemplo_livre.jpg` | 7 | 7 | 0 | 0,89 a 0,95 |
| `exemplo_ocupado.jpg` | 7 | 0 | 7 | 0,87 a 0,91 |
| `exemplo_misto.jpeg` | 7 | 3 | 4 | 0,80 a 0,93 |

O carro parado na pista, fora das vagas, não é contado. A vaga com o pedaço de
papelão sai como ocupada, porque está bloqueada.

Ou rode tudo de uma vez: `uv run just demo` (ou `just demo`, com a venv
ativada).

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
- Os datasets de treino/validação (`train-maquete/`, `valid-maquete/`) não
  são versionados (ver `.gitignore`) por tamanho; ficam disponíveis
  localmente após o clone original ou sob pedido à equipe.


## 5.1. Problema conhecido no Windows: Smart App Control

Se ao subir o serviço aparecer:

```
DLL load failed while importing _C: Uma política de Controle de Aplicativo
bloqueou este arquivo.
```

o **Smart App Control** do Windows está bloqueando as DLLs do PyTorch, que não
são assinadas digitalmente. Para confirmar:

```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy").VerifiedAndReputablePolicyState
# 0 = desligado | 1 = ligado (bloqueia) | 2 = modo avaliação
```

Não é um problema do projeto: qualquer `import torch` nessa máquina falha do
mesmo jeito. Há dois caminhos:

1. **Rodar pelo container** (recomendado). Dentro do container é Linux, e a
   política não se aplica. O `bentoml build` funciona mesmo com o torch
   bloqueado, porque `src/blu_service/model.py` envolve o import da
   ultralytics em `bentoml.importing()` — isso marca o torch como dependência
   de *runtime*, necessária para atender requisições, mas não para o build
   inspecionar a API.
2. **Desligar o Smart App Control** (Windows Security → Controle de aplicativos
   e navegador → Smart App Control → Desativado) e reiniciar. ⚠️ É
   irreversível: só dá para religar reinstalando o Windows.


## 6. Monitoramento: log em arquivo, `/metrics`, Prometheus e Grafana

O serviço registra o que recebe e o que devolve de três formas:

- **Log em arquivo:** uma linha por pedido em `logs/inferencias.jsonl`. É
  daqui que saem histogramas, boxplots e a comparação com a referência
  (KS/PSI).
- **`GET /metrics`:** o endpoint do próprio BentoML, no formato Prometheus,
  com as métricas de imagem e de resposta do modelo (`blu_*`) acrescentadas.
  Como analisá-lo está em [`docs/metrics.md`](docs/metrics.md), com a
  planilha [`docs/analise_metrics.xlsx`](docs/analise_metrics.xlsx).
- **Prometheus + Grafana:** guardam o histórico do `/metrics` e mostram um
  painel ao vivo.
- **Alertmanager:** manda um e-mail quando a qualidade das imagens indica
  problema na câmera (seção 6.6).

### 6.1. Passo a passo: gerar carga e ver o rastro evoluir

Tudo roda em contêiner: o serviço, o Prometheus e o Grafana. O log em arquivo
continua aparecendo em `logs/`, na pasta do repositório, porque essa pasta é
montada dentro do contêiner do serviço. Use dois terminais na raiz do
repositório.

```bash
# Terminal 1: constrói a imagem do serviço (só na primeira vez, ou se o código
# mudar; leva alguns minutos) e sobe serviço + Prometheus + Grafana
uv run just docker
uv run just up
```

```powershell
# Terminal 2: acompanha o log enquanto ele cresce
Get-Content logs\inferencias.jsonl -Wait -Tail 3     # Windows (PowerShell)
tail -f logs/inferencias.jsonl                       # Linux / macOS / Git Bash
```

```bash
# Terminal 1: gera a carga
uv run just carga --n 200 --degradar
```

> **Sem Docker para o serviço:** rode `uv run just serve` no lugar de
> `just docker` e `just up`, e `uv run just monitor-up` para subir só o
> Prometheus e o Grafana. O resto do passo a passo é igual. Não rode os dois
> modos juntos: os dois usam a porta 3000.

O `just carga` envia 200 pedidos aos dois endpoints com as imagens de
`samples/`. Ele mistura de propósito:
- 10% de arquivos inválidos, que geram respostas 400;
- com `--degradar`, 30% de imagens borradas, escurecidas ou estouradas,
  degradadas no próprio cliente.

Opções úteis:

| Opção | Efeito |
|---|---|
| `--n` | quantidade de pedidos |
| `--intervalo` | segundos entre pedidos |
| `--fracao-erro` | fração de arquivos inválidos |
| `--degradar 0.5` | fração de imagens degradadas |
| `--tipo borrada` | só um tipo de degradação (`borrada`, `escura` ou `estourada`), para simular um defeito específico da câmera e disparar o alerta (seção 6.6) |
| `--pasta DIR` | outra pasta de imagens |

Enquanto a carga roda:

| Onde | O que ver |
|---|---|
| Terminal 2 | uma linha nova por pedido, inclusive os 400 |
| http://localhost:3000/metrics | os contadores `bentoml_*` e `blu_*` subindo |
| http://localhost:3001 | painel **BLU - Monitoramento** no Grafana (sem login; admin/admin para editar) e, pelo link no topo, o **BLU - Estatísticas (AV2)** |
| http://localhost:9090/targets | o Prometheus coletando o serviço (`host.docker.internal:3000`, estado `UP`) |
| http://localhost:9090/alerts | as regras de alerta de câmera e se alguma está disparando (seção 6.6) |
| http://localhost:8025 | a caixa de e-mail local (Mailpit), onde chegam os e-mails de alerta |

Depois da carga:

```bash
uv run just planilha     # foto do /metrics -> docs/analise_metrics.xlsx (histograma, boxplot, taxa de erros)
uv run just analisar     # log x referência: KS, PSI, histogramas e boxplots em analise/
uv run just down         # derruba os contêineres
```

Sem o `just`, os comandos equivalentes estão no [justfile](justfile), por
exemplo `uv run python scripts/gerar_carga.py --n 200 --degradar` e
`docker compose up -d`.

A carga, a planilha e a análise rodam na máquina e falam com o serviço pela
porta 3000. Funcionam igual com o serviço em contêiner ou fora dele.

![Painel do Grafana durante uma carga com imagens degradadas](docs/img/grafana.png)

### 6.2. O log em arquivo

- **Arquivo:** `logs/inferencias.jsonl`, no formato
  [JSON Lines](https://jsonlines.org/), com um pedido por linha. Lê-se direto
  no pandas com `pd.read_json(caminho, lines=True)`.
- **Append-only:** o `RotatingFileHandler` do `logging` só acrescenta linhas.
  Quando o arquivo passa de 10 MB, ele vira `inferencias.jsonl.1` (o `.1`
  anterior vira `.2`, até `.5`) e um arquivo novo começa. Nenhuma linha é
  reescrita. Os limites são configuráveis por `BLU_LOG_MAX_BYTES` e
  `BLU_LOG_BACKUPS`.
- **Um evento por pedido, inclusive os recusados:** o registro é feito por um
  middleware que envolve o app inteiro (`MiddlewareRegistro` em
  `src/blu_service/monitoramento.py`). Por isso entram no log até os pedidos
  que o BentoML recusa antes de chegar ao endpoint, como um arquivo que não é
  imagem. Rotas de infraestrutura (`/metrics`, `/readyz`, `/docs`…) ficam de
  fora.
- **Data e hora:** `ts` em ISO 8601 com **microssegundos e fuso horário**,
  mais `ts_unix` (segundos desde 1970, com 6 casas). O `request_id` da linha
  também volta ao cliente no cabeçalho `X-Request-ID`, para cruzar um pedido
  com sua linha.

Linha real (o histograma e a lista de detecções foram encurtados aqui):

```json
{
  "ts": "2026-11-15T21:27:37.981221-03:00", "ts_unix": 1794788857.981221,
  "versao_log": 1, "request_id": "4fe23922b725484bbef90fe9ee89a760", "pid": 17764,
  "metodo": "POST", "endpoint": "/prever_ocupacao", "status_http": 200,
  "latencia_total_ms": 4153.706, "latencia_modelo_ms": 4010.648,
  "modelo": "best.pt (yolov8n-obb, ultralytics)", "limiar_confianca": 0.25,
  "entrada": {
    "content_type": "multipart/form-data", "content_length": 111402,
    "formato": "JPEG", "modo": "RGB", "largura": 1310, "altura": 1600, "megapixels": 2.096,
    "luminancia_media": 170.9446, "luminancia_desvio": 63.306,
    "rgb_media": [175.2583, 168.621, 172.2986],
    "fracao_estourada": 0.000323, "fracao_escura": 0.003895, "nitidez": 320.2797,
    "histograma_l": [689, 743, 558, 763, "... 256 valores"]
  },
  "saida": {
    "total_vagas": 7, "livres": 3, "ocupadas": 4, "taxa_ocupacao": 0.5714,
    "descartadas_baixa_confianca": 0,
    "confianca_media": 0.8823, "confianca_min": 0.8047, "confianca_max": 0.9281,
    "confianca_media_livre": 0.8781, "confianca_media_ocupada": 0.8856,
    "vagas_esperadas": 7, "vagas_divergentes": false,
    "deteccoes": [{"estado": "LIVRE", "confianca": 0.9281, "area_rel": 0.041662, "cx_rel": 0.3392, "cy_rel": 0.0672}, "..."]
  },
  "alertas": [],
  "erro": null
}
```

Este foi o primeiro pedido depois que o serviço subiu, por isso a latência de
4 s: o modelo é carregado na primeira chamada. Na carga de exemplo, a maioria
dos pedidos seguintes ficou entre 0,1 e 0,5 s (ver `docs/metrics.md`).

Num pedido recusado (arquivo que não é imagem), `entrada` só tem
`content_type` e `content_length`, `saida` é `null` e `erro` traz o motivo:

```json
{"ts": "2026-11-15T21:09:15.032583-03:00", "...": "...", "endpoint": "/prever_ocupacao", "status_http": 400,
 "latencia_total_ms": 41.201, "latencia_modelo_ms": null, "entrada": {"content_type": "multipart/form-data", "content_length": 229},
 "saida": null, "erro": {"codigo": 400, "detalhe": {"error": "1 validation error for Input", "detail": [{"loc": ["imagem"], "msg": "Value error, arquivo_invalido: ..."}]}}}
```

#### Por que estas variáveis (CRISP-DM)

O serviço apoia a decisão do gestor de mobilidade e do motorista (seção 1)
com o estado de cada vaga. Essa informação só vale se a **imagem** que chega
se parece com a que o modelo viu no treino e se a **resposta** se comporta
como esperado. As variáveis cobrem os três pontos em que isso pode falhar.

| Grupo | Variáveis | Para que servem |
|---|---|---|
| **Entrada: iluminação** | `luminancia_media`, `luminancia_desvio` (contraste), `rgb_media`, `fracao_estourada` (pixels ≥ 250), `fracao_escura` (≤ 5), `histograma_l` (256 faixas) | A maquete é fotografada sob luz variável (sombra, contraluz, lâmpada). O `histograma_l` e a `luminancia_media` são as mesmas variáveis dos exemplos de KS (003) e PSI (008) da disciplina. |
| **Entrada: qualidade** | `nitidez` (ver 6.3), `content_length`, `largura`, `altura`, `megapixels`, `formato` | Câmera fora de foco, lente suja ou resolução trocada mudam o que o modelo vê. `content_length ÷ pixels` (bytes por pixel) denuncia compressão JPEG excessiva. |
| **Saída: resposta do modelo** | `total_vagas`, `livres`, `ocupadas`, `taxa_ocupacao`, `confianca_*`, `descartadas_baixa_confianca`, `vagas_divergentes`, `deteccoes` (estado, confiança, área e centro relativos de cada caixa) | Distribuição da ocupação (a informação de negócio) e da confiança (queda de confiança costuma vir antes do erro). A maquete tem 7 vagas fixas: um total diferente indica câmera deslocada, vaga encoberta ou falha do modelo. O centro e a área de cada caixa permitem detectar a câmera mexida. |
| **Operação** | `status_http`, `erro`, `latencia_total_ms`, `latencia_modelo_ms`, `endpoint`, `request_id`, `modelo`, `limiar_confianca`, `versao_log` | Erros e tempo de resposta por pedido, com a configuração em vigor (modelo e limiar), para comparar períodos. |
| **Diagnóstico** | `alertas` | Motivos pelos quais a imagem saiu da faixa da referência (`borrada`, `escura`, `estourada`, `vagas_divergentes`). |

### 6.3. Métrica de qualidade de imagem: nitidez

Além da iluminação, o serviço acompanha a **nitidez**: a variância do
Laplaciano da imagem em cinza, medida numa cópia com 640 px de largura para
não depender da resolução.

Por que esta métrica:
- a câmera da maquete é fixa e de foco fixo;
- o modelo depende das **linhas finas** que delimitam as vagas;
- foto fora de foco, lente suja ou câmera tremida apagam essas linhas, e a
  nitidez cai **antes** de o modelo começar a errar.

Medido na imagem `samples/exemplo_livre.jpg`:

| Imagem | Nitidez | Luminância |
|---|---:|---:|
| original | 287,0 | 206,4 |
| com desfoque gaussiano raio 2 | 34,6 | 206,4 |
| com desfoque gaussiano raio 4 | 6,7 | 206,3 |
| escurecida para 45% | 58,8 | 92,3 |
| clareada 1,8x | 227,8 | 249,3 (95% dos pixels estourados) |
| JPEG com qualidade 5 | 282,9 | 204,4 |

O desfoque de raio 2 ainda fica acima do limite de `borrada` (15) nesta foto,
que é bem nítida; o de raio 4, usado pelo `just carga --degradar`, fica abaixo.
A foto também é mais clara que todas as do treino (luminância 206, contra no
máximo 189), mas isso não gera alerta: as regras olham só imagem escura e
pixels estourados.

Uma imagem é marcada como suspeita quando sai da faixa das 245 fotos de
treino (`referencia/resumo.json`). Os limites são configuráveis pelo `.env`:

| Motivo | Regra | Origem do limite |
|---|---|---|
| `borrada` | nitidez < 15 | p1 da referência (14,5) |
| `escura` | luminância média < 100 | p1 da referência (106,5) |
| `estourada` | mais de 1% dos pixels ≥ 250 | máximo da referência: 0,07% |
| `vagas_divergentes` | vagas detectadas ≠ 7 | regra da maquete |

**Limitação:** compressão JPEG extrema **não** derruba a nitidez, porque os
blocos da compressão criam bordas novas (última linha da tabela). Esse caso
aparece nos bytes por pixel, que também vão para o log.

### 6.4. Conjunto de referência e drift (KS/PSI)

O enunciado sugere comparar o que chega ao serviço com o conjunto de
treinamento. Por isso `referencia/referencia.jsonl` passa as **245 fotos de
treino** (`train-maquete/images`) pelas mesmas funções do log e pelo modelo.
O arquivo tem o mesmo formato do log, e `referencia/resumo.json` traz os
percentis de cada variável. Como os datasets não são versionados, os dois
arquivos são; para refazê-los, use `just referencia`.

O que a referência revelou:
- **As fotos de treino são de celular, de perto:** só 67 das 245 (27%) mostram
  as 7 vagas. A regra `vagas_divergentes` vale para a câmera fixa de produção,
  não para o treino. Quando houver fotos do ESP32-CAM, a referência deve ser
  refeita com elas (`just referencia --pasta <fotos>`). É a "referência
  estabelecida manualmente" que o enunciado prevê.
- **As 25 imagens de `valid-maquete/` são as mesmas de `train-maquete/`**
  (mesmo nome e tamanho). A validação do modelo não mediu generalização.
- **`descartadas_baixa_confianca` é sempre 0:** a ultralytics já corta as
  detecções abaixo de 0,25 por padrão, igual ao limiar do serviço.

O `just analisar` compara o log com a referência, variável a variável:
- **KS:** `scipy.stats.ks_2samp`.
- **PSI:** 10 faixas pelos quantis da referência, a mesma construção do
  exemplo 008. Abaixo de 0,1 é estável; de 0,1 a 0,25, mudança moderada;
  acima de 0,25, drift.
- **Gráficos:** histograma e boxplot (referência × produção) em `analise/`.

Como checagem do cálculo, comparar metade da referência com a outra metade dá
PSI entre 0,03 e 0,16.

> Com a carga padrão, o PSI aponta drift em quase tudo. É esperado: as 3
> imagens de `samples/` repetidas não representam a distribuição das 245
> fotos de treino. Para uma carga "sem drift", use
> `just carga --pasta train-maquete/images`.

### 6.5. `/metrics`, Prometheus e Grafana

O `GET /metrics` é o do próprio BentoML, no formato Prometheus.
- **Séries do BentoML:** pedidos por endpoint e código HTTP, e o histograma
  da latência.
- **Séries do projeto (`blu_*`):** nitidez, luminância, vagas detectadas,
  confiança, ocupação e imagens suspeitas por motivo.

A explicação completa está em [`docs/metrics.md`](docs/metrics.md): como
calcular o histograma e o boxplot da latência e a taxa de erros, com um
exemplo resolvido. A planilha
[`docs/analise_metrics.xlsx`](docs/analise_metrics.xlsx) faz as mesmas contas
com fórmulas, a partir de uma foto real do `/metrics`.

**Retenção:**

| Onde | Por quanto tempo |
|---|---|
| `/metrics` | Só a memória do processo: os contadores acumulam desde que o serviço subiu e **zeram ao reiniciar**. |
| Prometheus | Coleta a cada 5 s e guarda **15 dias** (`--storage.tsdb.retention.time=15d`, volume `prometheus-dados`). |
| Grafana | Não guarda dados: consulta o Prometheus. |
| Log em arquivo | Até 6 arquivos de 10 MB, cerca de 25 mil pedidos. |

São dois dashboards, carregados automaticamente de
`monitoramento/grafana/dashboards/`, com um link para o outro no topo:
- **BLU - Monitoramento** (`blu.json`), o painel inicial, com a operação do
  serviço;
- **BLU - Estatísticas (AV2)** (`blu_estatisticas.json`), com o drift e os
  percentis exatos (seção 9.7).

Painéis do **BLU - Monitoramento**:

| Painel | Tipo de gráfico | Por que este tipo |
|---|---|---|
| Pedidos, erros 4xx, falhas 5xx, p95 estimado, p95 exato, ocupação média | cartões com o número | leitura de relance do estado atual; a cor muda nos limites (p95 vermelho acima do SLA de exemplo de 400 ms; qualquer 5xx em vermelho) |
| Pedidos por segundo, por endpoint e status | linhas | volume ao longo do tempo, com uma série por combinação |
| Taxa de erros: cliente (4xx) × servidor (5xx) | linhas, em % | separa o erro de quem chama (arquivo inválido) da falha do serviço, que é o que importa para a operação |
| Latência de `/prever_ocupacao`: estimada × p95 exato | linhas em escala log, com o SLA tracejado | p50/p95/p99 estimados pelas faixas do histograma e, em branco, o p95 exato do exportador de drift. A diferença entre os dois mostra o erro da estimativa por faixas. Só respostas 200: os erros 400 respondem em milissegundos e puxariam os quantis para baixo |
| Histograma da latência no tempo | heatmap | a distribuição inteira minuto a minuto, que os quantis resumem |
| Nitidez e luminância | linhas (mediana e p10) com a faixa proibida em vermelho | o p10 mostra a parte pior das imagens chegando à faixa de problema antes da mediana |
| Imagens com problema de qualidade | linhas, em % das imagens dos últimos 5 min, com a linha de 30% | é exatamente o que o alerta avalia (seção 6.6); cada motivo é uma linha independente |
| Ocupação das vagas ao longo do tempo | área, em % | a informação de negócio do serviço; mudança brusca sem mudança real no estacionamento sugere erro do modelo |
| Vagas detectadas por imagem | linha (média) com referência tracejada em 7 | queda sustentada abaixo de 7 indica câmera deslocada ou vaga encoberta |
| Confiança média das detecções | linhas por estado (LIVRE / OCUPADA) | queda de confiança costuma vir antes do erro |

**O que mudou na revisão deste painel:**
- **A taxa de erros foi separada em 4xx e 5xx.** Antes, os 10% de arquivos
  inválidos do `just carga` deixavam o cartão laranja, como se o serviço
  estivesse falhando.
- **O p95 agora é só do `/prever_ocupacao`, fica vermelho acima de 400 ms e
  ganhou o valor exato ao lado.** Antes, o cartão misturava os dois
  endpoints e só mudava de cor a partir de 1 s. Além disso, as faixas do
  histograma do BentoML pulam de 100 para 250 e 500 ms, e o
  `histogram_quantile` interpola entre elas, então o valor é aproximado.
- **"Ocupação (última imagem)" virou média de 5 minutos.** Uma única foto
  fazia o número pular entre 0% e 100%.
- **O painel "Imagens suspeitas por minuto" saiu**, porque repetia o de
  "% com problema". No lugar entrou a ocupação ao longo do tempo.
- **O heatmap não mostra mais as faixas vazias acima de 10 s.**

O `docker-compose.yml` sobe seis contêineres:
- **`blu`:** o serviço, na imagem gerada por `just docker`. O log é gravado em
  `./logs` por um volume montado.
- **`prometheus`:** coleta o `/metrics` em `host.docker.internal:3000`, a
  porta 3000 da máquina. É o contêiner `blu` publicado ali, ou o `just serve`
  no modo sem contêiner. Um alvo só para os dois modos evita contar o mesmo
  serviço duas vezes. Também avalia as regras de alerta (seção 6.6).
- **`grafana`:** os dois dashboards.
- **`alertmanager`** e **`mailpit`:** os alertas por e-mail (seção 6.6).
- **`estatisticas`:** o exportador de drift (seção 9.7), na porta 9101. Lê o
  log em `./logs` e a referência em `./estatisticas`, e o Prometheus coleta
  as métricas dele.

O contêiner do serviço roda como root. O usuário padrão da imagem do BentoML
(`bentoml`, uid 1034) não teria permissão de escrita na pasta `./logs`
montada do host no Linux. Funciona no Docker Desktop (Windows/macOS) e no
Linux.

### 6.6. Alertas por e-mail: problema na câmera

Os painéis do Grafana só ajudam se alguém estiver olhando. Para o caso que
mais ameaça o serviço, uma **câmera com defeito** que passa a mandar imagens
ruins, o Prometheus avalia regras de alerta e o **Alertmanager** manda um
e-mail à equipe.

```
serviço (/metrics) → Prometheus (regras, a cada 5 s) → Alertmanager (agrupa, evita repetir) → e-mail
```

**As regras** (`monitoramento/alertas.yml`):

| Alerta | Dispara quando, nos últimos 5 min | Provável causa |
|---|---|---|
| `CameraImagemBorrada` | mais de 30% das imagens com nitidez < 15 | foco, lente suja, câmera tremendo |
| `CameraImagemEscura` | mais de 30% das imagens com luminância média < 100 | pouca luz, algo cobrindo a câmera |
| `CameraImagemEstourada` | mais de 30% das imagens com mais de 1% dos pixels estourados | reflexo, sol direto na lente |

Os limites por imagem são os da seção 6.3, tirados das fotos de treino. Cada
regra tem três salvaguardas para não alertar à toa:
- olha **uma fração** das imagens, não uma foto isolada (alguém passando a mão
  na frente não dispara alerta);
- exige **pelo menos 5 imagens** nos 5 minutos;
- a situação precisa **durar 1 minuto** (`for: 1m`) antes de virar alerta.

A regra "vagas detectadas diferente de 7" (câmera deslocada) ficou de fora de
propósito. A referência vem das fotos de treino, que são de celular e de perto
(só 27% mostram as 7 vagas), então não há uma base confiável para ela. As fotos
de `samples/` já mostram a maquete inteira, e o modelo encontra as 7 vagas nas
três. Vale ativar a regra quando a referência for refeita com fotos da câmera
fixa.

**O e-mail:** por padrão vai para o **Mailpit**, uma caixa de e-mail local do
próprio `docker-compose`, em http://localhost:8025. Não precisa de conta nem
de senha. O assunto é `[BLU] ALERTA: ...` quando o problema começa e
`[BLU] Resolvido: ...` quando as imagens voltam ao normal. O corpo traz o
percentual de imagens afetadas, a provável causa, o horário e links para o
Grafana e o Alertmanager.

**Testar** (com `just up` rodando):

```bash
# simula lente suja: 80% das imagens borradas
uv run just carga --n 70 --intervalo 0.3 --degradar 0.8 --tipo borrada --fracao-erro 0
```

| Tempo | O que acontece | Onde ver |
|---|---|---|
| durante a carga | o alerta fica `pending` | http://localhost:9090/alerts |
| ~1 min depois | o alerta vira `firing` | http://localhost:9090/alerts e http://localhost:9093 |
| ~30 s depois | chega o e-mail `[BLU] ALERTA: Imagens borradas...` | http://localhost:8025 |
| após imagens normais | chega o e-mail `[BLU] Resolvido: ...` (até ~5 min depois) | http://localhost:8025 |

Para gerar as imagens normais, rode `uv run just carga --n 160 --intervalo
0.05 --fracao-erro 0`. Com `--tipo escura` ou `--tipo estourada` testam-se os
outros dois alertas.

**Mandar para um e-mail de verdade:**
1. Copie `.env.alertas.example` para `.env.alertas`. Esse arquivo não vai para
   o git.
2. Preencha servidor, usuário, senha e destinatário. No Gmail, use uma
   "senha de app", não a senha normal.
3. Rode `uv run just up` de novo.

O `monitoramento/alertmanager/iniciar.sh` monta a configuração do
Alertmanager com esses valores quando o contêiner sobe. As credenciais ficam
num arquivo separado do `.env` para a senha do e-mail não chegar ao contêiner
do serviço.


## 7. Licença

Este projeto é distribuído sob a **GNU Affero General Public License v3.0**
(ver [LICENSE](LICENSE)). A escolha não é arbitrária: `best.pt` deriva dos
pesos pré-treinados `yolov8n-obb.pt` da ultralytics, distribuídos sob
AGPL-3.0, e o código que os carrega herda essa obrigação de licenciamento.

## Estrutura do repositório

```
src/blu_service/
  model.py         - carrega best.pt e roda a inferência (ultralytics YOLO)
  logic.py         - regra de negócio: classe → estado_vaga, agregação, limiar
  render.py        - desenha o ResultadoOcupacao sobre a imagem (endpoint de imagem)
  service.py       - endpoints BentoML (/prever_ocupacao, /prever_ocupacao_imagem)
  monitoramento.py - log em arquivo, nitidez/iluminação da imagem e métricas blu_*
tests/
  test_logic.py         - testes da regra de negócio (rápidos, sem carregar o modelo)
  test_render.py        - testes do desenho das anotações (rápidos, sem carregar o modelo)
  test_monitoramento.py - características da imagem, formato do log, rotação e middleware
  test_estatisticas.py  - percentis, KS e PSI do relatório da seção 9
scripts/
  gerar_carga.py      - carga no serviço (just carga), com imagens degradadas opcionais
  gerar_referencia.py - conjunto de referência a partir das fotos de treino (just referencia)
  analisar_log.py     - log × referência: KS, PSI, histogramas e boxplots (just analisar)
  exportar_metrics.py - foto do /metrics em planilha com fórmulas (just planilha)
  gerar_producao.py   - conjunto de produção: 120 variações das fotos de samples/ (seção 9)
  submeter_conjunto.py - manda um conjunto ao serviço e grava tempo e respostas (seção 9)
  relatorio_estatisticas.py - p90/p95/p99, boxplot/violino, KS e PSI (seção 9)
  exportador_drift.py - KS, PSI e percentis exatos ao vivo para o Grafana (seção 9.7)
  _estatisticas.py    - funções de cálculo (percentis, KS, PSI) usadas pelos dois acima
referencia/       - conjunto de referência (245 fotos de treino) e resumo com percentis
estatisticas/     - respostas dos dois conjuntos (CSV), relatorio.md e gráficos (seção 9)
monitoramento/
  prometheus.yml  - coleta do /metrics e ligação com o Alertmanager
  alertas.yml     - regras de alerta de qualidade da imagem (câmera)
  alertmanager/   - configuração do Alertmanager e modelo do e-mail de alerta
  grafana/        - fonte de dados e os dois dashboards do Grafana (provisionados)
  exportador/     - Dockerfile do exportador de drift (contêiner estatisticas)
docs/             - metrics.md (como ler o /metrics), planilha, screenshot do Grafana e
                    relatorio_av2.pdf (relatório da seção 9; fonte em relatorio_av2.html)
docker-compose.yml - sobe o serviço (3000), Prometheus (9090), Grafana (3001),
                     Alertmanager (9093), Mailpit (8025) e exportador de drift (9101)
.env.alertas.example - modelo para mandar os alertas a um e-mail de verdade
samples/          - 3 imagens de exemplo (maquete inteira, vista de cima) + 1 arquivo
                    inválido, para os testes da seção 4
best.pt           - pesos do modelo (versionado; ver seção 5)
bentofile.yaml    - manifesto de build/containerização do BentoML
justfile          - atalhos: serve, demo, lint, test, docker, docker-run, up, down,
                    monitor-up, carga, planilha, analisar, referencia,
                    gerar-producao, submeter, estatisticas, av2
requirements.txt  - instala só o uv via pip (não as dependências do serviço)
```

## 8. Uso de IA

Utilização do Claude Opus 5.5 para criação inicial do esqueleto do código e correção de erros, e utilização do Copilot do VS Code para verificar o funcionamento de algumas funcionalidades do projeto.

## 9. Estatísticas de monitoramento: percentis, KS e PSI

Esta seção calcula as estatísticas de monitoramento a partir de pedidos reais
ao serviço:
1. os percentis p90, p95 e p99 do tempo de resposta de `POST /prever_ocupacao`;
2. um conjunto de **referência** e um de **produção**, cada um com mais de 100
   imagens válidas;
3. a comparação entre os dois conjuntos de uma variável de entrada
   (`nitidez`, por KS) e de uma variável de saída (`ocupadas`, por PSI).

O relatório de entrega, com 2 páginas, está em
[`docs/relatorio_av2.pdf`](docs/relatorio_av2.pdf). O relatório gerado pelo
script, com todas as tabelas e gráficos, está em
[`estatisticas/relatorio.md`](estatisticas/relatorio.md).

### 9.1. Como rodar

```bash
uv run just docker    # só na primeira vez, ou se o código do serviço mudar
uv run just up        # sobe o serviço (e o monitoramento) em contêiner
uv run just av2       # gera a produção, submete os dois conjuntos e monta o relatório
```

O `just av2` executa, em ordem:

| Passo | Comando | Saída |
|---|---|---|
| gera 120 imagens de produção | `just gerar-producao` | `dados/producao/` e `manifesto.csv` com os parâmetros de cada imagem |
| submete a referência | `just submeter referencia train-maquete/images` | `estatisticas/respostas_referencia.csv` |
| submete a produção | `just submeter producao dados/producao` | `estatisticas/respostas_producao.csv` |
| calcula as estatísticas | `just estatisticas` | `estatisticas/relatorio.md` e os gráficos `latencia.png`, `latencia_boxplot_violino.png`, `nitidez.png` e `ocupadas.png` |

Enquanto o `just av2` roda, o painel **BLU - Estatísticas (AV2)** no Grafana
(http://localhost:3001) mostra o KS, o PSI e os percentis mudando ao vivo
(seção 9.7).

O `train-maquete/` não é versionado (ver seção 5), mas os CSVs de resposta
são. Sem os datasets, `uv run just estatisticas` refaz o relatório a partir
dos CSVs. O `dados/producao/` também não é versionado, porque a semente fixa
faz o `just gerar-producao` gerar sempre as mesmas imagens.

Cada linha dos CSVs é um pedido, com:
- `tempo_resposta_ms`: medido no cliente;
- `ocupadas`, `livres` e `total_vagas`: lidos da resposta;
- `nitidez` e `latencia_servidor_ms`: copiados do log do servidor
  (`logs/inferencias.jsonl`) pelo `request_id`. Assim a variável de entrada
  vem do próprio monitoramento da seção 6, e não de um cálculo à parte.

### 9.2. Os dois conjuntos

| Conjunto | Imagens | De onde vêm |
|---|---:|---|
| **Referência** | 245 | As fotos de treino do modelo (`train-maquete/images`), como o enunciado indica para quem treinou o próprio modelo. São fotos de celular, de perto. |
| **Produção** | 120 | As 3 fotos de `samples/`, que mostram a maquete inteira vista de cima como a câmera fixa veria, com 40 variações de cada uma ([`scripts/gerar_producao.py`](scripts/gerar_producao.py)). |

As variações simulam um dia de operação da câmera fixa:
- **luz do dia:** brilho de 0,75 a 1,2 e contraste de 0,85 a 1,15;
- **câmera levemente mexida:** rotação de até ±3° e deslocamento de até ±2%;
- **foco e lente suja:** desfoque leve (raio de 0 a 1,2) e, em 12 das 120
  imagens, desfoque forte (raio de 2 a 3);
- **ruído do sensor e compressão:** ruído gaussiano de até 6 níveis de cinza
  e JPEG com qualidade de 60 a 95.

Os 365 pedidos voltaram com status `200`.

### 9.3. Tempo de resposta: p90, p95 e p99

Os pedidos foram enviados **um de cada vez, em sequência**, ao serviço no
Docker, rodando em CPU. O tempo de cada pedido vai do envio até a resposta
completa e é medido no cliente com `time.perf_counter()`. Os 3 primeiros
pedidos são de aquecimento e ficam fora da conta, porque o primeiro pedido
carrega o modelo e leva alguns segundos.

| Conjunto | n | p50 (mediana) | p90 | p95 | p99 | máx | IC 95% do p95 |
|---|---:|---:|---:|---:|---:|---:|---|
| referência | 245 | 76,1 ms | 109,8 ms | 125,1 ms | 144,2 ms | 261,3 ms | 116,9 a 132,2 ms |
| produção | 120 | 78,4 ms | 116,7 ms | 126,3 ms | 136,3 ms | 138,7 ms | 116,7 a 132,0 ms |
| **todos** | **365** | **76,9 ms** | **114,7 ms** | **125,6 ms** | **142,9 ms** | **261,3 ms** | **120,1 a 131,5 ms** |

![Tempo de resposta](estatisticas/latencia.png)

Como ler:
- **p50 = 77 ms:** metade dos pedidos foi respondida em até 77 ms.
- **p95 = 126 ms:** 95% dos pedidos levaram até 126 ms, e só 1 em cada 20
  passou disso. Um SLA de "p95 abaixo de 400 ms" seria **atendido com folga**:
  o p95 ficou em cerca de um terço do limite, e nenhum pedido passou de
  400 ms.
- **p90 e p99:** o p90 (115 ms) e o p99 (143 ms) estão perto do p95, ou seja,
  a cauda é curta. O único pedido muito lento (261 ms) é um ponto isolado e
  aparece no máximo, não nos percentis.
- **Por que percentis, e não média:** a distribuição é assimétrica, com a
  maioria dos pedidos entre 65 e 90 ms e uma cauda à direita. A média
  (82 ms) seria puxada pelos pedidos lentos e não diz quantos usuários
  esperaram mais que um limite. O percentil diz isso diretamente, por isso
  os SLAs são escritos sobre o p95.
- **Incerteza:** com n = 365, o p99 fica entre o 4º e o 5º maior tempo, então
  depende de poucos pedidos e muda bastante de uma rodada para outra. O
  intervalo de confiança do p95 por bootstrap (120 a 132 ms) mostra que o
  p95 é bem mais estável.
- **O que aparece na foto não muda o tempo:** a referência e a produção têm
  praticamente o mesmo p95 (125 contra 126 ms). O modelo redimensiona toda
  imagem para 640 px antes da inferência. Só o **formato** da foto pesa um
  pouco, como mostra o violino abaixo.

> **O estado da máquina pesa no resultado.** Na primeira rodada, feita logo
> depois de ligar o Docker Desktop, o p95 da referência deu **224 ms**, e a
> mediana subiu de 80 para 188 ms ao longo da própria rodada. Repetindo com o
> ambiente estável, o p95 caiu para 125 ms e a mediana ficou em torno de
> 75 ms do começo ao fim. Os números acima são dessa segunda rodada. Por
> isso, um SLA deve ser medido com o serviço aquecido, em mais de uma rodada
> e nas mesmas condições de produção. Com vários clientes ao mesmo tempo, os
> percentis seriam maiores que os deste teste sequencial.

#### Boxplot e gráfico de violino

![Boxplot e violino do tempo de resposta](estatisticas/latencia_boxplot_violino.png)

| Conjunto | bigode inf. | Q1 | mediana | média | Q3 | bigode sup. | atípicos |
|---|---:|---:|---:|---:|---:|---:|---:|
| referência | 47,4 | 70,4 | 76,1 | 80,8 | 86,1 | 109,5 | 25 (10,2%) |
| produção | 64,0 | 74,3 | 78,4 | 84,8 | 88,9 | 104,5 | 18 (15,0%) |
| todos | 49,7 | 71,5 | 76,9 | 82,1 | 86,4 | 105,4 | 50 (13,7%) |

Valores em ms. Os bigodes seguem a regra de Tukey: vão até o pedido mais
extremo a no máximo 1,5 IQR da caixa.

**O que o boxplot mostra:**
- **A caixa é estreita:** metade dos pedidos ficou entre 71 e 86 ms (IQR de
  15 ms). O tempo típico é bem previsível.
- **A cauda é só para cima:** a média (82 ms, o losango) fica acima da
  mediana (77 ms), e os pontos atípicos estão todos acima do bigode
  superior. Nenhum pedido foi anormalmente rápido, mas cerca de 14% foram
  anormalmente lentos.
- **Os dois conjuntos são parecidos:** as caixas quase coincidem. A produção
  tem o bigode inferior mais alto (64 ms) porque não tem o grupo de pedidos
  rápidos da referência, explicado abaixo.

**O que o violino mostra e o boxplot esconde:**
- **Os "atípicos" não são pontos soltos, são um grupo.** No violino, entre
  110 e 140 ms, há um segundo bojo, pequeno mas claro, nos dois conjuntos. O
  boxplot marca esses pedidos como atípicos um a um, e fica parecendo ruído.
  No log do servidor, o modelo leva o mesmo tempo nesses pedidos (cerca de
  58 ms, contra 50 a 55 ms nos demais). Os cerca de 60 ms a mais acontecem
  fora da inferência.
  - Eles chegam em **rajadas**: vários pedidos lentos seguidos, com 0,1 a
    0,4 s de intervalo.
  - **Não** acompanham a coleta do Prometheus, que é a cada 5 s, nem o tipo
    de imagem.
  - A causa mais provável é disputa de CPU na máquina, com o Docker Desktop
    no Windows dividindo o processador com outros programas. Não foi
    confirmada.
- **A referência tem um grupo rápido, por volta de 54 ms**, que a produção não
  tem. O violino da referência se alarga embaixo, e o da produção termina em
  64 ms. São as fotos de treino estreitas e compridas, com proporção de 1:2
  ou mais estreita (por exemplo, 739 × 1600 px): 45 das 245, com mediana de
  57 ms. O modelo encaixa a foto num quadro de 640 px mantendo a proporção, e
  uma foto estreita gera uma matriz menor: a inferência cai de cerca de 52
  para 37 ms. As fotos de produção têm todas proporção perto de 4:5, então
  não formam esse grupo.
- **Consequência para o SLA:** o p95 (126 ms) cai dentro do bojo lento. Ou
  seja, o p95 é definido por esse grupo de pedidos atrasados em rajada, e não
  pelo tempo típico do modelo. Para baixar o p95, o ganho está em tirar a
  disputa de CPU (máquina dedicada, mais CPU para o contêiner), e não em
  acelerar o modelo.

### 9.4. Variável de entrada: `nitidez` → KS

A nitidez (variância do Laplaciano, seção 6.3) é um número real não negativo,
então é **contínua**. Pelo critério do enunciado, a comparação é pela
**distância de Kolmogorov-Smirnov**.

| Conjunto | p25 | mediana | p75 |
|---|---:|---:|---:|
| referência | 105,0 | 148,9 | 300,9 |
| produção | 141,3 | 221,3 | 311,7 |

**KS: D = 0,253, p-valor = 5·10⁻⁵ → as distribuições são diferentes.**

![Nitidez](estatisticas/nitidez.png)

Como ler:
- D é a maior distância vertical entre as duas curvas acumuladas (CDFs). Ela
  aparece na nitidez de 201: **66%** das fotos de referência estão abaixo
  desse valor, contra só **41%** das de produção. Com essa diferença e essas
  quantidades de imagens, o p-valor é muito menor que 0,05.
- As imagens de produção são, no geral, **mais nítidas**: a mediana subiu de
  149 para 221. A vista de cima mostra as linhas finas das 7 vagas por
  inteiro, enquanto as fotos de celular de perto têm áreas lisas e partes
  fora de foco.
- A cauda de baixo é parecida nos dois conjuntos (cerca de 18% abaixo de 90),
  porque as 12 imagens de "lente suja" da produção (nitidez de 12 a 52)
  fazem o papel das fotos de treino menos nítidas. O KS não se importa com
  essa cauda: a diferença está no meio da distribuição.

### 9.5. Variável de saída: `ocupadas` → PSI

`ocupadas` é um inteiro de 0 a 7 (a maquete tem 7 vagas), muito abaixo dos
100 valores que o enunciado usa como limite para tratar uma variável como
contínua. Então é **discreta**, e a comparação é pelo **Population Stability
Index**, com uma faixa para cada valor.

| ocupadas | referência | produção |
|---:|---:|---:|
| 0 | 13,5% | 33,3% |
| 1 | 28,2% | 0% |
| 2 | 35,5% | 0% |
| 3 | 11,4% | 0% |
| 4 | 9,0% | 33,3% |
| 5 | 2,4% | 0% |
| 6 | 0% | 0% |
| 7 | 0% | 33,3% |

**PSI = 14,4 → drift** (abaixo de 0,1 é estável, de 0,1 a 0,25 é mudança
moderada, acima de 0,25 é drift).

![Ocupadas](estatisticas/ocupadas.png)

Como ler:
- Na referência, a maioria das fotos tem 1 ou 2 vagas ocupadas, porque a
  foto de perto mostra só uma parte da maquete. Na produção, cada imagem
  mostra as 7 vagas, e as respostas são sempre 0, 4 ou 7: as três cenas de
  `samples/` (livre, misto e ocupado).
- **O modelo deu a mesma resposta nas 40 variações de cada cena**, incluindo
  as 12 com desfoque forte. A mudança na saída vem da cena fotografada, e não
  de erros do modelo causados pelas variações.
- **Sobre o tamanho do número:** quando um valor aparece num conjunto e não
  aparece no outro (como 7 vagas ocupadas, que não existe na referência), a
  proporção zero vira 0,000001 para o logaritmo existir. Isso faz o PSI
  passar de 10. O valor exato não tem significado, mas a conclusão tem: está
  muito acima de 0,25.

### 9.6. Conclusão

- **Desempenho:** o endpoint responde em 77 ms na mediana e 126 ms no p95,
  bem dentro de um SLA de 400 ms para um cliente por vez.
- **Drift na entrada e na saída:** os dois testes apontam que a produção é
  diferente da referência. Não é falha do modelo. A referência são fotos de
  celular de perto, e a produção é a maquete inteira vista de cima, como a
  câmera fixa vai ver.
- **Próximo passo:** refazer a referência com fotos da câmera fixa
  (`just referencia --pasta <fotos>` e `just submeter referencia <fotos>`).
  A partir daí, um KS ou PSI alto volta a significar o que deve: a câmera
  mudou (foco, luz, posição) ou o padrão de ocupação mudou. Acompanhados de
  forma contínua (seção 9.7), esses dois números dizem se o modelo está
  recebendo o tipo de imagem para o qual foi preparado e se está respondendo
  como no treino.

### 9.7. Dashboard no Grafana: KS, PSI e percentis ao vivo

No lugar de um dashboard em Streamlit, as estatísticas desta seção estão no
Grafana do projeto, no painel **BLU - Estatísticas (AV2)**
(http://localhost:3001, link no topo do painel principal). Elas são
calculadas **continuamente**, sobre os pedidos que chegam ao serviço, e não
só uma vez pelo relatório.

**Como funciona:** o contêiner `estatisticas` roda o
[`scripts/exportador_drift.py`](scripts/exportador_drift.py). A cada 10 s,
ele:
1. lê as linhas novas do log `logs/inferencias.jsonl`;
2. guarda os **últimos 100 pedidos** bem-sucedidos (a "janela");
3. compara a janela com o conjunto de referência
   (`estatisticas/respostas_referencia.csv`) pelas mesmas funções do
   relatório ([`scripts/_estatisticas.py`](scripts/_estatisticas.py)): KS da
   `nitidez` e PSI de `ocupadas`;
4. calcula os percentis **exatos** do tempo de resposta da janela, por
   endpoint;
5. publica tudo em http://localhost:9101/metrics. O Prometheus coleta e o
   Grafana desenha.

O que o painel mostra:

| Parte | Painéis |
|---|---|
| Resumo | pedidos na janela; p95 exato (vermelho acima de 400 ms); KS e PSI com o veredito "estável" ou "DRIFT" |
| Tempo de resposta | p50/p90/p95/p99 ao longo do tempo com o SLA tracejado; **boxplot ao longo do tempo** (cada vela é o boxplot dos últimos 100 pedidos: corpo de Q1 a Q3, pavios até os bigodes, linha branca na mediana); tabela com o resumo não paramétrico dos dois endpoints |
| Entrada (nitidez) | D do KS ao longo do tempo contra o D crítico de α = 0,05; quartis da nitidez, referência × janela |
| Saída (vagas ocupadas) | PSI ao longo do tempo em escala log, com os limites 0,1 e 0,25; distribuição de vagas ocupadas, referência × janela |

**Teste ao vivo:**

| Janela com... | KS da nitidez | D crítico | PSI de `ocupadas` | Painel |
|---|---:|---:|---:|---|
| as fotos de referência (`just submeter referencia ...`) | 0,118 | 0,161 | 0,019 | estável / estável |
| as fotos de produção (`just submeter producao ...`) | 0,311 | 0,161 | 15,3 | DRIFT / DRIFT |

O painel acusa drift só quando a entrada muda.

O gráfico de violino não existe no Grafana. Ele está no relatório
(`estatisticas/latencia_boxplot_violino.png`, seção 9.3).

**Diferença entre o painel e o relatório:**
- **Tempo de resposta:** o relatório mede no cliente, do envio à resposta. O
  painel mede no servidor (`latencia_total_ms` do log), sem a rede.
- **Janela:** o relatório olha cada conjunto inteiro (245 e 120 imagens). O
  painel olha sempre os últimos 100 pedidos, venham de onde vierem.

Para mudar o tamanho da janela ou o intervalo de cálculo, altere
`BLU_DRIFT_JANELA` e `BLU_DRIFT_INTERVALO` no `docker-compose.yml`.

## Diferenciais implementados

- **Container Docker**: `uv run just docker` empacota o Bento e gera a imagem
  via `python -m bentoml build` + `python -m bentoml containerize ...
  --image-tag blu_service:latest` (o `--image-tag` fixa o nome da imagem; sem
  ele, cada build gera uma tag com hash aleatório). Depois,
  `uv run just docker-run` (ou `docker run --rm -p 3000:3000
  blu_service:latest`) sobe o container.
  O `bentofile.yaml` instala as bibliotecas de sistema que o OpenCV (puxado
  pela ultralytics) exige em imagem slim — sem elas o container falha com
  `libxcb.so.1: cannot open shared object file` — e aponta o `torch` para o
  índice CPU-only do PyTorch. Essa segunda parte importa: no Linux, o wheel
  padrão do `torch` no PyPI traz suporte a GPU e arrasta ~2,4 GB de
  dependências NVIDIA/CUDA (cudnn, cublas, nccl, triton…) que este serviço
  nunca usa, já que a inferência roda em CPU. Com o índice CPU-only, os
  downloads do container caem de **3,2 GB para ~0,35 GB** (de 111 para 90
  pacotes), e o build fica proporcionalmente mais rápido.
- Comando único do clone à predição: `uv sync && just serve`.
- Proveniência do modelo documentada (seção 5).
- `ruff` rodando em CI (`.github/workflows/ci.yml`).
- O health check padrão `healthz` do BentoML verifica a disponibilidade do serviço.
- **Monitoramento além do pedido** (seção 6):
  - Prometheus e Grafana com painel pronto;
  - alertas por e-mail (Alertmanager) quando as imagens indicam problema na câmera;
  - métrica de nitidez para detectar câmera fora de foco ou lente suja;
  - métricas próprias `blu_*` no `/metrics`;
  - `X-Request-ID` em cada resposta, que liga o pedido à sua linha no log;
  - conjunto de referência versionado e análise KS/PSI de verificação.
