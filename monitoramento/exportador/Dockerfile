# Imagem do exportador de drift (scripts/exportador_drift.py).
# Só as bibliotecas de cálculo; o código vem de ./scripts, montado em /app
# pelo docker-compose, então mudar o script não exige reconstruir a imagem.
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.1.3 scipy==1.14.1 prometheus-client==0.21.0

WORKDIR /app
EXPOSE 9101
CMD ["python", "-u", "/app/exportador_drift.py"]
