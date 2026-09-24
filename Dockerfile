FROM rayproject/ray:2.58.0-py311

WORKDIR /app
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       "celery>=5.3" "redis>=5.0" "panns-inference>=0.1.1" \
       "soundfile>=0.12" "librosa>=0.10" "numpy>=1.24"

COPY ray_panns ./ray_panns
COPY main.py run.py rayservice.yaml ./

EXPOSE 8000
ENV RAY_SERVE_HTTP_HOST=0.0.0.0
CMD ["python", "run.py"]
