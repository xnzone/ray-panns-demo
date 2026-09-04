FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       "ray[serve]>=2.9" "jinja2>=3.1" "panns-inference>=0.1.1" \
       "soundfile>=0.12" "librosa>=0.10" "numpy>=1.24"

COPY ray_panns ./ray_panns
COPY main.py run.py rayservice.yaml ./

EXPOSE 8000
ENV RAY_SERVE_HTTP_HOST=0.0.0.0
CMD ["python", "run.py"]
