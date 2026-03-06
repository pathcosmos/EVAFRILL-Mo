# AI/ML Packages Detail

**Venv**: `/home/ghong/project-ghong/taketimes/llm-star/ai-env/`
**Python**: 3.12.3
**Package Manager**: uv 0.9.18
**Date**: 2026-03-06
**Total Packages**: 252

---

## 1. LLM Providers

| Package | Version | Import Test | Note |
|---------|---------|-------------|------|
| anthropic | 0.84.0 | PASS | Claude API |
| openai | 2.24.0 | PASS | OpenAI / GPT API |
| google-generativeai | 0.8.6 | PASS | Gemini API (deprecated → google.genai 마이그레이션 권장) |
| google-ai-generativelanguage | 0.6.15 | PASS | Google AI 저수준 API |
| google-api-core | 2.30.0 | PASS | Google API 공통 |
| google-api-python-client | 2.191.0 | PASS | Google API 클라이언트 |
| google-auth | 2.49.0.dev0 | PASS | Google 인증 |
| google-auth-httplib2 | 0.3.0 | PASS | httplib2 인증 어댑터 |
| googleapis-common-protos | 1.72.0 | PASS | gRPC proto 정의 |

---

## 2. LangChain Ecosystem

| Package | Version | Import Test |
|---------|---------|-------------|
| langchain | 1.2.10 | PASS |
| langchain-classic | 1.0.1 | PASS |
| langchain-community | 0.4.1 | PASS |
| langchain-core | 1.2.17 | PASS |
| langchain-huggingface | 1.2.1 | PASS |
| langchain-text-splitters | 1.1.1 | PASS |
| langgraph | 1.0.10 | PASS |
| langgraph-checkpoint | 4.0.1 | PASS |
| langgraph-prebuilt | 1.0.8 | PASS |
| langgraph-sdk | 0.3.9 | PASS |
| langsmith | 0.7.11 | PASS |

---

## 3. HuggingFace Ecosystem

| Package | Version | Import Test |
|---------|---------|-------------|
| transformers | 5.2.0 | PASS |
| datasets | 4.6.1 | PASS |
| tokenizers | 0.22.2 | PASS |
| huggingface-hub | 1.5.0 | PASS |
| hf-xet | 1.3.2 | PASS |
| safetensors | 0.7.0 | PASS |
| accelerate | 1.12.0 | PASS |
| peft | 0.18.1 | PASS |
| sentence-transformers | 5.2.3 | PASS |
| sentencepiece | 0.2.1 | PASS |

---

## 4. PyTorch & GPU Stack

### Core

| Package | Version | Import Test | Note |
|---------|---------|-------------|------|
| torch | 2.10.0 (+cu128) | PASS | CUDA 12.8 빌드, Blackwell sm_100 지원 |
| torchaudio | 2.10.0 | PASS | |
| torchvision | 0.25.0 | PASS | |
| triton | 3.6.0 | PASS | torch.compile 백엔드 |

### NVIDIA Runtime Libraries (PyTorch 의존성, pip 패키지)

| Package | Version |
|---------|---------|
| nvidia-cublas-cu12 | 12.8.4.1 |
| nvidia-cuda-cupti-cu12 | 12.8.90 |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 |
| nvidia-cuda-runtime-cu12 | 12.8.90 |
| nvidia-cudnn-cu12 | 9.10.2.21 |
| nvidia-cufft-cu12 | 11.3.3.83 |
| nvidia-cufile-cu12 | 1.13.1.3 |
| nvidia-curand-cu12 | 10.3.9.90 |
| nvidia-cusolver-cu12 | 11.7.3.90 |
| nvidia-cusparse-cu12 | 12.5.8.93 |
| nvidia-cusparselt-cu12 | 0.7.1 |
| nvidia-nccl-cu12 | 2.27.5 |
| nvidia-nvjitlink-cu12 | 12.8.93 |
| nvidia-nvshmem-cu12 | 3.4.5 |
| nvidia-nvtx-cu12 | 12.8.90 |

### GPU Acceleration

| Package | Version | Import Test | Note |
|---------|---------|-------------|------|
| bitsandbytes | 0.49.2 | PASS | 4-bit/8-bit quantization |
| onnxruntime-gpu | 1.24.3 | PASS | Providers: TensorRT, CUDA, CPU |
| onnxruntime | 1.24.3 | PASS | CPU fallback |
| cuda-bindings | 12.9.4 | PASS | CUDA Python bindings |
| cuda-pathfinder | 1.4.0 | PASS | CUDA path resolution |

### PyTorch Blackwell Feature Support

| Feature | Status |
|---------|--------|
| CUDA available | True |
| CUDA runtime version | 12.8 |
| cuDNN version | 91002 |
| Compute Capability | 10.0 (sm_100) |
| Supported architectures | sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| BF16 matmul | PASS |
| FP8 (float8_e4m3fn) | PASS |
| TF32 matmul | OFF (default, 수동 활성화 가능) |
| Flash Attention (SDPA) | PASS |
| torch.compile | PASS |
| NCCL distributed | PASS |
| Multi-GPU (7 GPUs) | PASS |

---

## 5. Classical ML & Scientific Computing

| Package | Version | Import Test |
|---------|---------|-------------|
| scikit-learn | 1.8.0 | PASS |
| scipy | 1.17.0 | PASS |
| xgboost | 3.2.0 | PASS |
| lightgbm | 4.6.0 | PASS |
| numba | 0.64.0 | PASS |
| llvmlite | 0.46.0 | PASS |
| joblib | 1.5.3 | PASS |
| sympy | 1.14.0 | PASS |
| mpmath | 1.3.0 | PASS |
| networkx | 3.6.1 | PASS |
| shapely | 2.1.2 | PASS |

---

## 6. Vector Store & Embedding

| Package | Version | Import Test |
|---------|---------|-------------|
| chromadb | 1.5.2 | PASS |
| faiss-cpu | 1.13.2 | PASS |

---

## 7. Data Processing

### DataFrames & Numerical

| Package | Version | Import Test |
|---------|---------|-------------|
| pandas | 3.0.0 | PASS |
| numpy | 2.4.1 | PASS |
| pyarrow | 23.0.1 | PASS |

### Image Processing

| Package | Version | Import (as) | Test |
|---------|---------|-------------|------|
| pillow | 12.1.0 | PIL | PASS |
| opencv-python | 4.13.0.92 | cv2 | PASS |
| opencv-python-headless | 4.13.0.92 | cv2 | PASS |
| scikit-image | 0.26.0 | skimage | PASS |
| imageio | 2.37.2 | imageio | PASS |
| tifffile | 2026.3.3 | tifffile | PASS |

### Document Processing

| Package | Version | Import (as) | Test |
|---------|---------|-------------|------|
| pdfplumber | 0.11.9 | pdfplumber | PASS |
| pdfminer-six | 20251230 | pdfminer | PASS |
| pypdf | 6.7.5 | pypdf | PASS |
| pypdfium2 | 5.5.0 | pypdfium2 | PASS |
| python-docx | 1.2.0 | docx | PASS |
| python-pptx | 1.0.2 | pptx | PASS |
| openpyxl | 3.1.5 | openpyxl | PASS |
| xlsxwriter | 3.2.9 | xlsxwriter | PASS |
| lxml | 6.0.2 | lxml | PASS |

### OCR

| Package | Version | Import Test |
|---------|---------|-------------|
| easyocr | 1.7.2 | PASS |
| pytesseract | 0.3.13 | PASS |

### Audio

| Package | Version |
|---------|---------|
| pydub | 0.25.1 |
| ffmpy | 1.0.0 |

---

## 8. Database & ORM

| Package | Version | Import (as) | Test |
|---------|---------|-------------|------|
| sqlalchemy | 2.0.48 | sqlalchemy | PASS |
| alembic | 1.18.4 | alembic | PASS |
| psycopg2-binary | 2.9.11 | psycopg2 | PASS |
| pymysql | 1.1.2 | pymysql | PASS |
| oracledb | 3.4.2 | oracledb | PASS |
| cx-oracle | 8.3.0 | cx_Oracle | PASS |
| clickhouse-connect | 0.13.0 | clickhouse_connect | PASS |
| clickhouse-driver | 0.2.10 | clickhouse_driver | PASS |

---

## 9. Web Frameworks & HTTP

### Frameworks

| Package | Version | Import Test |
|---------|---------|-------------|
| fastapi | 0.135.1 | PASS |
| starlette | 0.52.1 | PASS |
| uvicorn | 0.41.0 | PASS |
| uvloop | 0.22.1 | PASS |
| flask | 3.1.3 | PASS |
| werkzeug | 3.1.6 | PASS |
| gradio | 6.8.0 | PASS |
| gradio-client | 2.2.0 | PASS |

### HTTP Clients

| Package | Version | Import Test |
|---------|---------|-------------|
| httpx | 0.28.1 | PASS |
| httpx-sse | 0.4.3 | PASS |
| httpcore | 1.0.9 | PASS |
| requests | 2.32.5 | PASS |
| requests-oauthlib | 2.0.0 | PASS |
| requests-toolbelt | 1.0.0 | PASS |
| urllib3 | 2.0.7 | PASS |
| aiohttp | 3.13.3 | PASS |

### Server Utilities

| Package | Version |
|---------|---------|
| h11 | 0.16.0 |
| httptools | 0.7.1 |
| websockets | 16.0 |
| websocket-client | 1.9.0 |
| watchfiles | 1.1.1 |
| python-multipart | 0.0.22 |

### Templating

| Package | Version |
|---------|---------|
| jinja2 | 3.1.2 |
| markupsafe | 2.1.5 |
| itsdangerous | 2.2.0 |

---

## 10. Auth & Security

| Package | Version | Import (as) | Test |
|---------|---------|-------------|------|
| pyjwt | 2.7.0 | jwt | PASS |
| bcrypt | 5.0.0 | bcrypt | PASS |
| cryptography | 41.0.7 | cryptography | PASS |
| pyopenssl | 23.2.0 | OpenSSL | PASS |

---

## 11. Visualization

| Package | Version | Import Test |
|---------|---------|-------------|
| matplotlib | 3.10.8 | PASS |
| seaborn | 0.13.2 | PASS |
| contourpy | 1.3.3 | PASS |
| cycler | 0.12.1 | PASS |
| fonttools | 4.61.1 | PASS |
| kiwisolver | 1.4.9 | PASS |

---

## 12. Experiment Tracking

| Package | Version | Import Test |
|---------|---------|-------------|
| wandb | 0.25.0 | PASS |

---

## 13. Messaging & Streaming

| Package | Version | Import (as) | Test |
|---------|---------|-------------|------|
| aiokafka | 0.13.0 | aiokafka | PASS |
| kafka-python | 2.3.0 | kafka | PASS |
| paho-mqtt | 2.1.0 | paho.mqtt | PASS |

---

## 14. Cloud & Infrastructure

| Package | Version | Import Test |
|---------|---------|-------------|
| boto3 | 1.34.46 | PASS |
| botocore | 1.34.46 | PASS |
| s3transfer | 0.10.1 | PASS |
| kubernetes | 35.0.0 | PASS |

---

## 15. Monitoring & Telemetry

| Package | Version | Import Test |
|---------|---------|-------------|
| opentelemetry-api | 1.39.1 | PASS |
| opentelemetry-sdk | 1.39.1 | PASS |
| opentelemetry-exporter-otlp-proto-common | 1.39.1 | PASS |
| opentelemetry-exporter-otlp-proto-grpc | 1.39.1 | PASS |
| opentelemetry-proto | 1.39.1 | PASS |
| opentelemetry-semantic-conventions | 0.60b1 | PASS |
| sentry-sdk | 2.54.0 | PASS |
| prometheus-client | 0.24.1 | PASS |
| posthog | 5.4.0 | PASS |

---

## 16. Core Utilities

| Package | Version | Import Test |
|---------|---------|-------------|
| pydantic | 2.12.5 | PASS |
| pydantic-core | 2.41.5 | PASS |
| pydantic-settings | 2.13.1 | PASS |
| python-dotenv | 1.2.2 | PASS |
| click | 8.3.1 | PASS |
| typer | 0.24.1 | PASS |
| typer-slim | 0.24.0 | PASS |
| loguru | 0.7.3 | PASS |
| structlog | 25.5.0 | PASS |
| tqdm | 4.67.3 | PASS |
| colorama | 0.4.6 | PASS |
| rich | 13.7.1 | PASS |
| pygments | 2.17.2 | PASS |
| tenacity | 9.1.4 | PASS |
| backoff | 2.2.1 | PASS |

---

## 17. Async Libraries

| Package | Version | Import Test |
|---------|---------|-------------|
| anyio | 4.12.1 | PASS |
| sniffio | 1.3.1 | PASS |
| aiofiles | 24.1.0 | PASS |
| aiohappyeyeballs | 2.6.1 | PASS |
| aiosignal | 1.4.0 | PASS |
| frozenlist | 1.8.0 | PASS |
| multidict | 6.7.1 | PASS |
| yarl | 1.23.0 | PASS |
| propcache | 0.4.1 | PASS |

---

## 18. Serialization & Schema

| Package | Version | Import (as) | Test |
|---------|---------|-------------|------|
| orjson | 3.11.7 | orjson | PASS |
| ormsgpack | 1.12.2 | ormsgpack | PASS |
| pyyaml | 6.0.1 | yaml | PASS |
| jsonschema | 4.26.0 | jsonschema | PASS |
| jsonschema-specifications | 2025.9.1 | — | PASS |
| jsonpatch | 1.33 | jsonpatch | PASS |
| jsonpointer | 2.0 | jsonpointer | PASS |
| jsonlines | 4.0.0 | jsonlines | PASS |
| marshmallow | 3.26.2 | marshmallow | PASS |
| dataclasses-json | 0.6.7 | dataclasses_json | PASS |

---

## 19. gRPC & Protobuf

| Package | Version |
|---------|---------|
| grpcio | 1.78.0 |
| grpcio-status | 1.71.2 |
| proto-plus | 1.27.1 |
| protobuf | 5.29.6 |

---

## 20. File I/O & Build

| Package | Version |
|---------|---------|
| filelock | 3.20.0 |
| fsspec | 2025.12.0 |
| flatbuffers | 25.12.19 |
| packaging | 24.0 |
| setuptools | 68.1.2 |
| wheel | 0.42.0 |
| build | 1.4.0 |
| pyproject-hooks | 1.2.0 |
| ninja | 1.13.0 |
| lazy-loader | 0.4 |
| pybase64 | 1.4.3 |

---

## 21. Git

| Package | Version |
|---------|---------|
| gitpython | 3.1.46 |
| gitdb | 4.0.12 |
| smmap | 5.0.2 |

---

## 22. Scheduling

| Package | Version |
|---------|---------|
| apscheduler | 3.11.2 |
| python-dateutil | 2.8.2 |
| pytz | 2024.1 |
| tzlocal | 5.3.1 |

---

## 23. NLP & Text

| Package | Version |
|---------|---------|
| regex | 2026.2.28 |
| python-bidi | 0.6.7 |
| markdown | 3.5.2 |
| markdown-it-py | 3.0.0 |
| mdurl | 0.1.2 |

---

## 24. Misc Dependencies

| Package | Version |
|---------|---------|
| attrs | 23.2.0 |
| certifi | 2023.11.17 |
| cffi | 2.0.0 |
| chardet | 5.2.0 |
| charset-normalizer | 3.4.4 |
| distro | 1.9.0 |
| dnspython | 2.6.1 |
| greenlet | 3.3.2 |
| h5py | 3.15.1 |
| idna | 3.6 |
| importlib-metadata | 8.7.1 |
| importlib-resources | 6.5.2 |
| jiter | 0.13.0 |
| jmespath | 1.0.1 |
| lz4 | 4.4.5 |
| mako | 1.3.10 |
| mmh3 | 5.2.0 |
| multiprocess | 0.70.18 |
| mypy-extensions | 1.1.0 |
| oauthlib | 3.2.2 |
| overrides | 7.7.0 |
| pexpect | 4.9.0 |
| platformdirs | 4.9.2 |
| psutil | 7.2.2 |
| ptyprocess | 0.7.0 |
| pyclipper | 1.4.0 |
| pycparser | 3.0 |
| pyparsing | 3.1.1 |
| pypika | 0.51.1 |
| pyrsistent | 0.20.0 |
| pyserial | 3.5 |
| python-magic | 0.4.27 |
| pyasn1 | 0.4.8 |
| pyasn1-modules | 0.2.8 |
| referencing | 0.37.0 |
| rpds-py | 0.30.0 |
| safehttpx | 0.1.7 |
| semantic-version | 2.10.0 |
| shellingham | 1.5.4 |
| six | 1.16.0 |
| threadpoolctl | 3.6.0 |
| tomlkit | 0.13.3 |
| typing-extensions | 4.15.0 |
| typing-inspect | 0.9.0 |
| typing-inspection | 0.4.2 |
| uritemplate | 4.2.0 |
| uuid-utils | 0.14.1 |
| watchdog | 6.0.0 |
| xxhash | 3.6.0 |
| zipp | 3.23.0 |
| zstandard | 0.25.0 |
| brotli | 1.2.0 |
| dill | 0.4.0 |
| docstring-parser | 0.17.0 |
| durationpy | 0.10 |
| groovy | 0.1.2 |
| et-xmlfile | 2.0.0 |
| async-timeout | 5.0.1 |
| blinker | 1.9.0 |
| annotated-doc | 0.0.4 |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total packages | 252 |
| Import-tested packages | 80 |
| Import test pass rate | **80/80 (100%)** |
| Venv size on disk | 9.4 GB |
| Python version | 3.12.3 |
| ONNX Runtime providers | TensorRT, CUDA, CPU |
