# Project Feedback - machine-setting 개선 사항

**Date**: 2026-03-06
**Source**: DCTN-0306095349 머신 설치 중 발견된 이슈 기반

---

## 1. 이미 적용된 수정 (코드 변경 완료)

### 1.1 Shell 경로 하드코딩 수정

**파일**: `shell/install-shell.sh` line 48
- **Before**: `for f in $HOME/machine_setting/shell/bashrc.d/[0-9]*.sh; do`
- **After**: `for f in $REPO_DIR/shell/bashrc.d/[0-9]*.sh; do`
- **이유**: repo가 `~/machine_setting/` 이외의 경로에 clone 가능

**파일**: `shell/bashrc.d/50-ai-env.sh`
- `$HOME/machine_setting` 참조를 `$_MS_REPO_DIR` 변수로 교체 (동적 경로 탐지)
- `_MS_REPO_DIR`은 `BASH_SOURCE`에서 자동 계산

---

## 2. 반영 완료된 수정 (ALL APPLIED on 2026-03-06)

### 2.1 [HIGH] gpu-index-urls.conf - CUDA 13.1 지원 추가

**파일**: `config/gpu-index-urls.conf`

```conf
# 추가 필요:
cu131=https://download.pytorch.org/whl/cu131
```

**추가 고려**: detect-hardware.sh에 fallback 로직
```bash
# CUDA suffix 매칭 안 되면 가장 가까운 하위 버전으로 fallback
if ! grep -q "^${CUDA_SUFFIX}=" "$REPO_DIR/config/gpu-index-urls.conf"; then
    # cu131 → cu130 → cu126 순으로 시도
    for fallback in $(grep -oP '^cu\d+' "$REPO_DIR/config/gpu-index-urls.conf" | sort -rV); do
        if [[ "$fallback" < "$CUDA_SUFFIX" ]] || [[ "$fallback" == "$CUDA_SUFFIX" ]]; then
            echo "  Warning: $CUDA_SUFFIX not found, falling back to $fallback"
            CUDA_SUFFIX="$fallback"
            break
        fi
    done
fi
```

### 2.2 [HIGH] setup-venv.sh - GPU 패키지 설치 개선

**파일**: `scripts/setup-venv.sh` line 95

**현재 문제**:
1. `--index-url`만 사용하면 bitsandbytes 등 PyPI 전용 패키지 설치 실패
2. nvidia-* 핀 버전이 인덱스에 없을 수 있음

**제안 수정**:
```bash
# Before:
$UV_PIP install $INSTALL_ARGS -r "$PKG_DIR/requirements-gpu.txt" --index-url "$INDEX_URL" 2>&1 | tail -1

# After:
$UV_PIP install $INSTALL_ARGS -r "$PKG_DIR/requirements-gpu.txt" \
    --index-url "$INDEX_URL" \
    --extra-index-url "https://pypi.org/simple" 2>&1 | tail -1
```

### 2.3 [HIGH] cx-Oracle 빌드 격리 문제

**파일**: `scripts/setup-venv.sh` 또는 `packages/requirements-data.txt`

**옵션 A** (권장): cx-Oracle을 requirements-data.txt에서 제거
- `oracledb==3.4.2`가 이미 포함되어 있으며 cx-Oracle의 후속 패키지
- cx-Oracle은 Python 3.12+에서 빌드 이슈가 잦음

**옵션 B**: 별도 설치 로직 추가
```bash
# setup-venv.sh에 추가
# cx-Oracle requires setuptools at build time (no-build-isolation)
if grep -q "cx-Oracle" "$REQ_FILE" 2>/dev/null; then
    $UV_PIP install $INSTALL_ARGS setuptools
    $UV_PIP install $INSTALL_ARGS cx-Oracle --no-build-isolation
fi
```

### 2.4 [MEDIUM] requirements-gpu.txt 구조 개선

**현재 문제**: nvidia-* 패키지를 특정 버전으로 핀하면, 다른 CUDA 인덱스에서 해결 불가

**제안**: 2개 파일로 분리
```
packages/requirements-gpu-core.txt    # torch, torchvision, torchaudio, triton
packages/requirements-gpu-extras.txt  # bitsandbytes, onnxruntime-gpu (PyPI에서 설치)
```
또는 nvidia-* 패키지는 핀하지 않고 torch 의존성으로 자동 해결

### 2.5 [MEDIUM] 디스크 공간 사전 검증

**파일**: `setup.sh` (step [3/6] 전에 추가)

```bash
# 디스크 공간 검증
REQUIRED_GB=15
VENV_DIR=$(dirname "$VENV_PATH")
AVAILABLE_GB=$(df -BG "$VENV_DIR" | awk 'NR==2{print $4}' | tr -d 'G')
if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo "Warning: Only ${AVAILABLE_GB}GB available at $VENV_DIR (${REQUIRED_GB}GB recommended)"
    echo "Consider using --venv <path> with a larger partition"
    if [ "$INTERACTIVE" = true ]; then
        read -rp "  Continue anyway? [y/N]: " CONT
        [[ "$CONT" =~ ^[Yy]$ ]] || exit 1
    else
        exit 1
    fi
fi
```

### 2.6 [MEDIUM] HOME 환경변수 검증

**파일**: `setup.sh` (상단에 추가)

```bash
# HOME 환경변수 검증
if [ -z "${HOME:-}" ] || [ ! -d "$HOME" ]; then
    echo "Warning: HOME is not set or invalid ('$HOME')"
    echo "Attempting to resolve from /etc/passwd..."
    HOME=$(getent passwd "$(whoami)" | cut -d: -f6)
    export HOME
    echo "  HOME set to: $HOME"
fi
```

### 2.7 [LOW] setup-venv.sh - 비대화형 모드에서 read 문제

**파일**: `scripts/setup-venv.sh` line 49-55

setup.sh에서 비대화형으로 호출되면 `read -rp`가 행을 걸 수 있음.
```bash
# 비대화형 호출 감지 추가
if [ -d "$VENV_PATH" ] && [ -t 0 ]; then
    read -rp "  Venv already exists at $VENV_PATH. Recreate? [y/N]: " RECREATE
    ...
else
    echo "  Keeping existing venv."
fi
```

### 2.8 [INFO] PyTorch cu130 인덱스 실태

- cu130 인덱스에서 가져온 PyTorch가 실제로는 `2.10.0+cu128` 빌드
- PyTorch의 cu130 인덱스가 완전한 CUDA 13.0 네이티브를 아직 제공하지 않을 수 있음
- `docs/` 또는 README에 이 사실을 문서화 권장

---

## 3. 수정 우선순위 요약

| 순위 | 이슈 | 영향도 | 파일 |
|------|------|--------|------|
| 1 | GPU 설치 시 --extra-index-url 추가 | 모든 CUDA 설치 | setup-venv.sh |
| 2 | CUDA suffix fallback 로직 | 새로운 CUDA 버전 대응 | detect-hardware.sh |
| 3 | cx-Oracle 제거 또는 빌드 격리 우회 | data 패키지 설치 | requirements-data.txt |
| 4 | 디스크 공간 검증 | 소용량 파티션 | setup.sh |
| 5 | HOME 검증 | 비표준 환경 | setup.sh |
| 6 | requirements-gpu.txt 분리 | 유지보수성 | packages/ |
| 7 | 비대화형 read 문제 | 자동화 환경 | setup-venv.sh |
