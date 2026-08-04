# 3MD 실행 가이드

3MD의 로컬 PC는 데이터 변환과 실험 실행을 맡고, 원격 GPU 서버는 MedGemma/vLLM 추론만 맡는다. 이 폴더는 로컬 실행 코드, 영어 프롬프트, 설정, 테스트를 관리한다.

## 0. 경로와 원칙

프로젝트 루트에서 아래 명령을 실행한다.

```powershell
cd E:\dhbp\01_PROJECTS\3md
```

- 원본 benchmark 데이터: `03_DATA/raw/` (읽기 전용)
- 변환된 manifest: `03_DATA/metadata/`
- VQA 이미지: `03_DATA/processed/images/`
- 활성 영어 프롬프트: `02_CODE/configs/prompts/`
- 원격 서버의 SSH 별칭, IP, 비밀번호, Hugging Face token은 repo에 저장하지 않는다.

## 1. 로컬 환경

처음 한 번만 실행한다.

```powershell
conda create -n 3md python=3.11 -y
conda activate 3md
python -m pip install --upgrade pip
pip install -r 02_CODE\requirements.txt
```

이후 작업을 시작할 때마다 실행한다.

```powershell
cd E:\dhbp\01_PROJECTS\3md
conda activate 3md
$env:PYTHONPATH = '02_CODE\src'
```

로컬 PC에는 vLLM, CUDA, MedGemma 가중치가 필요 없다. 이들은 원격 GPU 서버에만 둔다.

## 2. 원격 GPU 서버 (WSL) 준비

원격 서버에서 최초 한 번만 환경과 모델을 준비한다.

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install vllm
vllm --help
```

MedGemma는 원격 서버에 다운로드한다. gated model 접근 권한이 있는 Hugging Face 계정으로 먼저 인증한다.

```bash
hf auth login
hf download google/medgemma-1.5-4b-it --local-dir ~/models/medgemma-1.5-4b-it
```

## 3. vLLM 서버 시작

원격 WSL 터미널에서 실행한다. 이 터미널은 서버가 동작하는 동안 열어 둔다.

```bash
conda activate vllm

CUDA_VISIBLE_DEVICES=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve ~/models/medgemma-1.5-4b-it \
  --served-model-name google/medgemma-1.5-4b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --generation-config vllm \
  --max-model-len 8192
```

서버를 멈출 때는 해당 원격 터미널에서 `Ctrl+C`를 누른다.

`02_CODE/scripts/start_remote_vllm.py`는 SSH 명령을 자동으로 구성하는 helper다. 실제 실행 전에는 반드시 dry-run을 확인한다.

```powershell
python 02_CODE\scripts\start_remote_vllm.py `
  --dry-run `
  --host <SSH_ALIAS> `
  --model google/medgemma-1.5-4b-it `
  --port 8000 `
  --gpu-devices 0 `
  --vllm-bin vllm `
  -- --trust-remote-code --generation-config vllm
```

## 4. SSH tunnel과 endpoint 확인

로컬 PowerShell 창 하나에서 터널을 열어 둔다. `<SSH_ALIAS>`는 `C:\Users\<USER>\.ssh\config`에 설정한 host alias다.

```powershell
ssh -N -L 8000:127.0.0.1:8000 <SSH_ALIAS>
```

별도 로컬 PowerShell 창에서 endpoint를 확인한다.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

응답에 `google/medgemma-1.5-4b-it`이 있으면 로컬 runner의 base URL은 항상 다음과 같이 쓴다.

```text
http://127.0.0.1:8000/v1
```

## 5. 텍스트 smoke test

서버와 SSH tunnel이 켜진 상태에서 실행한다. `BASE`, `PLACEBO`, 16개 MBTI arm을 각각 지정할 수 있다.

```powershell
python 02_CODE\scripts\smoke_vllm.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model google/medgemma-1.5-4b-it `
  --arm BASE
```

```powershell
python 02_CODE\scripts\smoke_vllm.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model google/medgemma-1.5-4b-it `
  --arm PLACEBO
```

```powershell
python 02_CODE\scripts\smoke_vllm.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model google/medgemma-1.5-4b-it `
  --arm INTJ
```

Gemma chat template이 system role을 문제 삼을 때만 `--no-system-role`을 추가한다.

```powershell
python 02_CODE\scripts\smoke_vllm.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model google/medgemma-1.5-4b-it `
  --arm BASE `
  --no-system-role
```

성공 기준은 응답의 마지막 부분에 `ANSWER: <선택지>`가 있는 것이다. Gemma의 내부 thought channel은 smoke script가 제거한 뒤 검사한다.

## 6. VQA 이미지 입력 smoke test

원격 서버는 로컬 파일 경로를 볼 수 없으므로, 이미지를 base64 data URL로 만들어 전송한다. 아래 명령은 VQA-RAD의 materialized 이미지 한 장을 사용해 이미지 전송 경로만 검증한다. 이 단계에서는 정답 정확도가 아니라 HTTP 응답과 `ANSWER:` 출력 여부가 성공 기준이다.

```powershell
$imagePath = '03_DATA\processed\images\vqa_rad\test\synpic42202.jpg'
$imageBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))

$payload = @{
  model = 'google/medgemma-1.5-4b-it'
  messages = @(
    @{
      role = 'user'
      content = @(
        @{
          type = 'text'
          text = @"
Look at the medical image and answer the question.
Reply on one line only, in exactly this format:
ANSWER: yes
or
ANSWER: no

Question: Is there evidence of an aortic aneurysm?
"@
        }
        @{
          type = 'image_url'
          image_url = @{ url = "data:image/jpeg;base64,$imageBase64" }
        }
      )
    }
  )
  temperature = 0
  top_p = 1
  seed = 42
  max_completion_tokens = 128
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8000/v1/chat/completions' `
  -Headers @{ Authorization = 'Bearer EMPTY' } `
  -ContentType 'application/json' `
  -Body $payload

$response.choices[0].message.content
```

## 7. Benchmark manifest 생성

원본 데이터가 `03_DATA/raw/`에 준비된 뒤 실행한다.

```powershell
python 02_CODE\scripts\build_manifests.py --overwrite
```

출력은 `03_DATA/metadata/*.jsonl`, `03_DATA/metadata/splits.csv`, `03_DATA/processed/images/`에 생성된다. PMC-VQA 이미지는 용량이 크므로 충분한 로컬 디스크 공간을 확보한다.

작은 smoke 변환은 별도 출력 경로로 실행한다.

```powershell
python 02_CODE\scripts\build_manifests.py `
  --limit 10 `
  --out-dir 03_DATA\metadata\_smoke `
  --processed-dir 03_DATA\processed\_smoke `
  --overwrite
```

특정 benchmark만 다시 만들 때는 `--benchmarks`를 쓴다.

```powershell
python 02_CODE\scripts\build_manifests.py `
  --benchmarks vqa_rad `
  --overwrite
```

SLAKE의 비영어 문항까지 포함하려면 다음 옵션을 추가한다.

```powershell
python 02_CODE\scripts\build_manifests.py `
  --benchmarks slake `
  --include-non-english-slake `
  --overwrite
```

MedMCQA는 test split의 정답이 숨겨져 있어 기본 평가 split으로 `validation`을 사용한다.

## 8. Manifest 검증

manifest 하나를 schema 기준으로 읽어 검증한다.

```powershell
python 02_CODE\scripts\validate_manifest.py 03_DATA\metadata\vqa_rad.jsonl
```

현재 전체 변환 기준 레코드 수는 아래와 같다.

| benchmark | records |
| --- | ---: |
| MedQA | 1,273 |
| MedMCQA | 4,183 |
| PubMedQA | 500 |
| MMLU-medical | 945 |
| VQA-RAD | 450 |
| SLAKE (영어) | 1,061 |
| PathVQA | 6,719 |
| PMC-VQA | 33,430 |
| 합계 | 48,561 |

## 9. 테스트

코드와 프롬프트 검증을 함께 실행한다.

```powershell
$env:PYTHONPATH = '02_CODE\src'
python -m unittest discover -s 02_CODE\tests
```

## 10. PLACEBO control

`PLACEBO`는 장황한 추가 지시문 자체의 효과와 성격 조건 효과를 분리하기 위해 필수다.

```text
PLACEBO - BASE    = 일반적인 추가 context/persona 텍스트 효과
MBTI    - PLACEBO = MBTI 조건화 효과
```

따라서 `02_CODE/configs/prompts/personas/placebo.md`와 16개 조건화 파일은 길이와 섹션 구조를 맞춰야 한다. `02_CODE/tests/test_prompts.py`는 각 조건화 파일이 placebo 단어 수의 15% 이내인지 검사한다.
