# Basil Health Analyzer API

바질 식물의 건강 상태를 분석하고 엽면적(PLA, Projected Leaf Area)을 계산하는 FastAPI 기반 서비스입니다.

## 기능

- 🖼️ **객체 탐지**: YOLO11을 사용하여 바질과 기준 스티커(Scale) 탐지
- 📐 **PLA 계산**: Scale 마커를 기준으로 정확한 엽면적 계산 (mm², cm²)
- 🏥 **상태 분류**: YOLO 분류 모델로 식물 상태 판단 (Healthy/Unhealthy)
- 🎯 **높은 정확도**: HSV 색상 기반 초록색 픽셀 감지

## 프로젝트 구조

```
my_ai_service/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 서버 (엔드포인트 정의)
│   ├── ai_logic.py      # 핵심 AI 로직 (BasilAnalyzer 클래스)
│   └── config.py        # 설정 및 상수 정의
├── weights/
│   ├── det_best.pt      # YOLO11 객체 탐지 모델
│   └── cls_best.pt      # YOLO 분류 모델
├── requirements.txt     # Python 의존성
├── Dockerfile           # Docker 이미지 빌드
└── README.md
```

## 설치 및 실행

### 1. 로컬 개발 환경

#### 필수 요구사항
- Python 3.11+
- pip

#### 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

#### 실행

```bash
# 개발 모드 (Hot reload 지원)
cd my_ai_service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 시작되면 다음 주소에서 API 문서를 볼 수 있습니다:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Docker를 사용한 배포

#### 이미지 빌드

```bash
docker build -t basil-analyzer:latest .
```

#### 컨테이너 실행

```bash
docker run -p 8000:8000 basil-analyzer:latest
```

## API 엔드포인트

### 1. 헬스 체크

```http
GET /health
```

**응답:**
```json
{
  "status": "healthy",
  "message": "서버가 정상적으로 작동 중입니다."
}
```

---

### 2. 식물 이미지 분석 (메인 엔드포인트)

```http
POST /analyze
Content-Type: multipart/form-data

{
  "file": <이미지 파일>
}
```

**요청 예시 (cURL):**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -F "file=@/path/to/plant_image.jpg"
```

**성공 응답 (Status: 200):**

```json
{
  "status": "success",
  "data": {
    "diagnosis": "healthy",
    "confidence": "95.42%",
    "pla_mm2": 1523.45,
    "pla_cm2": 15.23,
    "green_pixels": 5847,
    "message": "분석이 정상적으로 완료되었습니다."
  }
}
```

**실패 응답:**

```json
{
  "status": "error",
  "message": "기준 스티커(Scale)가 탐지되지 않았습니다. 촬영 상태를 확인해주세요."
}
```

**응답 필드 설명:**

| 필드 | 설명 |
|------|------|
| `diagnosis` | 식물 상태: `healthy` (건강) 또는 `unhealthy` (질병) |
| `confidence` | 분류 모델의 신뢰도 (%) |
| `pla_mm2` | 엽면적 (제곱밀리미터) |
| `pla_cm2` | 엽면적 (제곱센티미터) |
| `green_pixels` | 검출된 초록색 픽셀 수 |

---

## 설정

`app/config.py`에서 다음 항목을 조정할 수 있습니다:

```python
# Scale(기준 스티커) 실제 크기
SCALE_REAL_DIAMETER_MM = 16.0  # 스티커의 실제 지름 (mm)

# 초록색 인식 범위 (HSV)
GREEN_HSV_LOWER = [35, 40, 40]    # 초록색 하한
GREEN_HSV_UPPER = [85, 255, 255]  # 초록색 상한

# 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.5  # 탐지 신뢰도
```

## 촬영 권장사항

정확한 분석을 위해 다음 가이드라인을 따르세요:

1. **조명**: 밝고 균등한 자연광 또는 백색 LED 조명 사용
2. **배경**: 단순하고 명확한 배경 (흰색 또는 검은색 권장)
3. **각도**: 식물을 정면에서 촬영 (45도 이상 기울이지 않기)
4. **Scale 마커**: 사진에 기준 스티커가 반드시 포함되어야 함
5. **초점**: 식물과 스티커 모두 선명하게 촬영
6. **해상도**: 최소 640x640 픽셀 이상 권장

## 계산 방식

### PLA(엽면적) 계산 프로세스

1. **Scale 마커 탐지**
   - YOLO 객체 탐지 모델로 스티커의 지름(픽셀) 측정
   - 실제 지름(16mm) / 픽셀 지름 = mm/pixel 비율 계산

2. **초록색 픽셀 추출**
   - 바질 이미지를 HSV로 변환
   - 설정된 범위의 초록색만 추출
   - 노이즈 제거 (모폴로지 연산)

3. **면적 계산**
   ```
   PLA(mm²) = 초록색_픽셀_수 × (mm/pixel)²
   PLA(cm²) = PLA(mm²) / 100
   ```

## 모델 정보

### det_best.pt (객체 탐지)
- **모델**: YOLO11 Detect
- **클래스**:
  - ID 0: Basil (바질/식물)
  - ID 1: Scale (기준 스티커)
- **입력 크기**: 640x640

### cls_best.pt (분류)
- **모델**: YOLO Classify
- **클래스**:
  - Healthy (건강한 식물)
  - Unhealthy (질병 있는 식물)
- **입력 크기**: 224x224

## 에러 처리

| 에러 메시지 | 원인 | 해결책 |
|-----------|------|------|
| Scale 마커가 탐지되지 않음 | 기준 스티커가 사진에 없거나 식별 불가 | 스티커를 식물과 함께 촬영하기 |
| 바질이 탐지되지 않음 | 식물이 사진에 없거나 모델이 인식 못함 | 식물을 정면에서 명확하게 촬영하기 |
| 처리 중 오류 | 시스템 에러 | 로그 확인 후 문제 분석 |

## 로깅

애플리케이션은 상세한 로그를 제공합니다:

```
2024-01-15 10:30:45 - app.ai_logic - INFO - 🤖 AI 모델 로딩 시작...
2024-01-15 10:30:52 - app.ai_logic - INFO - ✅ 모델 로딩 완료!
2024-01-15 10:31:10 - app.main - INFO - 📥 파일 수신: plant_001.jpg
2024-01-15 10:31:11 - app.ai_logic - INFO - 🔍 객체 탐지 시작...
...
```

## 성능 특성

- **응답 시간**: 약 1-3초 (이미지 크기, 서버 사양에 따라 변동)
- **메모리 사용량**: 약 4-6 GB (모델 로딩 시)
- **동시성**: uvicorn worker 수에 따라 조정 가능

## 주의사항

⚠️ **프로덕션 배포 시**
- CORS 설정에서 `allow_origins`을 특정 도메인으로 제한
- 환경변수로 민감한 설정 관리
- 로드 밸런싱 및 오토스케일링 설정
- 모델 가중치 파일의 보안 확보

## 기여

버그 리포트 또는 기능 제안은 이슈로 남겨주세요.

## 라이선스

MIT License
