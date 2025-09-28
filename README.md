Google Vision Extreme98 Filter API (Release)

개요
- Google Cloud Vision 라벨을 이용해 이미지 선택/거절을 수행하는 API 서버입니다.
- 디렉토리/파일 업로드/세션 JSON(URL 다운로드) 3가지 입력 방식을 지원합니다.

필수 준비
 - Python 3.10 이상 및 가상환경(venv)
- Google Cloud Vision 서비스 계정 키 파일 배치: 프로젝트 루트에 `gen-lang-client-0067666194-685a6efe4b6a.json`

설치
```bash
cd release
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

서버 실행
```bash
# 프로젝트 루트에서 실행 권장 (모듈 경로: release.vision_api)
cd <project_root>
source release/.venv/bin/activate
uvicorn release.vision_api:app --host 0.0.0.0 --port 8000
```

헬스체크
```bash
curl http://127.0.0.1:8000/health
```

엔드포인트
1) 디렉토리 처리: /process-directory
 - 입력(JSON):
```json
{
  "directory": "/absolute/path/to/images_or_root",
  "patterns": ["**/*.webp", "**/*.jpg", "**/*.jpeg", "**/*.png"],
  "save_images": true,
  "save_summary": true,
  "delete_source": false
}
```
 - 동작: 재귀적으로 이미지를 수집해 분석, 선택/거절 파일을 `results/selected`, `results/rejected`로 복사. `save_summary=true`이면 `results/summary.json` 저장. 응답 본문에는 항상 요약 `summary`가 포함.

2) 파일 업로드 처리: /process-uploads
 - 입력(form-data):
   - files: 이미지 파일들(여러 개)
   - save_images (bool)
   - save_summary (bool)
   - delete_source (bool, 기본 true)
 - 예시:
```bash
curl -X POST http://127.0.0.1:8000/process-uploads \
  -F "files=@/abs/path/a.jpg" -F "files=@/abs/path/b.png" \
  -F "save_images=true" -F "save_summary=true" -F "delete_source=true"
```

3) 세션 처리(URL 다운로드): /process-session
 - 입력(JSON):
```json
{
  "session_path": "/absolute/path/to/session_xxx.json",
  "url_field_priority": ["kakao_object_url", "image_cdn_url"],
  "save_images": true,
  "save_summary": true,
  "delete_source": true
}
```
 - 동작: 세션 JSON의 URL들을 임시 폴더에 다운로드 후 분석. URL 기준 결과 필드 `selected_urls`, `rejected_urls`, `failed_urls` 포함. `delete_source=true`면 임시 폴더 삭제.

응답 공통 필드
- total_processed, selected_count, rejected_count, selection_rate
- selected_images, rejected_images
- summary_path (save_summary=true일 때 경로), summary (본문 포함)
- process-uploads/process-session: temp_dir, deleted_source
- process-session: selected_urls, rejected_urls, failed_urls

예시 호출
```bash
# 디렉토리
curl -X POST http://127.0.0.1:8000/process-directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory":"./test_api2",
    "patterns":["**/*.jpg"],
    "save_images":true,
    "save_summary":true
  }'

# 세션
curl -X POST http://127.0.0.1:8000/process-session \
  -H "Content-Type: application/json" \
  -d '{
    "session_path":"./session_103_20250915_122735.json",
    "save_images":true,
    "save_summary":true,
    "delete_source":true
  }'
```

예상 응답(JSON 예시)
```json
// /process-directory (save_images=true, save_summary=true)
{
  "total_processed": 71,
  "selected_count": 31,
  "rejected_count": 40,
  "selection_rate": 0.4366,
  "selected_images": [
    "./test_api2/image_018.jpg",
    "./test_api2/image_030.jpg"
  ],
  "rejected_images": [
    "./test_api2/image_024.jpg",
    "./test_api2/image_031.jpg"
  ],
  "summary_path": "results/summary.json",
  "summary": {
    "filter_name": "Extreme 98% Filter",
    "timestamp": "2025-09-28T14:19:36.826Z",
    "total_processed": 71,
    "selected_count": 31,
    "rejected_count": 40,
    "selection_rate": 0.4366,
    "selected_images": ["..."],
    "rejected_images": ["..."]
  }
}

// /process-session (save_images=false, save_summary=false)
{
  "total_processed": 20,
  "selected_count": 12,
  "rejected_count": 8,
  "selection_rate": 0.6,
  "selected_images": [
    "data/uploads/session-<uuid>/20250915_122727_93347262.jpg"
  ],
  "rejected_images": [
    "data/uploads/session-<uuid>/20250915_122728_41102897.jpg"
  ],
  "selected_urls": [
    "https://objectstorage.../20250915_122727_93347262.jpg"
  ],
  "rejected_urls": [
    "https://objectstorage.../20250915_122728_41102897.jpg"
  ],
  "failed_urls": [],
  "summary_path": null,
  "summary": {
    "filter_name": "Extreme 98% Filter",
    "timestamp": "2025-09-28T10:12:34.567Z",
    "total_processed": 20,
    "selected_count": 12,
    "rejected_count": 8,
    "selection_rate": 0.6,
    "selected_images": ["..."],
    "rejected_images": ["..."],
    "selected_urls": ["..."],
    "rejected_urls": ["..."],
    "failed_urls": []
  },
  "deleted_source": true,
  "temp_dir": "data/uploads/session-<uuid>"
}
```

노트
- 저장 기능을 끈 상태에서도 응답 본문에 `summary`는 항상 포함됩니다.
- Google Vision 호출에는 과금이 발생할 수 있습니다.


기술적 기법(Extreme98Filter)
- 라벨 confidence 가중치 합산: Vision 라벨별 `score`를 가중으로 사용해 핵심 골드 패턴의 기여도를 합산
- 보너스 조합 점수: 특정 라벨 페어(예: "restaurant"+"cleanliness")가 동시 등장 시 추가 보너스 부여
- 제외 키워드 감점: 비골드에서 빈번하거나 문서성 라벨(예: "menu", "sign")에 페널티 적용
- 듀얼 스테이지 판정: 1) 골드 앵커(강한 핵심 패턴) 스테이지, 2) 엄격 스테이지(라벨 수/양성 풀/임계값)
- 로지스틱 회귀 백업: `models/logreg_labels.json`을 로드해 특징 존재 기반 확률 계산, 임계값 비교
- 동의어/중복 라벨 처리: 동일 라벨 중복 등장 시 최대 confidence를 사용하여 보수적 집계
- 라벨 수 보너스: 충분한 라벨 수 확보 시 소폭 가산하여 정보량 반영
- 그리드 튜닝된 임계값: 앵커/엄격 단계 각각의 임계값과 요구 조건을 실험 기반으로 최적화

