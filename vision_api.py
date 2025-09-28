from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
import glob
import json
from datetime import datetime
import requests

# 내부 필터 로직 사용
from src.extreme_98_filter import Extreme98Filter


app = FastAPI(title="Google Vision Extreme98 Filter API")


class ProcessDirectoryRequest(BaseModel):
    directory: str
    delete_source: bool = False
    patterns: Optional[List[str]] = None  # 예: ["**/*.webp", "**/*.jpg", "**/*.png"]
    save_images: bool = False
    save_summary: bool = False


class ProcessSessionRequest(BaseModel):
    # 세션 JSON 본문 또는 파일 경로 중 하나 제공
    session: Optional[Dict[str, Any]] = None
    session_path: Optional[str] = None
    # URL 필드 우선순위 (없으면 다음 필드로)
    url_field_priority: Optional[List[str]] = [
        "kakao_object_url",
        "image_cdn_url",
    ]
    # 저장/삭제 옵션
    save_images: bool = False
    save_summary: bool = False
    delete_source: bool = True


def _ensure_results_dirs() -> None:
    os.makedirs("results/selected", exist_ok=True)
    os.makedirs("results/rejected", exist_ok=True)


def _collect_images_from_directory(base_dir: str, patterns: Optional[List[str]]) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    default_patterns = ["**/*.webp", "**/*.jpg", "**/*.jpeg", "**/*.png"]
    use_patterns = patterns if patterns else default_patterns
    image_paths: List[str] = []
    for pattern in use_patterns:
        image_paths.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    return image_paths


def _copy_results(results: Dict[str, Any]) -> None:
    _ensure_results_dirs()
    # 선택된
    for img_path in results.get("selected", []):
        if not os.path.isfile(img_path):
            continue
        filename = os.path.basename(img_path)
        dest_path = os.path.join("results", "selected", filename)
        shutil.copy2(img_path, dest_path)
    # 거부된
    for img_path in results.get("rejected", []):
        if not os.path.isfile(img_path):
            continue
        filename = os.path.basename(img_path)
        dest_path = os.path.join("results", "rejected", filename)
        shutil.copy2(img_path, dest_path)


def _save_summary(results: Dict[str, Any]) -> str:
    _ensure_results_dirs()
    summary = {
        "filter_name": "Extreme 98% Filter",
        "timestamp": datetime.now().isoformat(),
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
    }
    summary_path = os.path.join("results", "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def _download_urls_to_temp(urls: List[str], filenames: Optional[List[Optional[str]]] = None) -> Dict[str, Any]:
    """
    원격 URL 목록을 임시 디렉토리에 다운로드.
    반환: {
      "temp_dir": str,
      "url_to_path": Dict[url, local_path],
      "paths": List[local_path],
      "failed_urls": List[url]
    }
    """
    temp_dir = os.path.join("data", "uploads", f"session-{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    url_to_path: Dict[str, str] = {}
    saved_paths: List[str] = []
    failed_urls: List[str] = []

    for idx, url in enumerate(urls):
        try:
            name_hint = None
            if filenames and idx < len(filenames):
                name_hint = filenames[idx]
            # 파일명 결정: 세션 제공 파일명 또는 URL basename
            if name_hint:
                filename = os.path.basename(name_hint)
            else:
                filename = os.path.basename(url.split("?")[0]) or f"img-{uuid.uuid4().hex}.jpg"
            dst_path = os.path.join(temp_dir, filename)

            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                failed_urls.append(url)
                continue
            with open(dst_path, "wb") as f:
                f.write(resp.content)
            url_to_path[url] = dst_path
            saved_paths.append(dst_path)
        except Exception:
            failed_urls.append(url)

    return {
        "temp_dir": temp_dir,
        "url_to_path": url_to_path,
        "paths": saved_paths,
        "failed_urls": failed_urls,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/process-directory")
def process_directory(payload: ProcessDirectoryRequest) -> JSONResponse:
    base_dir = payload.directory
    images = _collect_images_from_directory(base_dir, payload.patterns)

    filter_instance = Extreme98Filter()
    results = filter_instance.filter_images(images)

    if payload.save_images:
        _copy_results(results)
    summary_path: Optional[str] = None
    if payload.save_summary:
        summary_path = _save_summary(results)

    summary_json = {
        "filter_name": "Extreme 98% Filter",
        "timestamp": datetime.now().isoformat(),
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
    }

    deleted = False
    if payload.delete_source and os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
        deleted = True

    response = {
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
        "summary_path": summary_path,
        "summary": summary_json,
        "deleted_source": deleted,
    }
    return JSONResponse(content=response)


@app.post("/process-uploads")
async def process_uploads(
    files: List[UploadFile] = File(...),
    delete_source: bool = Form(True),
    save_images: bool = Form(False),
    save_summary: bool = Form(False),
) -> JSONResponse:
    # 업로드를 임시 폴더에 저장
    temp_dir = os.path.join("data", "uploads", f"tmp-{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    saved_paths: List[str] = []
    for uf in files:
        filename = os.path.basename(uf.filename) if uf.filename else f"upload-{uuid.uuid4().hex}"
        dst = os.path.join(temp_dir, filename)
        with open(dst, "wb") as out_f:
            out_f.write(await uf.read())
        saved_paths.append(dst)

    # 필터 수행
    filter_instance = Extreme98Filter()
    results = filter_instance.filter_images(saved_paths)

    if save_images:
        _copy_results(results)
    summary_path: Optional[str] = None
    if save_summary:
        summary_path = _save_summary(results)

    summary_json = {
        "filter_name": "Extreme 98% Filter",
        "timestamp": datetime.now().isoformat(),
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
    }

    deleted = False
    if delete_source and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        deleted = True

    response = {
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
        "summary_path": summary_path,
        "summary": summary_json,
        "deleted_source": deleted,
        "temp_dir": temp_dir,
    }
    return JSONResponse(content=response)


@app.post("/process-session")
def process_session(payload: ProcessSessionRequest) -> JSONResponse:
    # 세션 로드
    session_data: Optional[Dict[str, Any]] = None
    if payload.session is not None:
        session_data = payload.session
    elif payload.session_path:
        if not os.path.isfile(payload.session_path):
            raise HTTPException(status_code=400, detail="session_path not found")
        with open(payload.session_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
    else:
        raise HTTPException(status_code=400, detail="Provide session or session_path")

    images = session_data.get("images", []) if isinstance(session_data, dict) else []
    if not images:
        raise HTTPException(status_code=400, detail="No images in session")

    # URL 추출 (우선 kakao_object_url, 다음 image_cdn_url)
    urls: List[str] = []
    filenames: List[Optional[str]] = []
    priorities = payload.url_field_priority or []
    for item in images:
        if not isinstance(item, dict):
            continue
        found_url: Optional[str] = None
        for field in priorities:
            val = item.get(field)
            if isinstance(val, str) and val.strip():
                found_url = val.strip()
                break
        if found_url:
            urls.append(found_url)
            filenames.append(item.get("downloaded_filename"))

    if not urls:
        raise HTTPException(status_code=400, detail="No valid image URLs found in session")

    # 다운로드 수행
    dl = _download_urls_to_temp(urls, filenames)
    temp_dir = dl["temp_dir"]
    url_to_path = dl["url_to_path"]
    local_paths = dl["paths"]
    failed_urls = dl["failed_urls"]

    # 필터 수행
    filter_instance = Extreme98Filter()
    results = filter_instance.filter_images(local_paths)

    # 로컬 경로 → URL 역매핑
    path_to_url: Dict[str, str] = {v: k for k, v in url_to_path.items()}
    selected_urls = [path_to_url[p] for p in results.get("selected", []) if p in path_to_url]
    rejected_urls = [path_to_url[p] for p in results.get("rejected", []) if p in path_to_url]

    # 저장 옵션
    if payload.save_images:
        _copy_results(results)
    summary_path: Optional[str] = None
    summary_json = {
        "filter_name": "Extreme 98% Filter",
        "timestamp": datetime.now().isoformat(),
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
        "selected_urls": selected_urls,
        "rejected_urls": rejected_urls,
        "failed_urls": failed_urls,
    }
    if payload.save_summary:
        summary_path = _save_summary(results)

    # 임시 폴더 삭제 옵션
    deleted = False
    if payload.delete_source and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        deleted = True

    response = {
        "total_processed": results.get("total_processed", 0),
        "selected_count": len(results.get("selected", [])),
        "rejected_count": len(results.get("rejected", [])),
        "selection_rate": results.get("selection_rate", 0.0),
        "selected_images": results.get("selected", []),
        "rejected_images": results.get("rejected", []),
        "selected_urls": selected_urls,
        "rejected_urls": rejected_urls,
        "failed_urls": failed_urls,
        "summary_path": summary_path,
        "summary": summary_json,
        "deleted_source": deleted,
        "temp_dir": temp_dir,
    }
    return JSONResponse(content=response)


# 실행 방법 (예시):
# uvicorn src.vision_api:app --host 0.0.0.0 --port 8000


