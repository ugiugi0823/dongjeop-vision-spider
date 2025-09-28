#!/usr/bin/env python3
"""
Extreme 98% Gold Accuracy Filter
극도로 관대한 조건으로 98% 골드 정확도 달성을 목표로 하는 필터
"""

import os
import json
import math
from typing import List, Dict, Any, Optional
from google.cloud import vision
from google.oauth2 import service_account

class Extreme98Filter:
    def __init__(self):
        # Google Vision API 설정
        self.credentials_path = "gen-lang-client-0067666194-685a6efe4b6a.json"
        self.credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-vision']
        )
        self.client = vision.ImageAnnotatorClient(credentials=self.credentials)
        # 최근 라벨 confidence 캐시
        self.last_label_confidence_map: Dict[str, float] = {}

        # 로지스틱 회귀 모델 (라벨 캐시 기반)
        self.logreg_threshold: Optional[float] = None
        self.logreg_intercept: Optional[float] = None
        self.logreg_feature_to_weight: Dict[str, float] = {}
        self._load_logreg_model()
        # 로지스틱 결합 이중 임계값 (초기 값, 튜닝 대상)
        self.logreg_accept_threshold: float = 0.515
        self.logreg_reject_threshold: float = 0.45
        
        # 분석 결과 기반 골드 패턴 키워드 (골드에서 더 자주 나타나는 키워드에 높은 가중치)
        self.gold_pattern_keywords = {
            # 레스토랑 관련 (매우 높은 가중치)
            "restaurant": 5.0,  # 골드 79.4% vs 비골드 61.8% (가중치 증가)
            "dining": 2.5,
            "food": 1.0,  # 비골드에서도 자주 나타남 (가중치 감소)
            "meal": 2.0,
            "cuisine": 2.0,
            
            # 실내 공간 (골드에서 더 자주 나타나는 키워드에 높은 가중치)
            "chair": 4.0,  # 골드 70.6% vs 비골드 52.9% (가중치 증가)
            "ceiling": 3.5,  # 골드 52.9% vs 비골드 41.2% (가중치 증가)
            "light fixture": 3.0,  # 골드 50.0% vs 비골드 32.4% (가중치 증가)
            "furniture": 2.0,  # 골드 47.1% vs 비골드 50.0% (가중치 감소)
            "table": 1.0,  # 가중치 감소 (비골드에서도 자주 나타남)
            "floor": 1.5,
            "wall": 1.5,
            "room": 2.0,
            "interior": 1.0,  # 가중치 감소
            "indoor": 2.0,
            
            # 조명 및 분위기
            "lighting": 1.5,
            "lamp": 1.5,
            "chandelier": 0.0,  # 제외 키워드로 이동
            "ambiance": 1.5,
            
            # 가구 및 장식
            "decoration": 1.0,
            "art": 1.0,
            "plant": 1.0,
            "houseplant": 2.5,  # 골드 32.4% vs 비골드 20.6% (가중치 증가)
            "flowerpot": 3.0,  # 골드 20.6% vs 비골드에서 거의 없음 (가중치 증가)
            
            # 사람 관련 (레스토랑에서 자주 나타남)
            "person": 1.0,
            "people": 1.0,
            "customer": 2.5,  # 골드 32.4% vs 비골드 20.6% (가중치 증가)
            "waiter": 1.5,
            
            # 일반적인 실내 키워드
            "space": 1.0,
            "area": 1.0,
            "environment": 1.0,
            "atmosphere": 1.0,
            
            # 골드에서 매우 중요한 키워드들 (골드 정확도 향상을 위한 높은 가중치)
            "cleanliness": 15.0,  # 골드의 핵심 특징 (최고 가중치)
            "customer": 12.0,     # 레스토랑의 고객 (최고 가중치)
            "houseplant": 10.0,   # 골드에서 자주 나타남 (최고 가중치)
            "flowerpot": 10.0,    # 골드에서 자주 나타남 (최고 가중치)
            "daylighting": 10.0,  # 골드에서 자주 나타남 (최고 가중치)
            "kitchen & dining room table": 8.0,  # 골드 20.6% vs 비골드에서 거의 없음
            "flooring": 6.0,  # 골드 32.4% vs 비골드 26.5%
            
            # 골드에서 자주 나타나는 기본 키워드들 (높은 가중치)
            "restaurant": 8.0,   # 골드에서 매우 자주 나타남
            "chair": 8.0,        # 골드에서 매우 자주 나타남
            "ceiling": 6.0,      # 골드에서 자주 나타남
            "light fixture": 5.0,  # 골드에서 자주 나타남
            
            # 골드에서 자주 나타나는 기타 키워드들 (적절한 가중치)
            "furniture": 2.0,    # 골드에서 자주 나타남
            "table": 2.0,        # 골드에서 자주 나타남
            "floor": 2.0,        # 골드에서 자주 나타남
            "wall": 2.0,         # 골드에서 자주 나타남
            "room": 2.0,         # 골드에서 자주 나타남
            "indoor": 2.0,       # 골드에서 자주 나타남
            
            # 추가 골드 키워드들 (적절한 가중치)
            "food": 1.5,         # 골드에서도 자주 나타남
            "meal": 2.0,         # 골드에서 자주 나타남
            "cuisine": 2.0,      # 골드에서 자주 나타남
            "dining": 2.0,       # 골드에서 자주 나타남
            "lighting": 1.5,     # 골드에서 자주 나타남
            "lamp": 1.5,         # 골드에서 자주 나타남
            "ambiance": 1.5,     # 골드에서 자주 나타남
            "decoration": 1.0,   # 골드에서 자주 나타남
            "art": 1.0,          # 골드에서 자주 나타남
            "plant": 1.5,        # 골드에서 자주 나타남
            "person": 1.5,       # 골드에서 자주 나타남
            "people": 1.5,       # 골드에서 자주 나타남
            "waiter": 2.0,       # 골드에서 자주 나타남
            "space": 1.0,        # 골드에서 자주 나타남
            "area": 1.0,         # 골드에서 자주 나타남
            "environment": 1.0,  # 골드에서 자주 나타남
            "atmosphere": 1.0    # 골드에서 자주 나타남
        }
        
        # 분석 결과 기반 제외 키워드 (골드 보호를 위한 최소한의 제외 키워드)
        self.critical_exclude_keywords = [
            # 비골드에서만 나타나고 골드에서는 절대 없는 키워드들만 유지
            "warehouse", "storage", "fast food", "concrete", "composite material"
        ]
        
        # 중간 제외 키워드 (비워둠 - 모든 제외 키워드를 critical로 이동)
        self.moderate_exclude_keywords = []
        
        # 분석 결과 기반 보너스 키워드 조합 (골드 이미지에서 자주 나타나는 패턴)
        self.bonus_combinations = {
            # 골드 조합 (골드 정확도 향상을 위한 높은 가중치)
            ("restaurant", "cleanliness"): 12.0,      # 골드 특유의 깔끔함
            ("restaurant", "customer"): 10.0,         # 고객이 있는 레스토랑
            ("restaurant", "houseplant"): 8.0,       # 골드에서 자주 나타남
            ("restaurant", "flowerpot"): 8.0,        # 골드에서 자주 나타남
            ("restaurant", "daylighting"): 8.0,      # 자연광이 좋은 레스토랑
            ("restaurant", "chair"): 7.0,            # 골드에서 매우 자주 나타나는 조합
            ("restaurant", "ceiling"): 6.0,          # 골드에서 자주 나타나는 조합
            ("restaurant", "light fixture"): 5.0,    # 골드에서 자주 나타나는 조합
            ("chair", "ceiling"): 5.0,               # 골드에서 자주 나타나는 조합
            ("chair", "light fixture"): 4.0,         # 골드에서 자주 나타나는 조합
            ("kitchen & dining room table", "restaurant"): 6.0,  # 골드에서 자주 나타나는 조합
            
            # 추가 골드 조합들 (적절한 가중치)
            ("restaurant", "furniture"): 2.0,        # 골드에서 자주 나타나는 조합
            ("restaurant", "table"): 2.0,            # 골드에서 자주 나타나는 조합
            ("restaurant", "floor"): 1.5,            # 골드에서 자주 나타나는 조합
            ("restaurant", "wall"): 1.5,             # 골드에서 자주 나타나는 조합
            ("restaurant", "room"): 2.0,             # 골드에서 자주 나타나는 조합
            ("restaurant", "indoor"): 2.0,           # 골드에서 자주 나타나는 조합
            
            # 비골드 조합 약화 (비골드 정확도 유지)
            ("restaurant", "wood stain"): 0.3,       # 비골드에서 자주 나타남 (매우 낮은 가중치)
            ("restaurant", "varnish"): 0.3,          # 비골드에서 자주 나타남 (매우 낮은 가중치)
            ("restaurant", "interior design"): 0.5,  # 비골드에서 자주 나타남 (낮은 가중치)
            ("restaurant", "hardwood"): 0.3,         # 비골드에서 자주 나타남 (매우 낮은 가중치)
            ("restaurant", "brick"): 0.3,            # 비골드에서 자주 나타남 (매우 낮은 가중치)
            ("restaurant", "wood"): 0.5,             # 비골드에서 자주 나타남 (낮은 가중치)
            
            # 기타 조합들
            ("dining", "room"): 2.0,
            ("food", "restaurant"): 0.5,             # 비골드에서도 자주 나타남
            ("flooring", "restaurant"): 1.5          # 가중치 감소
        }
        
        # 분석 결과 기반 최적화된 임계값들 (골드 정확도 극대화)
        self.score_threshold = 0.1  # 기본(관대한) 점수 임계값 - 앵커 판단과 별도로 유지
        self.label_count_threshold = 0  # 기본 라벨 수 제한 - 앵커 판단과 별도로 유지
        self.bonus_score_threshold = 0.0  # 보너스 임계값 - 현재 계산식에서는 참조하지 않음

        # -------------------------
        # 듀얼 스테이지 설정 추가
        # 1) 골드 앵커 스테이지: 골드 핵심 패턴이 강할 경우 즉시 통과시켜 골드 리콜 확보
        # 2) 엄격 스테이지: 앵커가 약한 경우 강한 제외/높은 임계값으로 비골드 정확도 확보
        # -------------------------

        # 1) 골드 앵커 키워드 및 보너스 (부분 집합, 높은 가중)
        self.gold_anchor_keywords = {
            "cleanliness": 10.0,
            "customer": 6.0,
            "houseplant": 4.0,
            "flowerpot": 4.0,
            "daylighting": 4.0,
            "restaurant": 5.0,
            "chair": 3.5,
            "ceiling": 3.0,
            "light fixture": 3.0,
            "kitchen & dining room table": 5.0
        }
        self.gold_anchor_bonus = {
            ("restaurant", "cleanliness"): 6.0,
            ("restaurant", "customer"): 4.0,
            ("restaurant", "houseplant"): 3.0,
            ("restaurant", "flowerpot"): 3.0,
            ("restaurant", "daylighting"): 3.0,
            ("chair", "ceiling"): 2.0,
            ("chair", "light fixture"): 2.0,
            ("kitchen & dining room table", "restaurant"): 4.0
        }
        # 앵커 임계값 및 필수 키워드: 점수 + 맥락 + 강한 양성 + 조합 히트 필요
        self.gold_anchor_threshold = 7.0
        # 그리드 최적 구성: 앵커에서 컨텍스트/양성/네거티브 요구 없음
        self.gold_anchor_required_context_any = set()
        self.gold_anchor_required_positive_any = set()
        self.anchor_strict_negatives = set()

        # 2) 엄격 스테이지 설정(비골드 80% 목표)
        self.strict_critical_exclude_keywords = [
            # 명백한 비골드 패턴
            "warehouse", "storage", "fast food", "concrete", "composite material",
            # 비골드에서 상대적으로 더 자주 나타나는 자재/설비성 키워드
            "wood stain", "varnish", "hardwood", "wood flooring", "brick",
            "desk", "shade", "plywood", "aluminium", "cooking", "chandelier",
            "serveware", "dishware", "picture frame", "human body", "conversation",
            # 문서/텍스트/프로필/간판류(비골드에서 자주 등장)
            "menu", "menu board", "sign", "signage", "poster", "brochure",
            "leaflet", "catalog", "document", "paper", "receipt", "invoice",
            "bill", "text", "font", "logo", "brand", "graphic design",
            "profile", "profile photo", "selfie"
        ]
        # 엄격 스테이지 임계값들
        # 그리드 최적 구성 반영 (엄격 단계)
        self.strict_score_threshold = 1.0
        self.strict_label_count_threshold = 2

        # 엄격 스테이지 통과를 위한 긍정 라벨 집합
        self.strict_positive_pool = {"restaurant", "dining", "chair", "ceiling", "light fixture", "kitchen & dining room table", "indoor", "room"}
        self.strict_required_min_positive = 1
        
    def get_image_labels(self, image_path: str) -> List[str]:
        """Google Vision API를 사용하여 이미지의 라벨을 추출"""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.client.label_detection(image=image)
            labels = response.label_annotations
            # confidence 맵 구성 (0..1)
            conf_map: Dict[str, float] = {}
            out_labels: List[str] = []
            for label in labels:
                name = label.description.lower()
                conf = float(getattr(label, 'score', 1.0) or 1.0)
                out_labels.append(name)
                # 동어 반복 시 최대값 유지
                if name in conf_map:
                    conf_map[name] = max(conf_map[name], conf)
                else:
                    conf_map[name] = conf
            self.last_label_confidence_map = conf_map
            return out_labels
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def calculate_score(self, labels: List[str]) -> float:
        """라벨 기반으로 점수 계산 (극도로 관대한 조건)"""
        if not labels:
            return 0.0
        
        score = 0.0
        # confidence 맵 참조
        conf = self.last_label_confidence_map

        # 1. 골드 패턴 키워드 매칭
        for label in labels:
            if label in self.gold_pattern_keywords:
                w = self.gold_pattern_keywords[label]
                c = conf.get(label, 1.0)
                score += w * c
        
        # 2. 제외 키워드 체크 (극소수만)
        for label in labels:
            c = conf.get(label, 1.0)
            if label in self.critical_exclude_keywords:
                score -= 5.0 * c  # confidence 가중 제외
            elif label in self.moderate_exclude_keywords:
                score -= 2.0 * c  # confidence 가중 제외
        
        # 3. 보너스 조합 체크
        for (keyword1, keyword2), bonus in self.bonus_combinations.items():
            if keyword1 in labels and keyword2 in labels:
                c1 = conf.get(keyword1, 1.0)
                c2 = conf.get(keyword2, 1.0)
                score += bonus * min(c1, c2)
        
        # 4. 라벨 수 보너스 (더 많은 라벨 = 더 많은 정보)
        if len(labels) >= self.label_count_threshold:
            score += 0.5
        
        return score
    
    def calculate_anchor_score(self, labels: List[str]) -> float:
        """골드 앵커 점수 계산 (confidence 가중)"""
        if not labels:
            return 0.0
        conf = self.last_label_confidence_map
        score = 0.0
        for label in labels:
            if label in self.gold_anchor_keywords:
                score += self.gold_anchor_keywords[label] * conf.get(label, 1.0)
        for (k1, k2), bonus in self.gold_anchor_bonus.items():
            if k1 in labels and k2 in labels:
                score += bonus * min(conf.get(k1, 1.0), conf.get(k2, 1.0))
        return score

    def _load_logreg_model(self) -> None:
        """models/logreg_labels.json 로드하여 특징→가중치 매핑 구성"""
        model_path = os.path.join("models", "logreg_labels.json")
        try:
            if not os.path.exists(model_path):
                return
            with open(model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            coef_list = data.get("coef", [])
            intercept_list = data.get("intercept", [])
            self.logreg_threshold = float(data.get("best_threshold", 0.5))
            coef_vec = coef_list[0] if coef_list else []
            self.logreg_intercept = float(intercept_list[0]) if intercept_list else 0.0
            self.logreg_feature_to_weight = {feat: float(w) for feat, w in zip(features, coef_vec)}
        except Exception as e:
            print(f"Failed to load logreg model: {e}")
            self.logreg_threshold = None
            self.logreg_intercept = None
            self.logreg_feature_to_weight = {}

    def _predict_logreg_probability(self, labels: List[str]) -> Optional[float]:
        """현재 라벨 목록으로 로지스틱 확률 계산 (모델 없으면 None)"""
        if self.logreg_threshold is None or self.logreg_intercept is None or not self.logreg_feature_to_weight:
            return None
        z = self.logreg_intercept
        label_set = set(labels)
        for feat, w in self.logreg_feature_to_weight.items():
            if feat in label_set:
                z += w
        try:
            prob = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            prob = 0.0 if z < 0 else 1.0
        return prob

    def is_selected(self, image_path: str) -> bool:
        """이미지가 선택되는지 판단 - 로지스틱 단일 임계(있으면), 없으면 기존 백업"""
        labels = self.get_image_labels(image_path)

        # 1) 로지스틱 단일 임계값(모델 학습 기준과 동일하게 적용)
        prob = self._predict_logreg_probability(labels)
        if prob is not None and self.logreg_threshold is not None:
            return prob >= float(self.logreg_threshold)

        # 2) 모델이 없을 때만 백업 로직 사용 (앵커 → 엄격)
        anchor_score = self.calculate_anchor_score(labels)
        anchor_combo_hits = sum(1 for (k1, k2) in self.gold_anchor_bonus.keys() if k1 in labels and k2 in labels)
        if (
            anchor_score >= self.gold_anchor_threshold
            and anchor_combo_hits >= 1
            and (not self.gold_anchor_required_context_any or any(ctx in labels for ctx in self.gold_anchor_required_context_any))
            and (not self.gold_anchor_required_positive_any or any(pos in labels for pos in self.gold_anchor_required_positive_any))
            and (not self.anchor_strict_negatives or not any(neg in labels for neg in self.anchor_strict_negatives))
        ):
            return True

        if len(labels) < self.strict_label_count_threshold:
            return False

        positive_hits = sum(1 for pos in self.strict_positive_pool if (pos in labels and self.last_label_confidence_map.get(pos, 1.0) >= 0.5))
        if positive_hits < self.strict_required_min_positive:
            return False

        score = self.calculate_score(labels)
        return score >= self.strict_score_threshold
    
    def filter_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """이미지 목록을 필터링"""
        results = {
            "selected": [],
            "rejected": [],
            "total_processed": 0,
            "selection_rate": 0.0
        }
        
        for image_path in image_paths:
            results["total_processed"] += 1
            
            if self.is_selected(image_path):
                results["selected"].append(image_path)
            else:
                results["rejected"].append(image_path)
        
        if results["total_processed"] > 0:
            results["selection_rate"] = len(results["selected"]) / results["total_processed"]
        
        return results

def main():
    """메인 함수"""
    import glob
    import shutil
    from datetime import datetime
    
    filter_instance = Extreme98Filter()
    
    # data/images/indoor/ 폴더의 모든 이미지 파일 찾기
    image_patterns = [
        "data/images/indoor/**/*.webp",
        "data/images/indoor/**/*.jpg",
        "data/images/indoor/**/*.png"
    ]
    
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(all_images)} images to process")
    
    # 이미지 필터링 실행
    results = filter_instance.filter_images(all_images)
    
    # 결과 출력
    print("=== Extreme 98% Filter Results ===")
    print(f"Total processed: {results['total_processed']}")
    print(f"Selected: {len(results['selected'])}")
    print(f"Rejected: {len(results['rejected'])}")
    print(f"Selection rate: {results['selection_rate']:.2%}")
    
    # 선택된 이미지들을 results/selected/ 폴더로 복사
    for img_path in results['selected']:
        filename = os.path.basename(img_path)
        dest_path = f"results/selected/{filename}"
        shutil.copy2(img_path, dest_path)
        print(f"  ✓ Copied: {filename}")
    
    # 거부된 이미지들을 results/rejected/ 폴더로 복사
    for img_path in results['rejected']:
        filename = os.path.basename(img_path)
        dest_path = f"results/rejected/{filename}"
        shutil.copy2(img_path, dest_path)
        print(f"  ✗ Copied: {filename}")
    
    # 결과 요약을 JSON 파일로 저장
    summary = {
        "filter_name": "Extreme 98% Filter",
        "timestamp": datetime.now().isoformat(),
        "total_processed": results['total_processed'],
        "selected_count": len(results['selected']),
        "rejected_count": len(results['rejected']),
        "selection_rate": results['selection_rate'],
        "selected_images": results['selected'],
        "rejected_images": results['rejected']
    }
    
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to results/ folder")
    print(f"Summary saved to results/summary.json")

if __name__ == "__main__":
    main()
