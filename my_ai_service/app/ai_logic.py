import cv2
import numpy as np
import io
import logging
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO
from datetime import datetime

from app.config import (
    DET_MODEL_PATH,
    CLS_MODEL_PATH,
    SCALE_REAL_DIAMETER_MM,
    GREEN_HSV_LOWER,
    GREEN_HSV_UPPER,
)

logger = logging.getLogger(__name__)


class BasilAnalyzer:
    """바질 식물 분석 클래스"""

    def __init__(self):
        logger.info("🤖 AI 모델 로딩 시작...")

        # 두 개의 모델을 미리 로딩 (메모리에 상주)
        try:
            self.det_model = YOLO(str(DET_MODEL_PATH))  # 탐지용
            self.cls_model = YOLO(str(CLS_MODEL_PATH))  # 분류용
            logger.info("✅ 모델 로딩 완료!")
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise

    def _calculate_pla(self, basil_crop_bgr, mm_per_pixel):
        """
        PLA(엽면적) 계산

        Args:
            basil_crop_bgr: 바질 크롭 이미지 (BGR numpy array)
            mm_per_pixel: 픽셀-mm 변환 비율

        Returns:
            dict: PLA 계산 결과 {
                'pla_mm2': 면적(mm²),
                'pla_cm2': 면적(cm²),
                'green_pixels': 초록색 픽셀 수
            }
        """
        try:
            # 1. HSV로 변환
            basil_hsv = cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2HSV)

            # 2. 초록색 범위로 마스크 생성
            lower_green = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
            upper_green = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
            green_mask = cv2.inRange(basil_hsv, lower_green, upper_green)

            # 3. 노이즈 제거 (모폴로지 연산)
            kernel = np.ones((3, 3), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

            # 4. 초록색 픽셀 수 계산
            green_pixel_count = cv2.countNonZero(green_mask)

            # 5. 면적 계산: 픽셀 수 * (mm/pixel)²
            area_mm2 = green_pixel_count * (mm_per_pixel ** 2)
            area_cm2 = area_mm2 / 100.0

            logger.info(f"[PLA] 초록색 픽셀: {green_pixel_count}, 면적: {area_mm2:.2f} mm² ({area_cm2:.2f} cm²)")

            return {
                "pla_mm2": round(area_mm2, 2),
                "pla_cm2": round(area_cm2, 2),
                "green_pixels": int(green_pixel_count),
            }

        except Exception as e:
            logger.error(f"❌ PLA 계산 중 오류: {e}")
            return None

    def process(self, image_bytes):
        """
        이미지 분석 프로세스

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            dict: 분석 결과
        """
        try:
            # 1. 이미지 로드 및 EXIF 회전 처리 (스마트폰 사진 대응)
            origin_img_pil = Image.open(io.BytesIO(image_bytes))

            # EXIF 정보에 따른 자동 회전 처리
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil)
            origin_img_pil = origin_img_pil.convert("RGB")

            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)
            logger.info(f"📸 이미지 로드됨 (크기: {origin_img_pil.width}x{origin_img_pil.height})")

            # -------------------------------------------------
            # Step 1: YOLO로 scale 마커 검출
            # -------------------------------------------------
            logger.info("🔍 객체 탐지 시작...")
            results = self.det_model(origin_img_pil, conf=0.15)

            mm_per_pixel = 0
            scale_marker_info = None

            # 탐지된 클래스 ID 목록 확인
            found_ids = results[0].boxes.cls.cpu().numpy().astype(int) if len(results) > 0 and len(results[0].boxes) > 0 else []
            logger.info(f"👉 탐지된 ID 목록: {list(set(found_ids))}")

            # ID 1번이 Scale 마커인지 확인
            if 1 not in found_ids:
                logger.error("[Error] ID 1(Scale 마커)을 찾을 수 없습니다.")
                return {
                    "status": "error",
                    "message": "기준 스티커(Scale)가 탐지되지 않았습니다. 촬영 상태를 확인해주세요.",
                }

            # ID 1인 것의 인덱스 찾기
            for result in results:
                boxes = result.boxes
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])

                    # ID 1 = Scale 마커
                    if cls_id == 1:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # scale 마커의 중심과 크기
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        diameter_pixel = max(width, height)  # 더 긴 쪽을 지름으로 사용

                        # 스케일 계산 (실제 지름 16mm / 픽셀 지름)
                        mm_per_pixel = SCALE_REAL_DIAMETER_MM / diameter_pixel

                        logger.info(f"[Scale] ID 1 감지됨: 지름 {diameter_pixel:.2f}px, 신뢰도 {confidence:.2%}")
                        logger.info(f"[Scale] 1 Pixel = {mm_per_pixel:.4f} mm")

                        scale_marker_info = {
                            "class_id": 1,
                            "class_name": "scale",
                            "confidence": float(confidence),
                            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center_x": float(cx),
                            "center_y": float(cy),
                            "diameter_pixel": float(diameter_pixel),
                            "mm_per_pixel": float(mm_per_pixel),
                        }

                        break

                if mm_per_pixel > 0:
                    break

            # -------------------------------------------------
            # Step 2: 바질 탐지 (ID 0 = Basil)
            # -------------------------------------------------
            # ID 0번이 Basil인지 확인
            if 0 not in found_ids:
                logger.error("[Error] ID 0(Basil)을 찾을 수 없습니다.")
                return {
                    "status": "error",
                    "message": "바질(Basil)이 탐지되지 않았습니다.",
                }

            logger.info("🔍 바질(ID:0) 탐지 중...")
            basil_found = False
            basil_crop_bgr = None
            basil_confidence = 0

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    cls_id = int(box.cls[0])

                    # ID 0 = Basil
                    if cls_id == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # 이미지 Crop
                        basil_crop_bgr = origin_img_bgr[y1:y2, x1:x2]
                        basil_confidence = confidence
                        basil_found = True
                        logger.info(f"[Basil] ID 0 감지됨: 신뢰도 {confidence:.2%}")

                        break

                if basil_found:
                    break

            # -------------------------------------------------
            # Step 3: PLA 계산
            # -------------------------------------------------
            logger.info("📐 PLA 계산...")
            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)

            if pla_result is None:
                return {
                    "status": "error",
                    "message": "PLA 계산 중 오류가 발생했습니다.",
                }

            # -------------------------------------------------
            # Step 4: 분류 (Healthy vs Unhealthy)
            # -------------------------------------------------
            logger.info("🏥 식물 상태 분류...")
            basil_crop_pil = Image.fromarray(
                cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2RGB)
            )
            cls_results = self.cls_model(basil_crop_pil)[0]

            probs = cls_results.probs
            top1_idx = probs.top1
            class_name = cls_results.names[top1_idx]  # 'healthy' or 'unhealthy'
            confidence = float(probs.top1conf) * 100

            logger.info(f"분류 결과: {class_name} ({confidence:.2f}%)")

            # -------------------------------------------------
            # Step 5: 최종 결과 생성
            # -------------------------------------------------
            return {
                "status": "success",
                "data": {
                    "diagnosis": class_name,  # 'healthy' or 'unhealthy'
                    "confidence": f"{confidence:.2f}%",
                    "pla_mm2": pla_result["pla_mm2"],
                    "pla_cm2": pla_result["pla_cm2"],
                    "green_pixels": pla_result["green_pixels"],
                    "message": "분석이 정상적으로 완료되었습니다.",
                },
            }

        except Exception as e:
            logger.error(f"❌ 처리 중 오류: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"처리 중 오류가 발생했습니다: {str(e)}",
            }


# 서버 시작 시 인스턴스 생성
analyzer = BasilAnalyzer()
