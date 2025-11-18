from ultralytics import YOLO
from PIL import Image
import os

#모델 가중치가 얼마나 실제 이미지와 잘 구분하는지를 판단하기 위해 사용하는 코드

# --- 1. 설정 (사용자가 수정해야 함) ---

# 1-1. 학습된 YOLO 모델 가중치 파일 경로
# (예: 'runs/detect/train/weights/best.pt')
# 수정 후 (모두 / 로 변경)
MODEL_PATH = r"C:\Users\Junhyeok\Desktop\grown\healthy\runs\classify\test5\weights\best.pt"
# 1-2. 테스트할 이미지 파일 경로
TEST_IMAGE_PATH =r"C:\Users\Junhyeok\Desktop\grown\healthy\basil2.png"

# 1-3. 결과 이미지를 저장할 경로
RESULTS_DIR = r'C:\Users\Junhyeok\Desktop\healthy\test5 bestpt'
os.makedirs(RESULTS_DIR, exist_ok=True) # 결과 폴더 생성

# --- 2. 모델 불러오기 ---
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO 모델을 성공적으로 불러왔습니다: {MODEL_PATH}")
except Exception as e:
    print(f"오류: 모델을 불러오는 데 실패했습니다. {MODEL_PATH} 경로를 확인하세요.")
    print(e)
    exit()

# --- 3. 이미지 예측 수행 ---
try:
    print(f"\n이미지 예측 수행 중: {TEST_IMAGE_PATH}")
    
    # model.predict()는 이미지 경로, PIL 이미지, numpy 배열 등을 모두 받을 수 있습니다.
    # save=True : 원본 이미지에 바운딩 박스가 그려진 결과 이미지를 저장합니다.
    # save_txt=True : 바운딩 박스 좌표를 .txt 파일로 저장합니다.
    # conf=0.25 : 신뢰도 25% 이상인 것만 탐지 (기본값)
    results = model.predict(
        source=TEST_IMAGE_PATH,
        save=True,          # 결과를 'runs/detect/predict' 폴더에 자동 저장
        project=RESULTS_DIR, # 저장 위치를 'RESULTS_DIR'로 지정
        name="predict"        # 하위 폴더 이름 (예: test_results/predict)
    ) 
    
    print(f"예측 완료. 결과가 {RESULTS_DIR}\\predict 폴더에 저장되었습니다.")

except Exception as e:
    print(f"오류: 예측 중 오류가 발생했습니다.")
    print(e)
    exit()

# --- 4. 예측 결과 상세 확인 (선택 사항) ---
# results는 리스트이며, 보통 이미지가 1개이므로 results[0]을 봅니다.
if results:
    result = results[0] # 첫 번째 이미지의 결과
    
    # 4-1. 탐지된 객체의 바운딩 박스 정보 (xyxy 형식)
    boxes = result.boxes  # Boxes 객체
    
    print(f"\n--- 💡 탐지된 객체 정보 💡 ---")
    print(f"총 {len(boxes)}개의 객체 탐지됨")

    # 4-2. 클래스 이름 가져오기
    # model.names는 {0: 'classA', 1: 'classB', ...} 형태의 딕셔너리입니다.
    class_names = model.names
    print(f"모델 클래스: {class_names}")

    # 4-3. 각 객체 정보 순회
    for box in boxes:
        # box.cls : 클래스 인덱스 (tensor)
        class_index = int(box.cls[0])
        
        # class_names 딕셔너리에서 이름 찾기
        class_name = class_names[class_index]
        
        # box.conf : 신뢰도 (tensor)
        confidence = float(box.conf[0])
        
        # box.xyxy : [x1, y1, x2, y2] (tensor)
        coords = box.xyxy[0].cpu().numpy()
        
        print(f"  - 클래스: {class_name} (신뢰도: {confidence*100:.2f}%)")
        print(f"    좌표: {coords}")

    # 4-4. (선택) 결과 이미지를 PIL 이미지로 직접 열기
    # result.plot()은 바운딩 박스가 그려진 numpy 배열(BGR)을 반환합니다.
    img_with_boxes = Image.fromarray(result.plot()[:, :, ::-1]) # RGB로 변환
    img_with_boxes.show() # 이미지 보기
    
    # 별도 저장
    # save_path = os.path.join(RESULTS_DIR, "custom_result.jpg")
    # img_with_boxes.save(save_path)
    # print(f"결과 이미지가 {save_path} 에도 저장됨")