import json
import os
from glob import glob
# convert pose coordinates to bounding box txt file.  
# =================[ 설정 부분 ]=================
# JSON 파일들이 들어있는 폴더 경로
JSON_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/수신원" 

# 변환된 YOLO 텍스트 파일을 저장할 폴더 경로
OUTPUT_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/신호수_txt_V2"

# 클래스 번호 및 패딩 설정
CLASS_ID = 5
PAD_W_RATIO = 0.05
PAD_H_RATIO = 0.05 # initial was 0.15 
# ===============================================

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 해당 폴더 내의 모든 .json 파일 목록을 가져옵니다
json_files = glob(os.path.join(JSON_DIR, "*.json"))
total_files = len(json_files)

print(f"총 {total_files}개의 JSON 파일을 찾았습니다. 변환을 시작합니다.")

# 진행 상황을 확인하기 위한 카운터
count = 0

for json_path in json_files:
    try:
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # 이미지 해상도 정보
        img_width, img_height = data["Raw Data Info."]["resolution"]
        yolo_lines = []
        
        # annotation 정보 파싱
        annotations = data["Learning Data Info."]["annotation"]
        for ann in annotations:
            # 신호수(WO-02)인 경우만 처리
            if ann["class_id"] == "WO-02":
                valid_x = [kp["x"] for kp in ann["keypoint"] if kp["state"]["valid"]]
                valid_y = [kp["y"] for kp in ann["keypoint"] if kp["state"]["valid"]]
                
                if not valid_x or not valid_y:
                    continue
                
                # 최소/최대 좌표로 박스 생성
                min_x, max_x = min(valid_x), max(valid_x)
                min_y, max_y = min(valid_y), max(valid_y)
                
                w, h = max_x - min_x, max_y - min_y
                
                # 패딩 적용
                final_min_x = max(0, min_x - (w * PAD_W_RATIO))
                final_max_x = min(img_width, max_x + (w * PAD_W_RATIO))
                final_min_y = max(0, min_y - (h * PAD_H_RATIO))
                final_max_y = min(img_height, max_y + (h * PAD_H_RATIO))
                
                # YOLO 포맷 정규화 (0~1)
                cx = (final_min_x + final_max_x) / 2.0 / img_width
                cy = (final_min_y + final_max_y) / 2.0 / img_height
                nw = (final_max_x - final_min_x) / img_width
                nh = (final_max_y - final_min_y) / img_height
                
                yolo_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # 결과 저장 (JSON 내부의 실제 이미지 ID를 가져와서 파일명으로 사용)
        image_id = data["Source Data Info."]["source_data_ID"]
        base_filename = f"{image_id}.txt"

        if yolo_lines:
            with open(os.path.join(OUTPUT_DIR, base_filename), 'w', encoding='utf-8') as out_f:
                out_f.write("\n".join(yolo_lines))
        
        count += 1
        # 1000개마다 진행 상황 출력
        if count % 1000 == 0:
            print(f"진행 중... ({count}/{total_files})")

    except Exception as e:
        print(f"에러 발생 ({os.path.basename(json_path)}): {e}")

print(f"완료! 총 {count}개의 파일이 성공적으로 변환되었습니다.")