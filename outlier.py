import cv2
import os
import glob
from ultralytics import YOLO
import subprocess
import torch
import torchvision
import concurrent.futures
#check outlier by compring iou whethere it stays under the set iou. 
# =================[ 경로 및 설정 ]=================
LABEL_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/신호수_txt"
IMAGE_ROOT = "/media/fishduke/raid_disk/ai_dataset/aihub/goso/training/img"
MODEL_PATH = "/home/fishduke/Desktop/hai/hanjin_cctv.pt"

# [추가] Outlier 저장 경로
SAVE_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/outlier0.7"

IOU_THRESHOLD = 0.7 
PERSON_CLASS_ID = 2  
BATCH_SIZE = 64  
# ===============================================

# 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)

def process_partition(files, device_name, image_path_map):
    if device_name.startswith('cuda'):
        torch.cuda.set_device(device_name)

    model = YOLO(MODEL_PATH)
    model.to(device_name)
    suspicious = []
    
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]
        batch_imgs = []
        batch_meta = []

        for label_path in batch_files:
            base_name = os.path.basename(label_path).replace('.txt', '.jpg')
            if base_name not in image_path_map: continue
            
            img_path = image_path_map[base_name]
            img_array = cv2.imread(img_path)
            if img_array is None: continue
            
            h_img, w_img, _ = img_array.shape
            my_boxes = []
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5: continue
                    c, cx, cy, w, h = map(float, parts)
                    x1, y1 = (cx - w/2) * w_img, (cy - h/2) * h_img
                    x2, y2 = (cx + w/2) * w_img, (cy + h/2) * h_img
                    my_boxes.append([x1, y1, x2, y2])
            
            batch_imgs.append(img_array)
            batch_meta.append({'label_path': label_path, 'img_path': img_path, 'my_boxes': my_boxes})

        if not batch_imgs: continue

        results = model(batch_imgs, conf=0.3, verbose=False, device=device_name, batch=BATCH_SIZE)

        for idx, res in enumerate(results):
            meta = batch_meta[idx]
            det_boxes = [box.xyxy[0].to(device_name) for box in res.boxes if int(box.cls[0]) == PERSON_CLASS_ID]
            
            is_suspicious = False
            max_iou_val = 0

            if meta['my_boxes'] and det_boxes:
                my_tensor = torch.tensor(meta['my_boxes'], dtype=torch.float32, device=device_name)
                det_tensor = torch.stack(det_boxes)
                
                iou_matrix = torchvision.ops.box_iou(my_tensor, det_tensor)
                max_ious = iou_matrix.max(dim=1)[0]
                
                if (max_ious < IOU_THRESHOLD).any():
                    is_suspicious = True
                    max_iou_val = max_ious.min().item()
            elif meta['my_boxes'] and not det_boxes:
                is_suspicious = True
            
            if is_suspicious:
                suspicious.append({
                    'label_path': meta['label_path'],
                    'img_path': meta['img_path'],
                    'my_boxes': meta['my_boxes'],
                    'detect_boxes': [t.cpu().tolist() for t in det_boxes],
                    'max_iou': max_iou_val
                })
        
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"[{device_name}] Progress: {i}/{len(files)}")

    return suspicious

if __name__ == '__main__':
    # 1. 이미지 검색
    print("[STEP 1/3] 이미지 검색 및 경로 맵핑 중...")
    image_path_map = {}
    try:
        cmd = f'find "{IMAGE_ROOT}" -type f -iname "*.jp*g"'
        result = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        for path in result.strip().split('\n'):
            if path: image_path_map[os.path.basename(path)] = path
    except Exception as e:
        print(f"이미지 검색 중 오류 발생: {e}")
        exit()

    all_label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    
    # 2. 멀티 GPU 설정
    num_gpus = torch.cuda.device_count()
    devices = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
    print(f"사용 가능 디바이스: {devices}")

    partitions = [all_label_files[i::len(devices)] for i in range(len(devices))]
    suspicious_items = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(process_partition, partitions[i], devices[i], image_path_map) for i in range(len(devices))]
        for future in concurrent.futures.as_completed(futures):
            suspicious_items.extend(future.result())

    suspicious_items.sort(key=lambda x: x['label_path'])
    
    print("\n" + "="*50)
    print(f"계산 완료: 총 {len(suspicious_items)}개의 의심 데이터 발견")
    print("="*50)

    if not suspicious_items:
        print("검수할 항목이 없습니다.")
        exit()

    # [추가 포인트] 3. Outlier 이미지 파일 저장 로직
    print(f"[STEP 2/3] 의심 데이터 이미지를 {SAVE_DIR} 에 저장 중...")
    for item in suspicious_items:
        img = cv2.imread(item['img_path'])
        if img is None: continue
        
        base_name = os.path.basename(item['label_path']).replace('.txt', '.jpg')
        max_iou = item['max_iou']
        
        # 박스 그리기 (Green: YOLO Detect / Red: My Label)
        for det_b in item['detect_boxes']:
            cv2.rectangle(img, (int(det_b[0]), int(det_b[1])), (int(det_b[2]), int(det_b[3])), (0, 255, 0), 2)
        for my_b in item['my_boxes']:
            cv2.rectangle(img, (int(my_b[0]), int(my_b[1])), (int(my_b[2]), int(my_b[3])), (0, 0, 255), 3)
            cv2.putText(img, f"IoU: {max_iou:.2f}", (int(my_b[0]), int(my_b[1]-10)), 0, 0.7, (0, 0, 255), 2)
        
        # 파일 저장 (파일명에 IoU 값 포함하여 구분 용이하게 함)
        save_filename = f"iou_{max_iou:.2f}_{base_name}"
        cv2.imwrite(os.path.join(SAVE_DIR, save_filename), img)
        print(f": {save_filename}")

    print(f"\n[성공] 모든 의심 이미지 저장 완료!")

    # 4. 시각화 모드 (기존 루프 유지)
    print("\n[STEP 3/3] 시각화 검수 모드 시작")
    print("- D: 다음 | A: 이전 | X: 마킹 | Q: 종료")
    
    current_idx = 0
    total_suspicious = len(suspicious_items)
    while 0 <= current_idx < total_suspicious:
        # (기존 시각화 로직과 동일하므로 생략 가능하나 흐름상 유지)
        item = suspicious_items[current_idx]
        img = cv2.imread(item['img_path'])
        if img is None:
            current_idx += 1
            continue
            
        base_name = os.path.basename(item['label_path'])
        max_iou = item['max_iou']
        
        for det_b in item['detect_boxes']:
            cv2.rectangle(img, (int(det_b[0]), int(det_b[1])), (int(det_b[2]), int(det_b[3])), (0, 255, 0), 2)
        for my_b in item['my_boxes']:
            cv2.rectangle(img, (int(my_b[0]), int(my_b[1])), (int(my_b[2]), int(my_b[3])), (0, 0, 255), 4)

        display_img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
        cv2.putText(display_img, f"[{current_idx+1}/{total_suspicious}] {base_name} IoU:{max_iou:.2f}", (10, 30), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow("DATA CHECKER", display_img)
        
        key = cv2.waitKeyEx(0)
        if key in [ord('d'), ord('D'), 83, 2555904, 65363]: current_idx += 1
        elif key in [ord('a'), ord('A'), 81, 2424832, 65361]: current_idx = max(0, current_idx - 1)
        elif key & 0xFF in [ord('q'), ord('Q')]: break

    cv2.destroyAllWindows()