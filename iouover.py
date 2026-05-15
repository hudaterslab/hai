import cv2
import os
import glob
from ultralytics import YOLO
import subprocess
import torch
import torchvision
import concurrent.futures

# =================[ 경로 및 설정 ]=================
LABEL_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/신호수_txt_V2"
IMAGE_ROOT = "/media/fishduke/raid_disk/ai_dataset/aihub/goso/training/img"
MODEL_PATH = "/home/fishduke/Desktop/hai/hanjin_cctv.pt"

# [수정된 경로] IoU 0.7 이상 저장 폴더
SAVE_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/iouover0.8_v2"

IOU_THRESHOLD = 0.8  # 임계값 설정 (0.7 이상)
PERSON_CLASS_ID = 2  
BATCH_SIZE = 64  
# ===============================================

# 저장 폴더 생성
os.makedirs(SAVE_DIR, exist_ok=True)

def process_partition(files, device_name, image_path_map):
    if device_name.startswith('cuda'):
        torch.cuda.set_device(device_name)

    model = YOLO(MODEL_PATH)
    model.to(device_name)
    high_iou_items = []
    
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
            
            is_match = False
            top_iou_val = 0

            if meta['my_boxes'] and det_boxes:
                my_tensor = torch.tensor(meta['my_boxes'], dtype=torch.float32, device=device_name)
                det_tensor = torch.stack(det_boxes)
                
                iou_matrix = torchvision.ops.box_iou(my_tensor, det_tensor)
                max_ious = iou_matrix.max(dim=1)[0]
                
                # [핵심 로직 변경] 임계값 이상인 경우만 수집
                if (max_ious >= IOU_THRESHOLD).any():
                    is_match = True
                    top_iou_val = max_ious.max().item() # 가장 높은 IoU 기록
            
            if is_match:
                high_iou_items.append({
                    'label_path': meta['label_path'],
                    'img_path': meta['img_path'],
                    'my_boxes': meta['my_boxes'],
                    'detect_boxes': [t.cpu().tolist() for t in det_boxes],
                    'max_iou': top_iou_val
                })
        
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"[{device_name}] Progress: {i}/{len(files)}")

    return high_iou_items

if __name__ == '__main__':
    # 1. 이미지 검색
    print("[STEP 1/3] 이미지 검색 중...")
    image_path_map = {}
    cmd = f'find "{IMAGE_ROOT}" -type f -iname "*.jp*g"'
    result = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
    for path in result.strip().split('\n'):
        if path: image_path_map[os.path.basename(path)] = path

    all_label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    
    # 2. 멀티 GPU 추론
    num_gpus = torch.cuda.device_count()
    devices = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
    partitions = [all_label_files[i::len(devices)] for i in range(len(devices))]
    
    final_items = []
    print(f"추론 시작 (Threshold >= {IOU_THRESHOLD})...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(process_partition, partitions[i], devices[i], image_path_map) for i in range(len(devices))]
        for future in concurrent.futures.as_completed(futures):
            final_items.extend(future.result())

    # 3. 결과 저장 및 출력
    final_items.sort(key=lambda x: x['max_iou'], reverse=True) # IoU 높은 순으로 정렬
    
    print("\n" + "="*50)
    print(f"검색 결과: 총 {len(final_items)}개의 고정밀(High-IoU) 데이터 발견")
    print(f"저장 경로: {SAVE_DIR}")
    print("="*50)

    for item in final_items:
        img = cv2.imread(item['img_path'])
        if img is None: continue
        
        base_name = os.path.basename(item['label_path']).replace('.txt', '.jpg')
        max_iou = item['max_iou']
        
        # 시각화 (Green: 기존모델 / Red: 우리라벨)
        for det_b in item['detect_boxes']:
            cv2.rectangle(img, (int(det_b[0]), int(det_b[1])), (int(det_b[2]), int(det_b[3])), (0, 255, 0), 2)
        for my_b in item['my_boxes']:
            cv2.rectangle(img, (int(my_b[0]), int(my_b[1])), (int(my_b[2]), int(my_b[3])), (0, 0, 255), 3)
            cv2.putText(img, f"IoU: {max_iou:.2f}", (int(my_b[0]), int(my_b[1]-10)), 0, 0.7, (0, 0, 255), 2)
        
        save_filename = f"high_iou_{max_iou:.2f}_{base_name}"
        cv2.imwrite(os.path.join(SAVE_DIR, save_filename), img)

    print(f"\n[완료] 총 {len(final_items)}개의 이미지를 저장했습니다.")