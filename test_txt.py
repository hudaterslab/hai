import cv2
import os
import glob
# test bounding box by comparing to original image.
# =================[ 경로 설정 ]=================
LABEL_DIR = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/신호수_txt"
IMAGE_ROOT = "/media/fishduke/raid_disk/ai_dataset/aihub/goso/training/img"

COLORS = {5: (0, 0, 255), 'default': (0, 255, 0)}
# ===============================================

print(" 원본 이미지 위치를 검색 중입니다...")
image_path_map = {}

for root, dirs, files in os.walk(IMAGE_ROOT):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path_map[file] = os.path.join(root, file)

total_images_found = len(image_path_map)
print(f" 검색 완료! 총 {total_images_found}개의 이미지 위치를 확인했습니다.")

if total_images_found == 0:
    print(" 지정한 이미지 폴더에 이미지가 하나도 없습니다. 경로를 다시 확인해주세요.")
    exit()

label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
total_labels = len(label_files)

print(f" 총 {total_labels}개의 라벨에 대해 시각화를 시작합니다.")
print(" [▶] 또는 [D]: 다음 이미지, [◀] 또는 [A]: 이전 이미지, [Q]: 종료")

# -----------------------------------------------
# 방향키 제어를 위한 인덱스 기반 루프
# -----------------------------------------------
i = 0
while 0 <= i < total_labels:
    label_path = label_files[i]
    base_name = os.path.basename(label_path).replace('.txt', '.jpg')
    
    if base_name not in image_path_map:
        print(f" [{i+1}/{total_labels}] 이미지를 찾을 수 없음: {base_name}")
        i += 1 # 못 찾으면 다음으로 강제 이동
        continue
    
    img_path = image_path_map[base_name]
    img = cv2.imread(img_path)
    if img is None:
        i += 1
        continue

    h_img, w_img, _ = img.shape
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: continue
        
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        x1 = int((cx - w / 2) * w_img)
        y1 = int((cy - h / 2) * h_img)
        x2 = int((cx + w / 2) * w_img)
        y2 = int((cy + h / 2) * h_img)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        color = COLORS.get(cls_id, COLORS['default'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"Class {cls_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    display_img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
    
    # UI 텍스트 업데이트
    prog_text = f"[{i+1}/{total_labels}] {base_name}"
    guide_text = "Prev: <- | Next: -> | Quit: Q"
    cv2.putText(display_img, prog_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_img, guide_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Bbox Viewer", display_img)
    
    # 키 입력 대기
    key = cv2.waitKeyEx(0) # 방향키 인식을 위해 waitKeyEx 사용
    
    # 1. 오른쪽 방향키 (Next) - 시스템에 따라 83, 2555904, 3 등이 올 수 있음
    if key == 83 or key == 2555904 or key == 3 or key == ord('d') or key == ord('D'):
        i += 1
    # 2. 왼쪽 방향키 (Prev) - 시스템에 따라 81, 2424832, 2 등이 올 수 있음
    elif key == 81 or key == 2424832 or key == 2 or key == ord('a') or key == ord('A'):
        i -= 1
        if i < 0: i = 0 # 첫 페이지에서 뒤로 가기 방지
    # 3. 종료 (Q)
    elif key & 0xFF == ord('q') or key & 0xFF == ord('Q'):
        break
    # 4. 엔터나 스페이스바는 기본적으로 '다음'으로 설정
    elif key == 13 or key == 32:
        i += 1
    else:
        # 다른 키를 눌렀을 때는 아무 동작도 안 하고 대기
        continue

cv2.destroyAllWindows()
print(" 시각화 프로그램을 종료합니다.")