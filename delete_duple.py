import os
import re

# =================[ 경로 설정 ]=================
pre_folder = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/outlier0.3"
after_folder = "/home/fishduke/workspace/nas/Hudaters/dataset/122.고소작업 현장 실시간 영상 데이터/01.데이터/1.Training/라벨링데이터/추출_클래스샘플/outlier0.5"
# ===============================================

def get_original_name(filename):
    """
    'iou_0.21_image1.jpg' 와 같은 이름에서 앞의 iou 부분을 제거하고
    'image1.jpg' 라는 진짜 원본 파일명만 추출합니다.
    """
    return re.sub(r'^iou_\d+\.\d+_', '', filename)

def remove_duplicates():
    # 1. 경로 존재 확인
    if not os.path.exists(pre_folder):
        print(f"오류: pre_folder 경로를 찾을 수 없습니다.\n{pre_folder}")
        return
        
    if not os.path.exists(after_folder):
        print(f"오류: after_folder 경로를 찾을 수 없습니다.\n{after_folder}")
        return

    # 2. pre_folder에서 검토 완료된 원본 이미지 이름 추출
    pre_files = os.listdir(pre_folder)
    pre_basenames = set()
    
    for f in pre_files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            pre_basenames.add(get_original_name(f))
            
    print(f" [pre_folder] 검토 완료 데이터: {len(pre_basenames)}개 로드 완료")

    # 3. after_folder에서 중복 검사 및 삭제
    after_files = os.listdir(after_folder)
    deleted_count = 0
    
    for f in after_files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            after_basename = get_original_name(f)
            
            # pre_folder에 있던 이름이라면 삭제
            if after_basename in pre_basenames:
                file_path = os.path.join(after_folder, f)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"삭제 실패 ({f}): {e}")

    # ====================================================
    # 4. [추가] 남은 파일 목록 추출 및 출력/저장
    # ====================================================
    remaining_files = sorted([f for f in os.listdir(after_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print("-" * 50)
    print(f" 중복 제거 완료! (삭제된 중복 파일: {deleted_count}개)")
    print("-" * 50)
    
    if not remaining_files:
        print("남은 파일이 없습니다! 모두 이미 검토하신 파일이었습니다.")
        return

    # 터미널에 출력
    print(f"\n [새로 검토해야 할 남은 파일 목록 (총 {len(remaining_files)}개)]")
    for f in remaining_files:
        print(f)
        
    # 텍스트 파일로 자동 저장
    txt_save_path = os.path.join(after_folder, "remaining_files_list.txt")
    with open(txt_save_path, "w", encoding="utf-8") as txt_file:
        for f in remaining_files:
            txt_file.write(f + "\n")
            
    print("\n" + "-" * 50)
    print(f" 남은 파일 목록이 메모장(.txt) 파일로도 자동 저장되었습니다!")
    print(f" 저장 위치: {txt_save_path}")
    print("-" * 50)

if __name__ == "__main__":
    remove_duplicates()