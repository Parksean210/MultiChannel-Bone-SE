import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. DB 초기화 (스키마 변경 반영)
if os.path.exists("data/metadata.db"):
    os.remove("data/metadata.db")

# 2. Speech Indexing (KsponSpeech)
run_cmd('uv run python3 scripts/manage_db.py speech --path "data/speech/KsponSpeech" --dataset "KsponSpeech" --language "ko"')

# 3. Noise Indexing (극한소음)
# 매뉴얼 매핑: 폴더이름 -> (대분류명, 소분류명)
# 대분류/소분류 이름에서 01, 02 같은 번호를 완전히 제거하여 인덱싱
noise_root = "data/noise/136-2.극한 소음 환경 소리 데이터"
noise_mapping = {
    "01.교통수단": {
        "01.사람의비언어적소리": ("교통수단", "사람의비언어적소리"),
        "02.동물및자연물소리": ("교통수단", "동물및자연물소리"),
        "03.전자제품및생활환경소리": ("교통수단", "전자제품및생활환경소리"),
        "04.기타소리": ("교통수단", "기타소리"),
    },
    "02.공사장": {
        "01.사람의비언어적소리": ("공사장", "사람의비언어적소리"),
        "02.동물및자연물소리": ("공사장", "동물및자연물소리"),
        "03.전자제품및생활환경소리": ("공사장", "전자제품및생활환경소리"),
    },
    "03.공장": {
        "01.사람의비언어적소리": ("공장", "사람의비언어적소리"),
        "02.동물및자연물소리": ("공장", "동물및자연물소리"),
        "03.전자제품및생활환경소리": ("공장", "전자제품및생활환경소리"),
        "04.기타소리": ("공장", "기타소리"),
    },
    "04.시설류": {
        "01.사람의비언어적소리": ("시설류", "사람의비언어적소리"),
        "02.동물및자연물소리": ("시설류", "동물및자연물소리"),
        "03.전자제품및생활환경소리": ("시설류", "전자제품및생활환경소리"),
        "04.기타소리": ("시설류", "기타소리"),
    },
    "05.기타소음": {
        "01.사람의비언어적소리": ("기타소음", "사람의비언어적소리"),
        "02.동물및자연물소리": ("기타소음", "동물및자연물소리"),
        "03.전자제품및생활환경소리": ("기타소음", "전자제품및생활환경소리"),
        "04.기타소리": ("기타소음", "기타소리"),
    },
    "06.복합소음": {
        "01.사람의비언어적소리": ("복합소음", "사람의비언어적소리"),
        "02.동물및자연물소리": ("복합소음", "동물및자연물소리"),
        "03.전자제품및생활환경소리": ("복합소음", "전자제품및생활환경소리"),
        "04.기타소리": ("복합소음", "기타소리"),
    }
}

# [Manual Example] 사람이 하나씩 입력한다면 아래와 같은 형식으로 실행하게 됩니다:
# uv run python3 scripts/manage_db.py noise --path "data/noise/136-2.극한 소음 환경 소리 데이터/TS_06.복합소음_01.사람의비언어적소리" --dataset "136-2.극한 소음 환경 소리 데이터" --category "복합소음" --sub "사람의비언어적소리"

for cat_folder, sub_dict in noise_mapping.items():
    for sub_folder, (cat_label, sub_label) in sub_dict.items():
        # TS와 VS 폴더를 순회하며 인덱싱
        for prefix in ["TS", "VS"]:
            folder_name = f"{prefix}_{cat_folder}_{sub_folder}"
            path = os.path.join(noise_root, folder_name)
            
            if os.path.exists(path):
                # 번호가 없는 순수 한글 라벨만 전달
                run_cmd(f'uv run python3 scripts/manage_db.py noise --path "{path}" --dataset "136-2.극한 소음 환경 소리 데이터" --category "{cat_label}" --sub "{sub_label}"')

# 4. RIR Indexing
run_cmd('uv run python3 scripts/manage_db.py rir --path "data/rirs" --dataset "Generated_RIR" --split "train"')

# 5. Split Reallocation (8:1:1)
run_cmd('uv run python3 scripts/manage_db.py realloc --type speech --ratio 0.8 0.1 0.1')
run_cmd('uv run python3 scripts/manage_db.py realloc --type noise --ratio 0.8 0.1 0.1')
run_cmd('uv run python3 scripts/manage_db.py realloc --type rir --ratio 0.8 0.1 0.1')

print("Indexing and Reallocation completed.")
