import zipfile
import os

def unzip():
    zip_path1 = "./cell_image/Parasitized.zip"
    zip_path2 = "./cell_image/Uninfected.zip"

    # 압축 해제할 폴더 경로
    extract_path1 = "./cell_image/Parasitized"
    extract_path2 = "./cell_image/Uninfected"

    # ZIP 파일 열어서 압축 풀기
    # with zipfile.ZipFile([zip_path1,zip_path2], 'r') as zip_ref:
    if not os.path.exists(extract_path1):
        with zipfile.ZipFile(zip_path1, 'r') as zip_ref:
            zip_ref.extractall(extract_path1)
    
    if not os.path.exists(extract_path2):        
        with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
            zip_ref.extractall(extract_path2)
