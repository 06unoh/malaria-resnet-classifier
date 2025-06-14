# ResNet모델을 활용한 Malaria 감염 검출



## 🔹 프로젝트 구조

```
datasets/
    malaria_dataset.py
models/
    custom_resnet.py
utils/
    data_utils.py
    train.py
    test.py
    transforms.py
    unzip.py
    visualize.py
main.py
requirements.txt
Dockerfile
```
---
## 🔹 실행법 (for Local PC)

### ☝️ Requirements 설치

```
pip install -r requirements.txt
```

### ✌️ 실행

```
python main.py
```
---
## 🔹 도커 실행법 (for Docker User)

### ☝️ 도커 이미지 빌드

```
docker build -t malaria-det-resnet .
```

### ✌️ 컨테이너 실행

```
docker run --rm malaria-det-resnet
```

---

## 🔹 데이터셋

CIFAR-10 
CIFAR-10은 10개의 클래스(비행기, 자동차, 새, 고양이 등)에 대한 60,000장의 32x32 사이즈의 이미지로 구성된 데이터셋입니다.

데이터셋은 별도의 준비 없이, 실행 시 `torchvision` 라이브러리를 통해 자동으로 다운로드됩니다.

---

## 🔹 결과

40회 에포크 학습 후, 정확도:

```
Validation Accuracy: 80.52% 
Test Accuracy: 79.81%
```

예측 결과 예시:

![샘플 예측 결과](images/prediction.png)

---
06unoh