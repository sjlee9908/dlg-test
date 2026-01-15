### Deep Leakage from Gradients (DLG) PyTorch Implementation

이 저장소는 Deep Leakage from Gradients (DLG) 논문의 아이디어를 PyTorch로 구현한 프로젝트입니다. Federated Learning과 같이 Gradient만 공유되는 환경에서, Gradient를 역공학하여 원본 이미지와 라벨을 복원하는 공격 및 방어 실험을 수행할 수 있습니다.

✨ Features
다양한 모델 지원: LeNet, ResNet, VGG, FFN (Feed Forward Network)

방어 기법 실험: Gradient Noise 추가 및 연산 Precision(float16/32/64) 조절 가능

유연한 설정: config.yaml을 통해 배치 사이즈, 최적화 알고리즘(LBFGS, Adam 등), 반복 횟수 등을 쉽게 변경

결과 시각화: 원본(Original), 초기값(Initial), 복원된 이미지(Final) 비교 및 유사도 측정

🛠 Prerequisites
이 프로젝트는 Python 3.x 및 PyTorch 환경이 필요합니다.

```Bash
pip install torch torchvision numpy matplotlib pillow omegaconf pyyaml
```

🚀 Usage
main.py를 실행하여 실험을 시작합니다. --scenario 옵션으로 config.yaml에 정의된 실험 설정을 선택할 수 있습니다.

기본 실행
```Bash
python main.py --scenario org
```
주요 시나리오 예시
config.yaml에 정의된 다양한 시나리오를 실행할 수 있습니다.

모델 변경: resnet, vggnet, ffn

배치 사이즈 변경: batch_2, batch_4, batch_8

방어 기법 (Noise): noise_1 (Level 1), noise_2 (Level 2)

방어 기법 (Precision): quant_16_16 (Float16)

```Bash
# ResNet 모델 복원 실험
python main.py --scenario resnet
# 노이즈가 추가된 Gradient 복원 실험
python main.py --scenario noise_1
```

⚙️ Configuration
config.yaml 파일에서 세부 파라미터를 수정할 수 있습니다.

```YAML
scenario_name:
  model: lenet          # 대상 모델 (lenet, resnet, vggnet 등)
  data:
    batch_size: 1       # 복원할 이미지 배치 크기
    idx: 3845           # 데이터셋 인덱스
  dlg:
    optim: LBFGS        # DLG 최적화 알고리즘
    iter: 300           # 공격 반복 횟수
  client:
    noise_level: 0      # 방어: 노이즈 레벨 (0=없음)
    precision: float32  # 방어: 연산 정밀도
```

📂 Project Structure
```Bash
.
├── config.yaml           # 실험 설정 파일
├── main.py               # 메인 실행 파일
├── utils.py              # 유틸리티 (데이터 로드, 시각화 등)
├── dlg/
│   ├── dlg_runner.py     # DLG 공격(복원) 로직
│   └── client_runner.py  # 클라이언트(Gradient 계산 및 방어) 로직
├── models/               # 모델 아키텍처 (LeNet, ResNet, VGG, FFN)
└── result/               # 결과 이미지 및 로그 저장소
```

📊 Results
실행 결과는 result/ 폴더에 저장됩니다.

이미지 (.png): Original vs Initial vs Final (복원 결과) 비교 이미지

로그 (.txt): 원본과 복원 이미지 간의 Perceptual Similarity (Cosine Similarity) 점수
