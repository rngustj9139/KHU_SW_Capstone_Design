# AutoEncoder, Two Stage Detector, DeepSORT, ReID, UNet 기반 지능형 교통체계 구축 프로젝트

## 요약

지능형 교통시스템(ITS, Intelligent Transportation Systems)이란 여러 교통 문제들을 효율적으로 해결하기
위해 정보통신기술(ICT)을 적용한 교통 시스템을 의미한다. 본 연구에서는 딥러닝 기술과 대용량 도로
CCTV 데이터를 이용하여 차량 탐지 및 추적, 사고 및 이상 운전 탐지 등의 기능을 구현한 지능형 교통
시스템을 제안한다. 해당 시스템은 UNet 기반의 Semantic Segmentation 모델을 통해 도로 영역의 크기를
정밀하게 측정하고, 이를 FasterRCNN 모델을 통해 탐지된 차량 객체의 수로 나눔으로써 교통 혼잡도
지수(TCI)를 계산한다. 또한 교통 사고나 급발진, 음주운전, 졸음운전에 대한 비정형 데이터 부족 문제를
해결하기 위해 영상 감시분야에서 사용되는 비지도 학습 모델인 AutoEncoder를 도입하여 도로상의 이상
상황을 탐지하고, 예기치 않은 사고 발생 시 교통 관리 당국이 신속하게 대응할 수 있도록 지원한다. 이에
더불어 다양한 기상환경에서 본 프로젝트의 시스템을 적용할 수 있도록 Two Stage Object Detector의 학습
과정에서 다양한 데이터 증강(Data Augmentation) 기법을 적용하였고 이를 통해 모델의 Generalization
Performance를 증가시킬 수 있었다.


## 1. 서론

### 1.1 연구배경

#### 1.1.1 교통 혼잡 비용 문제
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/eea99c93-8363-4485-8cf7-051e8a19dc35)

“교통혼잡비용”이란 도로 혼잡으로 인해 발생하는 경제적 손실을 의미한다. 교통혼잡비용에는 교통혼잡에 따른 연료소모, 시간지연, 교통사고, 대기오염 등이 포함되는데, 우리나라의 교통혼잡비용 조사는 한국교통 연구원에서 이루어지고 있으며, 지난 1991년 한국개발연구원(KDI)에서 첫 연구작업이 시작되었다. 대한민국의 연도별 교통혼잡비용은 2022년 약 72조원으로 추정되며(한국교통연구원, 2023.07.02, 보도자료) 매년 약 10%가량 증가하는 추세를 보인다. 이는 한국 명목 GDP의 약 3%가량을 점유하는데 이를 통해 대한민국은 해외 주요 국과 비교했을 때 GDP 대비 교통혼잡비용이 높은 수준인 것을 유추할 수 있다. (텍사스, Urban Mobility, 2019 인릭스 Global Traffic Scorecard) 교통 혼잡에 대한 비용 문제는 미국과 EU에서도 떠오르고 있다. 2019년 미국 국가 교통 안전 보드 (NHTSA)는 당해 년도 미국에서 교통 체증으로 인해 약 3억 1,200만 갤런의 연료가 낭비되었 다 밝혔으며, 이를 통해 약 121억 달러의 비용 손실이 발생하였다는 것을 확인할 수 있다. 또한 당해 년도 교통 체증으로 인한 평균 손실 시간은 개인당 약 99시간에 달한다고 분석되었으며, 유럽연합에서는 교통 체 증으로 인한 시간 손실이 GDP의 약 1.0%에 해당하는 경제적 손실을 발생시킨다고 분석하였다. 이러한 높은 교통 혼잡은 긴급차량의 이동 제약을 발생시키며 이로 인해 긴급차량이 사고 현장에 도달하는 데 걸리는 시간을 늦춰 빠른 사고 대응을 방해하는 문제가 발생한다.

#### 1.1.2 도로 혼잡도 측정 방식의 신뢰도 및 안전 문제
현재 도로의 실시간 혼잡도 계산은 공압식 고무 튜브(매설형 차량 검지기) 및 GPS를 활용하여 이루어지고 있다. 하지만 이러한 방법이 항상 신뢰할 수 있는 것은 아니다. 예를 들어 2020년 베를린에 거주하는 한 예술가는 손수레와 99개의 중고 전화기를 이용해 슈프레 강을 가로지르는 주요 다리 중 하나에서 "가상" 교통 체증을 만들어낸 사례가 존재한다. 이후 Google 지도는 해당 지역을 매우 혼잡한 지역으로 표시하였으며 위와 같은 사례를 제외하고도 도로 혼잡도 추정치를 늘리기 위해 다른 유형의 데이터가 종종 사용된다. 여기에는 과거 패턴, 센서 데이터, 예정된 폐쇄에 대한 지자체 피드 및 사용자가 보고한 사건이 포함되는데, 이를 통해 가장 신뢰할 수 있는 상호 참조 중 하나는 교통 카메라에서 수집된 혼잡도를 실시간으로 시각적으로 인식하는 것임을 알 수 있다. 또한 공압식 고무 튜브 카운터는 쉽고 널리 사용되는 솔루션이지만 아래와 같은 여러 리스크가 존재한다.

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/3e1181ce-460d-4d32-b5ff-5f4dff42b322)

#### 1.1.3 도로 CCTV 데이터를 활용한 이상운전(e.g. 음주운전, 졸음운전, 급발진) 탐지 모델의 저조한 등장과 발전
음주운전이나 졸음운전, 급발진 등의 이상운전을 감지하기 위한 모델에 관한 논문은 아래의 예시와 같이 비전기술 및 센서 데이터를 기반으로 하여 다양한 연구 분야에서 다루어지고 있다.

- "Real-time Detection of Driver Fatigue Using Computer Vision - A Review" (IEEE, 2019) 비전 기반의 기술을 사용하여 운전자의 피로를 실시간으로 감지하는 방법에 대한 논문
- "Detection of Drunk Driving from Driving Patterns Using Support Vector Machines" (International Journal of Innovative Research in Computer and Communication Engineering, 2016) Support Vector Machines을 활용하여 운전 패턴에서 음주운전을 감지하는 방법에 대한 논문
- "Driver Fatigue and Drowsiness Monitoring System Based on Steering Wheel Dynamics" (IEEE Transactions on Intelligent Transportation Systems, 2016) 운전자의 조향 휠 다이내믹스를 기반으로 운전자의 피로와 졸음을 모니터링하는 시스템에 대한 논문
- "Detection of Aggressive Driving Behavior Using Smartphones: An Exploratory Study" (IEEE Transactions on Intelligent Transportation Systems, 2017) 스마트폰을 사용하여 급발진과 같은 공격적인 운전 행동을 감지하는 방법에 대한 논문
- "Detection of Drunk Drivers Based on Driving Performance: A Review of Existing Research Methods" (Forensic Science International, 2016) 운전 성능을 기반으로 음주운전자를 감지하는 기존 연구 방법에 대한 논문

하지만 이상 운전을 식별하기 위한 컴퓨터 비전 모델은 대부분 운전자의 모습을 3인칭으로 촬영하여 얼굴의 특징 점을 분석한 뒤 이상 여부를 예측 하는 기법을 많이 이용하고 있는데, 지능형 CCTV를 통해 차량의 주행 모습을 3인칭으로 확인하여 이상운전을 판단하는 모델은 등장과 발전이 저조한 상태이다. 따라서 위와 같은 이유들로 인해 도로 CCTV 영상 데이터 및 컴퓨터 비전, 특히 딥러닝 기술을 이용한 지능형 교통체계에 대한 필요성이 증대 되어 가고 있는 상황이다.

### 1.2 연구 목표
본 연구는 대용량 도로 CCTV 데이터 및 딥러닝 기술을 이용해 지능형 교통 시스템을 구축하고 일반 운전자나 지자체, 정부 유관 부서에게 제공하여 정확한 교통 혼잡도를 추정할 수 있게 함으로써 교통 혼잡도를 줄이는 프로세스에 직∙간접적으로 도움을 주고, 환경 오염 감소 및 신속한 사고 대응을 지원하는 것에 기여하는 것을 목표로 한다. 본 프로젝트의 지능형 교통 시스템은 Two Stage Detector(FasterRCNN)을 통해 차량 객체를 탐지하고 탐지된 객체의 수를 UNet 기반 Semantic Segmentation을 통해 도출 된 도로의 면적(Pixel의 개수)으로 나눔으로써 도로 혼잡 지수(TCI)를 계산한다. 본 연구에서 채택한 교통 혼잡 지수 도출 공식은 아래와 같다.

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/26405299-4853-431d-a1b4-d60a8d77e4bc)

또한 도출 된 교통 혼잡 지수(TCI)는 아래와 같은 기준으로 분류된다.

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/ebffe544-1e77-44c2-b4ce-7ffb4212783a)


이에 더불어 정상 데이터만을 이용해 학습하는 비지도 학습(Unsupervised Learning) 모델인 AutoEncoder를 도입하여 도로상의 이상운전(e.g. 음주운전, 졸음운전, 급발진) 및 교통 사고 여부를 식별하는 기능을 시스템에 추가함으로써 교통 사고 및 이상운전에 대한 데이터 부족 문제를 극복하는 것을 목표로 삼는다. 이상 운전 및 교통 사고 식별에 대한 대부분의 비전 연구에서는 운전자의 안면을 3인칭으로 확인하여 이상 탐지를 수행하지만 본 연구에서는 차량의 주행 모습을 3인칭으로 확인하여 이상 탐지를 수행함으로써 등장과 발전이 저조한 도로 CCTV 데이터 기반 이상 탐지 모델 구축에 선행연구로서 초석을 다질 수 있다. 또한 구축 된 지능형 교통 시스템에 DeepSORT 모델을 적용하여 Multi Object Tracking 기능을 구현하고, 뚜렷하게 구분되지 않는 특징을 갖는 차량 객체에 대해 ReID 모델을 적용 하였을 때와 적용하지 않았을 때의 MOT 성능(MOTA, MOTP) 비교 연구를 수행 할 것이다. 만약 유의미한 성능 향상이 발견될 시 본 연구는 ReID 모델의 잠재성을 증명하고, 범용적인 이용과 발전에 대한 새로운 연구 방향을 제시할 수 있다.

## 2. 배경 지식 및 관련 연구

### 2.1 주요 기술 및 모델

#### 2.1.1 FasterRCNN (For Object Detection)
Faster R-CNN(Faster Region-based Convolutional Neural Networks)은 딥러닝을 이용한 객체 검출(object detection) 알고리즘이다 객체 검출은 이미지에서 다양한 객체를 찾아내고 그 위치를 특정하는 작업을 의미하는데, Faster R-CNN은 이러한 객체 검출 문제를 효과적으로 해결하기 위해 개발된 모델로, 특히 높은 정확도를 갖는다는 장점을 가지고 있다. Faster R-CNN의 작동 원리는 다음과 같다.

1. 기본 이미지 특징 추출: 먼저, 입력 이미지를 Convolutional Neural Network(CNN)에 통과시켜 기본적인 이미지 특징(Feature Map)을 추출한다. 일반적으로 ResNet, VGG16 등과 같은 사전 학습된 네트워크가 사용된다.
2. Region Proposal Network (RPN): 추출된 특징 맵을 이용하여 객체가 있을 가능성이 높은 영역(Region Proposal)을 제안한다. RPN은 CNN의 특징 맵을 입력으로 받아서 객체가 있을 법한 영역들을 추정한다.
3. RoI Pooling: 제안된 영역들을 고정된 크기로 변환하여 이후 단계에서 동일한 크기로 처리할 수 있게 한다. 이 단계에서 RoI(Region of Interest) Pooling이 사용된다.
4. 객체 분류 및 위치 조정: 변환된 영역들은 Fully Connected Layer를 거쳐지고, 객체의 클래스와 정확한 바운딩 박스를 예측할 수 있게 된다.

#### 2.1.2 DeepSORT (For Multi Object Tracking)
DeepSORT(Deep Simple Online and Realtime Tracking)는 객체 추적 알고리즘 중 하나이며, SORT(Simple Online and Realtime Tracking)를 개선한 버전으로, 객체 검출 알고리즘과 결합하여 비디오에서 물체를 추적하는 데 사용된다. 특히 사람이나 차량 등의 움직이는 객체를 실시간으로 추적하는 데 강력한 성능을 발휘하는 특징을 갖는다. DeepSORT의 주요 작동 방식은 다음과 같다.

1. 객체 검출: 먼저, 각 프레임마다 객체 검출 알고리즘(e.g. YOLO, Faster R-CNN)을 사용하여 현재 프레임에서 객체를 내고 이후 객체의 위치와 크기를 나타내는 바운딩 박스를 얻는다.
2. Kalman 필터: 객체의 움직임을 예측하고 추적하는 데 사용된다. 이전 프레임의 정보를 바탕으로 현재 프레임에서의 객체 위치를 예측한다.
3. 연관성 매칭: 현재 프레임의 검출 결과와 이전 프레임의 추적 결과를 연결한다. 이를 위해 Hungarian 알고리즘을 사용하여 가장 적합한 매칭을 찾는다.
4. 재식별 정보 사용(ReID): DeepSORT는 객체의 시각적 특징을 활용하여 동일 객체를 식별한다. 이를 통해 객체가 일시적으로 검출되지 않더라도 지속적으로 추적할 수 있다.

DeepSORT는 SORT의 단순한 칼만 필터와 Hungarian 알고리즘을 기반으로 하되, 추가적으로 객체의 시각적 특징을 고려하여 더욱 정확한 추적을 가능하게 하는데, 이러한 이유로 자율 주행, 영상 감시 시스템, 스포츠 분석 등 다양한 분야에서 많이 활용되고 있다.

#### 2.1.3 AutoEncoder (For Anomaly Detection)
AutoEncoder는 인공 신경망의 일종으로, 주로 데이터의 차원 축소와 특징 추출을 위해 사용된다. AutoEncoder는 입력 데이터를 압축된 저차원 표현으로 변환한 후, 이를 다시 원래 데이터로 복원하는 과정을 통해 학습하는데, 이러한 과정에서 데이터의 중요한 특징을 추출할 수 있다. AutoEncoder의 기본 구성 요소는 다음과 같다.

1. 인코더(Encoder): 입력 데이터를 저차원 잠재 공간(latent space)으로 변환한다. 인코더는 여러 개의 신경망 층으로 구성될 수 있으며, 입력 데이터의 중요한 특징을 압축하여 잠재 벡터(latent vector)로 표현한다.
2. 디코더(Decoder): 잠재 벡터를 다시 원래의 데이터 형태로 복원한다. 디코더는 인코더의 반대 과정으로, 잠재 벡터를 입력으로 받아 원래의 데이터와 유사한 출력을 생성한다.

AutoEncoder의 학습 과정은 입력 데이터를 다시 출력으로 복원하는 동안 손실 함수(loss function)를 최소화하는 방향으로 이루어진다. 손실 함수는 일반적으로 입력 데이터와 출력 데이터 간의 차이를 측정하는 재구성 오류(reconstruction error)를 사용한다.

#### 2.1.4 UNet (For Segmentation)
U-Net은 주로 이미지 분할 작업에 사용되는 딥러닝 모델이다. U-Net은 이미지의 각 픽셀이 특정 클래스에 속하는지 여부를 예측하여 이미지를 분할하는 데 특화되어 있으며, 이름에서 알 수 있듯이 U-Net은 U자 형태의 네트워크 구조를 가지고 있다. U-Net의 주요 특징과 구조는 다음과 같다.

1. 인코딩 경로(Contracting Path):
    - 컨볼루션 블록: 여러 개의 컨볼루션 레이어를 통해 입력 이미지에서 점점 더 높은 수준의 특징을 추출한다. 각 컨볼루션 레이어는 활성화 함수(ReLU)와 함께 적용된다.
    - 맥스 풀링(Max Pooling): 특징 맵의 크기를 줄여서 공간적 차원을 축소하고, 중요한 특징만 남긴다. 이 과정을 통해 네트워크는 더 큰 맥락 정보를 학습할 수 있다.
2. 디코딩 경로(Expanding Path):
    - 업샘플링(Upsampling): 인코딩 경로에서 축소된 특징 맵을 다시 원래의 크기로 복원한다. 업샘플링은 주로 디컨볼루션(deconvolution) 또는 업컨볼루션(upconvolution) 레이어를 통해 이루어진다.
    - 컨볼루션 블록: 업샘플링된 특징 맵을 보다 세밀하게 조정하여 높은 해상도의 특징 맵을 생성한다.
3. 스킵 연결(Skip Connections): 인코딩 경로에서 추출된 특징 맵을 디코딩 경로의 대응되는 단계로 직접 연결한다. 이는 원래의 세부 정보를 보존하면서 업샘플링 과정에서 손실된 정보를 복원하는 데 도움을 준다. 스킵 연결은 네트워크의 학습 효율성을 높이고 성능을 향상시킨다.

### 2.2 기존 연구 문제점 및 해결 방안

#### 2.2.1 야간 시간대의 낮은 현장 조도로 인한 낮은 차량 검출 정확도 문제 극복
- Data Augmentation 적용: Brightness, Contrastm Saturation, Hue distortion 기법을 적용하여 다양한 기상환경 및
시간대에서도 FasterRCNN의 Generalization Performance를 증가시켰다. 하지만 MMDetection은 파이프라인 내부에서 모델의
학습과 평가가 이루어지기 때문에 Augmentation이 적용 된 이후의 정확한 Datasets의 Size를 식별하기 불가능하다는 추가적인
Issue가 발생하였다.
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/3f03d5d2-a8e5-4b1d-8251-4fd4ae4e7d6a)


#### 2.2.2 Object Detector가 비디오 이미지 스트림을 처리하는 경우 Occlusion(폐색) 및 Id-Switching 발생 문제
- MOT를 위한 DeepSORT 모델 도입: 사람이나 차량 등의 움직이는 객체를 실시간으로 추적하는 데 강력한 성능을 발휘하는 DeepSORT 모델을 도입하여 Occlustion(폐색) 및 Id-Switching 문제를 해결.

#### 2.2.3 도로 상의 이상운전 및 사고에 대한 데이터 부족 문제 & 주행 차량을 3인칭으로 확인하여 이상여부를 탐지하는 모델의 등장과 발전의 저조함
- AutoEncoder 도입: AutoEncoder 를 사용하면 데이터를 라벨링하지 않아도(모든 데이터가 정상 데이터라고 간주)
데이터의 주성분이 되는 정상 영역의 특징들을 학습 할 수 있다. 이 때 학습된 AutoEncoder 에 정상 데이터를 넣어주면 정상
상태를 잘 복원하기 때문에 입력 및 출력의 차이가 거의 발생하지 않는 반면, 비정상 데이터를 넣게 되면 AutoEncoder 는
결과물을 정상 데이터 처럼 복원하지 않기 때문에 비정상 데이터를 검출 할 수 있게 된다. (원본 데이터와 복원된 데이터
사이의 재구성 손실(Reconstruction Loss)을 최소화 하는 방향으로 학습이 진행 된다)

## 3. 추진 내용

### 3.1 전체 시스템 구성

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/f337d227-a761-49f1-be00-c3d5e28129ed)


### 3.2 핵심 기능 개발 및 테스트

#### 3.2.1 FasterRCNN (For Object Detection)
- AI-Hub의 교통문제 해결을 위한 CCTV 교통 영상(고속도로)을 이용하였다. 해당 데이터 셋에서는 이미지(2800x2100 해상도)
내에 일정 크기 이하의 소형 객체(200x200 픽셀 크기 이하)들만 존재하며 이미지에 대한 xml 형태의 어노테이션 파일 또한
포함하고 있다
   ![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/aad788d5-aac2-40e9-b466-af9ca5adf7a3)

- Training Model
 모델은 에포크 13회부터 Stable한 Performance를 나타내기 시작하였음
  <br />
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/a83df687-3189-4d4e-bff0-36ca3dca3209)
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/c665cbf7-28f0-4c5f-b2d9-cba99c5a70a6)
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/28c74272-7649-411c-9961-60bd9cb5ecce)
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/6f1e3d7e-cead-4efd-a436-07e883375878)
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/325074fe-2990-4be7-81c5-1e30f287e862)

Training Datasets에 대한 최종 mAP: 0.7470
Validation Datasets에 대한 최종 mAP: 0.7474





#### 3.2.2 ReID(For Multi Object Tracking)
이전 FasterRCNN과는 달리 학습 초반에도 Stable한 Performance를 도출하기 위해 Warm-Up 기법을 적용하였다. 딥러닝 모델을 훈련할 때 학습률(learning rate)을 조절하기 위한 주요 기법 중 하나인 워밍업(warm-up)이란 초기 학습 단계에서 학습률을 작게 시작하고, 이후 점진적, 역동적으로 변화시켜 최적의 학습률로 수렴하게 하는 과정이다. 이는 초기에는 학습이 불안정할 수 있으므로 학습률을 작게 설정하여 안정적으로 시작하고, 학습이 진행됨에 따라 학습률을 증가시켜 더 나은 결과를 얻을 수 있도록 도와준다.

### → 최종 triplet_loss: 0.0000

  Triplet Loss는 어떤 한 객체(Anchor)와 같은 객체(Positive), 다른 객체(Negative)이라는 파라미터를 이용해 학습 시 미니 배치 안에서 Anchor, Positive, Negative들이 임베딩 된 값들의 유클리드 거리를 구해 아래와 같은 Loss 함수를 만든다.
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/06853291-f6d8-4463-8a17-19ad3b5d61b2)

  대괄호 안의 첫번째 항이 의미하는 것은 Anchor와 Positive간의 Distance고, 두번째 항은 Anchor와 Negative와의 Distance이며 α는 마진(Hyper Parameter)을 의미한다. 따라서 L을 최소화한다는 것은 Positive와의 거리는 가까워지도록 하고 Negative와의 거리는 멀어지도록 하는 것이다. 즉 Triplet Loss가 0.000이 나왔다는 것은 ReID 모델이 학습 중에 모든 Anchor, Positive, Negative 쌍에 대해 거리를 올바르게 구별했다는 의미이며 Anchor와 Positive 간의 거리가 Negative 간의 거리보다 작거나 같게 학습되었다는 것을 나타낸다. 즉 이는 모델이 이미 학습 데이터에서 제시된 유사성 및 차이를 이해하고 있음을 시사할 수 있다.

### → 최종 top-1 accuracy: 100.0000


Top-1 Accuracy란 Softmax Activation Function에서의 Output에서 제일 높은 수치를 가지는 값이 정답일 경우에 대한 지표를 계산한 것을 의미한다. 즉 Top-1 Accuracy가 100.0000과 같이 높은 수치가 나왔다는 것은 해당 ReID 모델이 대부분의 경우에 대해 가장 높은 확률을 가진 클래스를 정확하게 식별한다는 의미이며 이는 모델이 주어진 분류 작업을 잘 수행하고 있음을 나타낸다.

#### 3.2.3 DeepSORT (For Multi Object Tracking)

MMTracking에서 제공하는 DeepSORT의 경우 Object Detection Model 과 ReID Model을 혼합하여 MOT을 수행할 수 있었다. Detection Model의 경우 구축한 Faster-RCNN Checkpoint 파일에 Tracking Video에 대한 전이 학습을 수행한 뒤 적용하였다. ReID(Re-Identification) Model의 경우 사용되는 데이터셋의 객체 간의 구별되는 특징이 없는 경우, 객체의 식별이 어려워지고 성능이 제한될 수 있다. 즉 객체 간의 차이가 충분히 크지 않거나 유의미한 특징이 부족하면 다중 객체에 대한 정확한 식별과 추적이 어려워지고 Generalization Performance의 저하를 초래할 수 있다. 따라서 ReID Model을 효과적으로 학습시키기 위해서는 데이터셋이 객체 간의 유의미하고 구별되는 특징을 포함하고 있어야하는데  학습 데이터는 사람과 같이 구별되는 특징을 가진 객체가 포함되지 않았기 때문에 ReID Model을 DeepSORT에 적용하였을 때 눈에 띄는 성능의 향상을 불러올 수 있을지 의문을 가지게 되었다. 그래서 ReID Model을 DeepSORT에 적용했을 때와 적용하지 않았을 때의 성능 비교 연구를 수행하였고 아래와 같은 결과를 도출할 수 있었다. 


### → DeepSORT 성능 평가 (ReID 모델을 적용 했을 때): MOTA(100%), MOTP(0.090)

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/b5838975-8351-4574-92e3-0cbec8c447eb)


### → DeepSORT 성능 평가 (ReID 모델을 적용 하지 않았을 때): MOTA(99.3%), MOTP(0.091)

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/1a456ede-ed8b-47f8-8e1c-53ebd45f230a)

ReID 모델을 적용한 DeepSORT를 기반 MOT 수행시 MOTA(Multi Object Tracking Accuracy)성능이 0.7% 향상하며, MOTP(Multi Object Tracking Precision) 성능은 0.001% 향상한 것을 확인할 수 있었다 (벤치마크마다 MOTP를 표현하는 방식이 다름) 또한 Recall과 Precision값이 각각 0.0%, 0.7% 증가하였다.

#### 3.2.4 AutoEncoder (For Anomaly Detection)
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/98d9497e-b0c6-4b3c-a7f3-f1997ffc3a75)


#### 3.2.5 UNet (For Semantic Segmentation)
![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/ecf74014-6727-4a5a-ac67-0acd74ac8030)


### 3.3 이슈 및 대응
도로의 혼잡도를 측정하기 위해 일정 시간 동안 도로를 통과하는 차량의 수를 도로 용량으로 나누는 방식을 채택하였다. 이를 위해 프로젝트 초기 Hough 변환 기법을 이용하여 도로 위의 대표적인 두 차선을 추출하고 그 사이의 영역을 구하는 코드를 구현했으나, 직선 형태의 차선은 잘 검출되었지만 곡선 형태의 차선은 제대로 검출되지 않는 문제가 발생하였다. 또한, 도로로 추정되는 관심영역(ROI)을 하드코딩으로 설정해야 하는 어려움도 존재하였는데, 이러한 문제를 극복하고자, Hough 변환 대신 Semantic Segmentation 모델을 도입하여 도로 영역을 추출하고, 이를 통해 보다 정확한 도로 용량을 측정하는 방식을 시도하였다.

## 4. 결과 (Inference)

### 4.1 FasterRCNN
Inference 결과, 안개가 낀 기상 환경 속에서도 높은 Confidence로 차량 객체들을 잘 Detection하는 모습을 보임
단일 이미지에 대한 Inference 결과

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/de1466a9-e2cc-4ed2-ae8b-c9d3d8275132)


Video에 대한 Inference 결과



![제목 없는 동영상 - Clipchamp로 제작 (2) (1)](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/4cee1cdc-edb6-4a3c-9306-a58b2d700e95)










### 4.2 DeepSORT
Video에 대한 Inference 결과



![제목 없는 동영상 - Clipchamp로 제작 (1) (1)](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/47e4aa6b-05e2-4641-96a0-d2e875ad3b7e)







### 4.3 AutoEncoder
AutoEncoder Inference 결과

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/fef28d2f-c69a-44c6-8183-b91ed527c8e3)


### 4.4 UNet
UNet Inference 결과

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/59803e35-42d7-48f6-a7ab-327ef66d94e6)

### 4.5 Final Product Inference
Final Product Inference 결과

![image](https://github.com/rngustj9139/KHU_SW_Capstone_Design/assets/43543906/fb23f503-9733-44e1-86ba-280286353319)


## 5. 결론
본 연구에서는 딥러닝 기술과 대용량 도로 CCTV 데이터를 이용한 지능형 교통 시스템을 구축하여 교통 혼잡도를 실시간으로 분석하고 이상 운전을 탐지하는 모델을 제안하였다. FasterRCNN, UNet, DeepSORT, AutoEncoder 등의 다양한 딥러닝 모델을 결합하여 교통 상황을 정밀하게 파악하고, 사고 및 이상 운전 상황에 신속히 대응할 수 있는 시스템을 구현하였다. 이를 통해 다음과 같은 결론을 도출할 수 있다:

### 5.1 정확한 교통 혼잡도 분석
UNet 기반의 도로 영역 분할과 FasterRCNN을 활용한 차량 객체 탐지를 통해 도로의 교통 혼잡도를 정확히 측정할 수 있었다. 이로 인해 교통 혼잡도 지수(TCI)를 실시간으로 계산하여 교통 흐름을 효율적으로 관리할 수 있게 되었다.

### 5.2 이상 운전 및 사고 탐지
AutoEncoder 모델을 도입하여 도로상의 이상 운전(e.g. 음주운전, 졸음운전, 급발진) 및 교통 사고를 효과적으로 탐지할 수 있었다. 이를 통해 교통 관리 당국이 비정상 상황 발생 시 신속히 대응할 수 있는 기반을 마련하였다.

### 5.3 다중 객체 추적
DeepSORT 모델을 적용하여 다중 객체 추적 성능을 개선하였으며, ReID 모델을 추가함으로써 차량의 재식별 능력을 향상시켰다. 이를 통해 보다 정교한 교통 상황 모니터링과 분석이 가능해졌다.

### 5.4 데이터 증강을 통한 모델 일반화 성능 향상
다양한 기상환경 및 시간대에서도 안정적인 성능을 유지하기 위해 데이터 증강 기법을 적용하였다. 이로 인해 모델의 일반화 성능이 향상되어 실제 환경에서도 높은 정확도를 보였다.

### 5.5 지속 가능한 교통 관리 시스템 구축
본 연구의 결과를 바탕으로, 교통 혼잡 문제를 해결하고 교통 사고를 줄이는 데 기여할 수 있는 지속 가능한 지능형 교통 관리 시스템을 구축할 수 있었다. 이는 환경 오염 감소 및 교통 안전성 향상에도 긍정적인 영향을 미칠 것이다.

본 프로젝트는 지능형 교통 시스템의 발전에 중요한 초석이 될 것이며, 향후 연구를 통해 더욱 고도화된 교통 관리 솔루션을 개발할 수 있을 것이다.

## 6. 참고문헌
1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28.
2. Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., & Meng, H. (2023). Strongsort: Make deepsort great again. IEEE Transactions on Multimedia.
3. Zhang, Y. (2018, March). A better autoencoder for image: Convolutional autoencoder. In ICONIP17-DCEC. Available online: [link](http://users.cecs.anu.edu.au/Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_58.pdf) (accessed on 23 March 2017).
4. Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., ... & Wu, J. (2020, May). Unet 3+: A full-scale connected unet for medical image segmentation. In ICASSP 2020-2020 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 1055-1059). IEEE.
5. Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2117-2125).
