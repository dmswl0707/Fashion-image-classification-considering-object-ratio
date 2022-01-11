# Fashion-image-classification-considering-object-ratio

## 프로젝트 개요도
![system architecture](https://user-images.githubusercontent.com/65028694/147312559-f4a08946-901c-4b15-b775-0e6cfe6297e5.jpg)
- 딥러닝 기술을 적용한 패션 이미지 인식 시스템은 이미지 검색을 기반으로 한 추천시스템과 상품 분류 등 다양한 플랫폼에 적용되고 있으며. 패션 이미지 객체 인식 기술에서 있어 객체의 스타일과, 재질, 사이즈 등의 요소는 패션 이미지 분류에 있어 중요한 요소로 평가된다. 
- 본 프로젝트는 CNN(Convolution Neural Network) 학습에서, 다양한 해상도의 입력 이미지가 정해진 사이즈의 입력 이미지로 맞춰지는 과정에서 발생하는 픽셀의 왜곡현상을 해결하고, 패션 이미지 객체의 비율 왜곡을 방지함으로써 패션 이미지 오분류를 개선한다.
- 다양한 해상도의 크기를 가지는 의류 이미지에서 픽셀의 왜곡 현상을 해결하기 위해, 가상의 픽셀을 할당하여 주어진 해상도의 종횡비를 유지하는 Ratio Preservation Module을 제안한다.
- 성능 비교를 위해, Pretrained ResNet50, DenseNet201을 사용하여, Ratio Presevation module을 적용한 데이터셋을 학습하여 수행하고 기존의 CNN 신경망 학습방식과 비교하였다. 그 결과, Ratio Preservation Module을 제안하여 전체 정확도를 약 2.5% 향상하였으며, FPR와 FNR를 감소시켜 클래스간 오분류를 감소시켰다.

## Ratio preservation method
CNN 학습 시, 고정된 사이즈로 리사이즈한 후 입력이미지로 넣어주는 과정에서 기존의 객체 정보를 소실시키는 픽셀 생략 또는 보간법을 통한 객체 이미지의 변형이 일어남.  
![픽셀왜곡](https://user-images.githubusercontent.com/65028694/148916432-a53cbb66-fa7f-495d-9108-19e51fb62edb.png)
이미지의 각 해상도가 가지는 종횡비의Maximum 기준에 따라 크기를 설정한 후, 원본 이미지와 Maximum 값의 차이를 가상의 픽셀(제로 패딩 또는 테두리 외삽법)을 통해 메워 입력 이미지를 생성한다.  
<center><img src="https://user-images.githubusercontent.com/65028694/147313306-b318e8fe-66c1-48cf-979c-405bc9bc0c6d.png" width="700" height="400"></center>

