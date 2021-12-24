# Fashion-image-classification-considering-object-ratio

딥러닝 기술을 적용한 패션 이미지 인식 시스템은 이미지 검색을 기반으로 한 추천시스템과 상품 분류 등 다양한 플랫폼에 적용되고 있으며. 패션 이미지 객체 인식 기술에서 있어 객체의 스타일과, 재질, 사이즈 등의 요소는 패션 이미지 분류에 있어 중요한 요소로 평가된다. 위 프로젝트에서는 CNN(Convolution Neural Network) 학습에서, 다양한 해상도의 입력 이미지가 정해진 사이즈의 입력 이미지로 맞춰지는 과정에서 발생하는 픽셀의 왜곡현상을 해결하기 위해 객체 비율 보존 기법을 적용하여 입력이미지가 가지는 해상도의 종횡비 보존을 통해 패션 이미지 분류 시스템의 성능을 향상시키고, FPR(False Positive Rate)와 FNR(Fasle Negative Rate)를 감소시킨다.

![system architecture](https://user-images.githubusercontent.com/65028694/147312559-f4a08946-901c-4b15-b775-0e6cfe6297e5.jpg)

### Ratio preservation method example
![image](https://user-images.githubusercontent.com/65028694/147313306-b318e8fe-66c1-48cf-979c-405bc9bc0c6d.png)
