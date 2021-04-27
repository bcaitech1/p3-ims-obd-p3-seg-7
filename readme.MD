# Pstage 03 재활용 품목 분류를 위한 Semantic Segmentation

### 개요
환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 우리나라의 분리 수거율은 굉장히 높은 것으로 알려져 있고, 또 최근 이러한 쓰레기 문제가 주목받으며 더욱 많은 사람이 분리수거에 동참하려 하고 있습니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

### 코드 설명
* inference.py
    * 메인 infernece 파일입니다.

* load_data.py
    * 모델을 train하고 infernece하는데 필요한 dataloader를 제공합니다.

* model.py
    * Semantic Segmentation을 수행할 수 있는 모델들의 모음 파일입니다.
    * FCN8s / DeconvNet / SegNet / Deeplab_V3_Resnet101

* train.py
    * 메인 training 파일입니다.

* utils.py
    * 학습에 필요한 유틸리티 함수들을 모아둔 파일입니다.


### 실행 방법
* 학습
```console
$ python train.py
```

* 추론
```console
$ python inference.py
```