## Semantic Segmentation   


<br>

- ### semantic segmentation의 목적      
  이미지 (input image)내에 있는 물체들을 의미 있는 단위로 분리하는 것.(클래스별로 구분)    

 
- ### FCN   
  기존의 AlexNet, VGGNet등의 네트워크는 convolution 층과 fully connected 층으로 이루어져 있다.
  따라서, 이미지 파일을 학습하고 예측해낼 수 있지만, 해당 object가 어느 위치에 있는지는 알 수 없다. (fully connected 층에서 위치 정보를 잃는다.)   
  
  이러한 문제를 해결하기 위해, 기존의 네트워크에서 fully connected 층들을 1x1 convolution 으로 대체한다. (<a href="https://github.com/jiyun1006/deeplearning-pytorch/blob/main/1x1convolution.md">1x1 convolution</a>)   
  네트워크 전체가 convolution 층이 됐고, 1x1 convolution으로 이미지 크기에 제한을 받지 않는다.   
  
  여러 convolution 층들을 지나 생성되는 feature map으로 클래스를 구분할 수 있다.   
  
  이후에, feature map을 원래 이미지 크기로 키우는 과정을 거친다. 이 과정을 upsampling이라 한다.   
  upsampling을 거친 feature map을 종합해서 최종적인 segmentation map을 만든다.   
 
 
 
