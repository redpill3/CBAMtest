# CBAMtest
프로젝트를 클론한후 'train'폴더를 생성하고 train 폴더 안에 'nv2.zip'과 '학습데이터.zip'을  압축풀어 
'akiec' 'bcc' 'bkl' 'df' 'mel' 'nv' 'vasc' 폴더가 위치하도록 한 후 
여러 모델들의 스크립트를 실행하면 학습이 시작됩니다.

## Classification 작업
cnn(densenet)+cbam.py : 실행하면 Segmentation fine-tunning 없이 바로 densenet+cbam 모델을 가지고 분류 학습 시작

## Segmentation 작업

