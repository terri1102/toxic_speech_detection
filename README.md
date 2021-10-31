## Table of Contents

- [👿 Toxic Speech Detection](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#23)
  * [🅱️ 프로젝트 배경](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#30)
  * [✅ 프로젝트 목표](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#36)
  * [📁 데이터셋 선정](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#48)
  * [🧹 데이터 전처리](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#54)
  * [🤖 모델 선정](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#61)
  * [🍱 BentoML로 로컬에서 서빙](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#67)
  * [🎯 목표 달성도](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#75)
  * [추후 과제 및 한계](https://github.com/terri1102/toxic_speech_detection/tree/master/README.md#86)
- [Quick Start](#quick-start)
  * [Build datasets](#build-datasets)
  * [Finetune a Bert model](#finetune-a-bert-model)
  * [Make a prediction](#make-a-prediction)
  * [Evaluate your model](#evaluate-your-model)


---



# 👿 Toxic Speech Detection
<img src="https://github.com/terri1102/terri1102.github.io/blob/master/assets/images/no-hate-2019922_1920.jpg?raw=true" alt="no_hate" style="zoom:67%;" />

챗봇에서 사용할 수 있는 앞 문장을 고려한 혐오발언 검출 모델입니다. 가벼운 모델을 사용해서 CPU inference time을 줄이고자 했으며, bentoML를 통해서 로컬에서 서빙이 가능합니다.



## 🅱️ 프로젝트 배경

여러 인터넷 커뮤니티와 챗봇의 혐오 발언을 검출하기 위한 노력이 많았지만, 기존의 Hate Speech Classification은 문장 하나만을 고려해서 분류를 하고 있어서, 앞 문장의 맥락에 따라 혐오 발언인 문장은 검출해 내지 못했습니다. 따라서 저는 앞 문장의 혐오발언에 그저 동의하기만 하는 발언도 혐오발언으로 분류해내기 위한 맥락을 고려하는 혐오 발언 분류모델 개발을 기획하게 되었습니다.



## ✅ 프로젝트 목표

- [ ] 클래스 비율이 비슷한 데이터셋 구축

- [ ] Sentence pair를 input으로 하고 문맥을 고려할 수 있는 모델

- [ ] 빠른 CPU Inference Time 

- [ ] BentoML로 서빙에 용이한 형태로 저장

  

## 📁 데이터셋 선정

여러 논문 리뷰와 스터디를 통해서 문맥이 있는 대화형의 혐오발언 분류 데이터셋을 찾아보려 했습니다. 대화형인 데이터셋은 Toxicity detection: Does context really matter?의 wiki dataset이 가장 이상적인 형태여서 전체를 사용하였습니다. 하지만 클래스 비율이 Non-toxic한 comment가 압도적으로 많아서 다른 데이터셋을 통해서 Toxic한 샘플들을 보강하고자 하였습니다. 그 결과 아래와 같은 최종 데이터셋으로 구축하여 프로젝트를 진행하였습니다. 



## 🧹 데이터 전처리

제가 사용한 데이터셋들 중 Wiki Dataset을 제외하고는 대부분 인터넷 커뮤니티의 커멘트를 그대로 가져온 raw data여서 Emoji, 웹주소, @user 태그 등이 포함되어 있었습니다. 처음에 전처리를 할 때는 emoji는 demoji 라이브러리를 이용해 text변환, @username와 웹주소는 각각 USER, URL로 변환했습니다.
Emoji가 들어갔을 때 어감이 달라지는 문장이 있었기 때문에, 변환해서 사용하는 것이 적절하다고 생각했으나, 대화형 데이터를 concat해서 사용하기 때문에 각 시퀀스의 길이를 줄이기 위해서 최대한 문장을 짧게 만들려고 하다보니 결국에는 emoji, username, url을 모두 제거해서 plain text를 만들어서 사용했습니다.



## 🤖 모델 선정

분류 모델로는 DistilBERT와 AlBERT를 선정하여 fine-tuning하여 결과를 비교했습니다.



## 🍱 BentoML로 로컬에서 서빙

BentoML을 이용하여 Swagger UI 형태로 로컬 서빙하였으며, json 형태로 predict이 가능하게 하였습니다.

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/df1ee646-69f2-4560-9cf1-edffa58cc46f/bentoml_demo.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210821%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210821T065106Z&X-Amz-Expires=86400&X-Amz-Signature=81200d458a714324f0fc8f807145b3c83b0e170625b88e2fb0fdee462c8f4757&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22bentoml_demo.gif%22)



## 🎯 목표 달성도

- [x]  클래스 비율이 비슷한 데이터셋 구축
- [x]  Sentence pair를 input으로 하고 문맥을 고려할 수 있는 모델
- [ ]  CPU Inference Time 50ms 미만
- [x]  BentoML로 서빙에 용이한 형태로 저장
- Inference Time: tokenizing 과정이 너무 오래 걸리는 것 같아서 tokenizerFast로 바꾸어서 했더니 1초 정도 감소하였습니다. 그럼에도 1문장 예측시 걸리는 시간이 200ms 정도여서 이 목표는 달성하지 못했습니다. 앞으로의 경량화를 위해
- 모델 성능의 경우 정확도 0.99, f1 score 0.99, rocauc 0.98 정도로 아주 높게 나왔으나, 욕설이 들어간 문장의 경우일 때만 제대로 분류하였기에 추후에 데이터셋을 더 다양하게 구성해서 성능을 개선할 수 있을 것 같습니다.



## 추후 과제 및 한계

- 한계로는 앞 문장에 혐오 문장이 있거나 뒷 문장에 혐오 문장이 있거나 둘 중 하나라도 혐오 문장이 들어가면 타겟 문장을 혐오발언으로 분류한다는 문제가 있었습니다.

- 최적화



# Quick Start

## Build datasets
전처리와 커스텀 데이터셋 구성 및 pytorch data_loader 형태로 저장하는 모듈입니다. `$data` 디렉토리 안에 csv 파일을 넣고 훈련시킬 수 있습니다.
```bash
$ python build_dataset.py 
```

`--path_to_data` : data 파일이 들어 있는 디렉토리입니다. 기본은 `$data`로 설정되어 있습니다.

`--max_seq_length` : 최대 시퀀스 길이를 설정합니다. 512

`--batch_size` : 배치 사이즈를 설정합니다. 기본은 16으로 설정되어 있습니다. 

`--bert_model` : Dataset를 빌드할 때 미세 조정할 때 사용할 BERT 모델에 맞는 Tokenizer를 사용합니다. 기본은 albert-base-v2로 되어있으며 현재 패키지에서 사용가능한 모델은 'albert-base-v2'와 'distilbert-base' 모델입니다. 



## Finetune a Bert model

```
$ python run_training.py
```

`--bert_model` : fine-tuning 과정에서 사용할 BERT 모델을 설정합니다. 기본은 albert-base-v2로 되어있으며, 현재 패키지에서 사용가능한 모델은 'albert-base-v2'와 'distilbert-base' 모델입니다. 

`--freeze_bert` : BERT의 레이어를 freeze하고 훈련할지, 모든 레이어를 훈련할지 설정합니다. 기본은 False로 되어 있습니다.

`--epochs` : 훈련할 epoch를 설정합니다. 기본은 4입니다.

`--lr` : learning rate를 설정합니다. 기본은 2e-5입니다.

`--iters_to_accumulate`: 기본은 2입니다.



## Make a prediction

`build_dataset.py`에서 만든 test data_loader를 이용해서 예측을 진행합니다. 
```bash
$ python run_prediction.py
```



## Evaluate your model

`build_dataset.py`를 통해 split한 test 파일과 예측 결과를 f1 score와 accuracy로 평가합니다.
```bash
$ python run_evaluation.py
```

