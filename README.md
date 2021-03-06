## Table of Contents

- [๐ฟ Toxic Speech Detection](#-toxic-speech-detection)
  * [๐ฑ๏ธ ํ๋ก์ ํธ ๋ฐฐ๊ฒฝ](#user-content-๐ฑ๏ธ-ํ๋ก์ ํธ-๋ฐฐ๊ฒฝ)
  * [โ ํ๋ก์ ํธ ๋ชฉํ](#user-content--ํ๋ก์ ํธ-๋ชฉํ)
  * [๐ ๋ฐ์ดํฐ์ ์ ์ ](#user-content--๋ฐ์ดํฐ์-์ ์ )
  * [๐งน ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ](#user-content--๋ฐ์ดํฐ-์ ์ฒ๋ฆฌ)
  * [๐ค ๋ชจ๋ธ ์ ์ ](#user-content--๋ชจ๋ธ-์ ์ )
  * [๐ฑ BentoML๋ก ๋ก์ปฌ์์ ์๋น](#user-content--bentoml๋ก-๋ก์ปฌ์์-์๋น)
  * [๐ฏ ๋ชฉํ ๋ฌ์ฑ๋](#user-content--๋ชฉํ-๋ฌ์ฑ๋)
  * [์ถํ ๊ณผ์  ๋ฐ ํ๊ณ](#user-content-์ถํ-๊ณผ์ -๋ฐ-ํ๊ณ)
- [Quick Start](#quick-start)
  * [Build datasets](#build-datasets)
  * [Finetune a Bert model](#finetune-a-bert-model)
  * [Make a prediction](#make-a-prediction)
  * [Evaluate your model](#evaluate-your-model)


---



# ๐ฟ Toxic Speech Detection
<img src="https://github.com/terri1102/terri1102.github.io/blob/master/assets/images/no-hate-2019922_1920.jpg?raw=true" alt="no_hate" style="zoom:67%;" />

์ฑ๋ด์์ ์ฌ์ฉํ  ์ ์๋ ์ ๋ฌธ์ฅ์ ๊ณ ๋ คํ ํ์ค๋ฐ์ธ ๊ฒ์ถ ๋ชจ๋ธ์๋๋ค. ๊ฐ๋ฒผ์ด ๋ชจ๋ธ์ ์ฌ์ฉํด์ CPU inference time์ ์ค์ด๊ณ ์ ํ์ผ๋ฉฐ, bentoML๋ฅผ ํตํด์ ๋ก์ปฌ์์ ์๋น์ด ๊ฐ๋ฅํฉ๋๋ค.



## ๐ฑ๏ธ ํ๋ก์ ํธ ๋ฐฐ๊ฒฝ

์ฌ๋ฌ ์ธํฐ๋ท ์ปค๋ฎค๋ํฐ์ ์ฑ๋ด์ ํ์ค ๋ฐ์ธ์ ๊ฒ์ถํ๊ธฐ ์ํ ๋ธ๋ ฅ์ด ๋ง์์ง๋ง, ๊ธฐ์กด์ Hate Speech Classification์ ๋ฌธ์ฅ ํ๋๋ง์ ๊ณ ๋ คํด์ ๋ถ๋ฅ๋ฅผ ํ๊ณ  ์์ด์, ์ ๋ฌธ์ฅ์ ๋งฅ๋ฝ์ ๋ฐ๋ผ ํ์ค ๋ฐ์ธ์ธ ๋ฌธ์ฅ์ ๊ฒ์ถํด ๋ด์ง ๋ชปํ์ต๋๋ค. ๋ฐ๋ผ์ ์ ๋ ์ ๋ฌธ์ฅ์ ํ์ค๋ฐ์ธ์ ๊ทธ์  ๋์ํ๊ธฐ๋ง ํ๋ ๋ฐ์ธ๋ ํ์ค๋ฐ์ธ์ผ๋ก ๋ถ๋ฅํด๋ด๊ธฐ ์ํ ๋งฅ๋ฝ์ ๊ณ ๋ คํ๋ ํ์ค ๋ฐ์ธ ๋ถ๋ฅ๋ชจ๋ธ ๊ฐ๋ฐ์ ๊ธฐํํ๊ฒ ๋์์ต๋๋ค.



## โ ํ๋ก์ ํธ ๋ชฉํ

- [ ] ํด๋์ค ๋น์จ์ด ๋น์ทํ ๋ฐ์ดํฐ์ ๊ตฌ์ถ

- [ ] Sentence pair๋ฅผ input์ผ๋ก ํ๊ณ  ๋ฌธ๋งฅ์ ๊ณ ๋ คํ  ์ ์๋ ๋ชจ๋ธ

- [ ] ๋น ๋ฅธ CPU Inference Time 

- [ ] BentoML๋ก ์๋น์ ์ฉ์ดํ ํํ๋ก ์ ์ฅ

  

## ๐ ๋ฐ์ดํฐ์ ์ ์ 

์ฌ๋ฌ ๋ผ๋ฌธ ๋ฆฌ๋ทฐ์ ์คํฐ๋๋ฅผ ํตํด์ ๋ฌธ๋งฅ์ด ์๋ ๋ํํ์ ํ์ค๋ฐ์ธ ๋ถ๋ฅ ๋ฐ์ดํฐ์์ ์ฐพ์๋ณด๋ ค ํ์ต๋๋ค. ๋ํํ์ธ ๋ฐ์ดํฐ์์ Toxicity detection: Does context really matter?์ wiki dataset์ด ๊ฐ์ฅ ์ด์์ ์ธ ํํ์ฌ์ ์ ์ฒด๋ฅผ ์ฌ์ฉํ์์ต๋๋ค. ํ์ง๋ง ํด๋์ค ๋น์จ์ด Non-toxicํ comment๊ฐ ์๋์ ์ผ๋ก ๋ง์์ ๋ค๋ฅธ ๋ฐ์ดํฐ์์ ํตํด์ Toxicํ ์ํ๋ค์ ๋ณด๊ฐํ๊ณ ์ ํ์์ต๋๋ค. ๊ทธ ๊ฒฐ๊ณผ ์๋์ ๊ฐ์ ์ต์ข ๋ฐ์ดํฐ์์ผ๋ก ๊ตฌ์ถํ์ฌ ํ๋ก์ ํธ๋ฅผ ์งํํ์์ต๋๋ค. 



## ๐งน ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ

์ ๊ฐ ์ฌ์ฉํ ๋ฐ์ดํฐ์๋ค ์ค Wiki Dataset์ ์ ์ธํ๊ณ ๋ ๋๋ถ๋ถ ์ธํฐ๋ท ์ปค๋ฎค๋ํฐ์ ์ปค๋ฉํธ๋ฅผ ๊ทธ๋๋ก ๊ฐ์ ธ์จ raw data์ฌ์ Emoji, ์น์ฃผ์, @user ํ๊ทธ ๋ฑ์ด ํฌํจ๋์ด ์์์ต๋๋ค. ์ฒ์์ ์ ์ฒ๋ฆฌ๋ฅผ ํ  ๋๋ emoji๋ demoji ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ด์ฉํด text๋ณํ, @username์ ์น์ฃผ์๋ ๊ฐ๊ฐ USER, URL๋ก ๋ณํํ์ต๋๋ค.
Emoji๊ฐ ๋ค์ด๊ฐ์ ๋ ์ด๊ฐ์ด ๋ฌ๋ผ์ง๋ ๋ฌธ์ฅ์ด ์์๊ธฐ ๋๋ฌธ์, ๋ณํํด์ ์ฌ์ฉํ๋ ๊ฒ์ด ์ ์ ํ๋ค๊ณ  ์๊ฐํ์ผ๋, ๋ํํ ๋ฐ์ดํฐ๋ฅผ concatํด์ ์ฌ์ฉํ๊ธฐ ๋๋ฌธ์ ๊ฐ ์ํ์ค์ ๊ธธ์ด๋ฅผ ์ค์ด๊ธฐ ์ํด์ ์ต๋ํ ๋ฌธ์ฅ์ ์งง๊ฒ ๋ง๋ค๋ ค๊ณ  ํ๋ค๋ณด๋ ๊ฒฐ๊ตญ์๋ emoji, username, url์ ๋ชจ๋ ์ ๊ฑฐํด์ plain text๋ฅผ ๋ง๋ค์ด์ ์ฌ์ฉํ์ต๋๋ค.



## ๐ค ๋ชจ๋ธ ์ ์ 

๋ถ๋ฅ ๋ชจ๋ธ๋ก๋ DistilBERT์ AlBERT๋ฅผ ์ ์ ํ์ฌ fine-tuningํ์ฌ ๊ฒฐ๊ณผ๋ฅผ ๋น๊ตํ์ต๋๋ค.



## ๐ฑ BentoML๋ก ๋ก์ปฌ์์ ์๋น

BentoML์ ์ด์ฉํ์ฌ Swagger UI ํํ๋ก ๋ก์ปฌ ์๋นํ์์ผ๋ฉฐ, json ํํ๋ก predict์ด ๊ฐ๋ฅํ๊ฒ ํ์์ต๋๋ค.

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/df1ee646-69f2-4560-9cf1-edffa58cc46f/bentoml_demo.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210821%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210821T065106Z&X-Amz-Expires=86400&X-Amz-Signature=81200d458a714324f0fc8f807145b3c83b0e170625b88e2fb0fdee462c8f4757&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22bentoml_demo.gif%22)



## ๐ฏ ๋ชฉํ ๋ฌ์ฑ๋

- [x]  ํด๋์ค ๋น์จ์ด ๋น์ทํ ๋ฐ์ดํฐ์ ๊ตฌ์ถ
- [x]  Sentence pair๋ฅผ input์ผ๋ก ํ๊ณ  ๋ฌธ๋งฅ์ ๊ณ ๋ คํ  ์ ์๋ ๋ชจ๋ธ
- [ ]  CPU Inference Time 50ms ๋ฏธ๋ง
- [x]  BentoML๋ก ์๋น์ ์ฉ์ดํ ํํ๋ก ์ ์ฅ
- Inference Time: tokenizing ๊ณผ์ ์ด ๋๋ฌด ์ค๋ ๊ฑธ๋ฆฌ๋ ๊ฒ ๊ฐ์์ tokenizerFast๋ก ๋ฐ๊พธ์ด์ ํ๋๋ 1์ด ์ ๋ ๊ฐ์ํ์์ต๋๋ค. ๊ทธ๋ผ์๋ 1๋ฌธ์ฅ ์์ธก์ ๊ฑธ๋ฆฌ๋ ์๊ฐ์ด 200ms ์ ๋์ฌ์ ์ด ๋ชฉํ๋ ๋ฌ์ฑํ์ง ๋ชปํ์ต๋๋ค. ์์ผ๋ก์ ๊ฒฝ๋ํ๋ฅผ ์ํด
- ๋ชจ๋ธ ์ฑ๋ฅ์ ๊ฒฝ์ฐ ์ ํ๋ 0.99, f1 score 0.99, rocauc 0.98 ์ ๋๋ก ์์ฃผ ๋๊ฒ ๋์์ผ๋, ์์ค์ด ๋ค์ด๊ฐ ๋ฌธ์ฅ์ ๊ฒฝ์ฐ์ผ ๋๋ง ์ ๋๋ก ๋ถ๋ฅํ์๊ธฐ์ ์ถํ์ ๋ฐ์ดํฐ์์ ๋ ๋ค์ํ๊ฒ ๊ตฌ์ฑํด์ ์ฑ๋ฅ์ ๊ฐ์ ํ  ์ ์์ ๊ฒ ๊ฐ์ต๋๋ค.



## ์ถํ ๊ณผ์  ๋ฐ ํ๊ณ

- ํ๊ณ๋ก๋ ์ ๋ฌธ์ฅ์ ํ์ค ๋ฌธ์ฅ์ด ์๊ฑฐ๋ ๋ท ๋ฌธ์ฅ์ ํ์ค ๋ฌธ์ฅ์ด ์๊ฑฐ๋ ๋ ์ค ํ๋๋ผ๋ ํ์ค ๋ฌธ์ฅ์ด ๋ค์ด๊ฐ๋ฉด ํ๊ฒ ๋ฌธ์ฅ์ ํ์ค๋ฐ์ธ์ผ๋ก ๋ถ๋ฅํ๋ค๋ ๋ฌธ์ ๊ฐ ์์์ต๋๋ค.

- ์ต์ ํ



# Quick Start

## Build datasets
์ ์ฒ๋ฆฌ์ ์ปค์คํ ๋ฐ์ดํฐ์ ๊ตฌ์ฑ ๋ฐ pytorch data_loader ํํ๋ก ์ ์ฅํ๋ ๋ชจ๋์๋๋ค. `$data` ๋๋ ํ ๋ฆฌ ์์ csv ํ์ผ์ ๋ฃ๊ณ  ํ๋ จ์ํฌ ์ ์์ต๋๋ค.
```bash
python build_dataset.py 
```

`--path_to_data` : data ํ์ผ์ด ๋ค์ด ์๋ ๋๋ ํ ๋ฆฌ์๋๋ค. ๊ธฐ๋ณธ์ `$data`๋ก ์ค์ ๋์ด ์์ต๋๋ค.

`--max_seq_length` : ์ต๋ ์ํ์ค ๊ธธ์ด๋ฅผ ์ค์ ํฉ๋๋ค. 512

`--batch_size` : ๋ฐฐ์น ์ฌ์ด์ฆ๋ฅผ ์ค์ ํฉ๋๋ค. ๊ธฐ๋ณธ์ 16์ผ๋ก ์ค์ ๋์ด ์์ต๋๋ค. 

`--bert_model` : Dataset๋ฅผ ๋น๋ํ  ๋ ๋ฏธ์ธ ์กฐ์ ํ  ๋ ์ฌ์ฉํ  BERT ๋ชจ๋ธ์ ๋ง๋ Tokenizer๋ฅผ ์ฌ์ฉํฉ๋๋ค. ๊ธฐ๋ณธ์ albert-base-v2๋ก ๋์ด์์ผ๋ฉฐ ํ์ฌ ํจํค์ง์์ ์ฌ์ฉ๊ฐ๋ฅํ ๋ชจ๋ธ์ 'albert-base-v2'์ 'distilbert-base' ๋ชจ๋ธ์๋๋ค. 



## Finetune a Bert model

```
python run_training.py
```

`--bert_model` : fine-tuning ๊ณผ์ ์์ ์ฌ์ฉํ  BERT ๋ชจ๋ธ์ ์ค์ ํฉ๋๋ค. ๊ธฐ๋ณธ์ albert-base-v2๋ก ๋์ด์์ผ๋ฉฐ, ํ์ฌ ํจํค์ง์์ ์ฌ์ฉ๊ฐ๋ฅํ ๋ชจ๋ธ์ 'albert-base-v2'์ 'distilbert-base' ๋ชจ๋ธ์๋๋ค. 

`--freeze_bert` : BERT์ ๋ ์ด์ด๋ฅผ freezeํ๊ณ  ํ๋ จํ ์ง, ๋ชจ๋  ๋ ์ด์ด๋ฅผ ํ๋ จํ ์ง ์ค์ ํฉ๋๋ค. ๊ธฐ๋ณธ์ False๋ก ๋์ด ์์ต๋๋ค.

`--epochs` : ํ๋ จํ  epoch๋ฅผ ์ค์ ํฉ๋๋ค. ๊ธฐ๋ณธ์ 4์๋๋ค.

`--lr` : learning rate๋ฅผ ์ค์ ํฉ๋๋ค. ๊ธฐ๋ณธ์ 2e-5์๋๋ค.

`--iters_to_accumulate`: ๊ธฐ๋ณธ์ 2์๋๋ค.



## Make a prediction

`build_dataset.py`์์ ๋ง๋  test data_loader๋ฅผ ์ด์ฉํด์ ์์ธก์ ์งํํฉ๋๋ค. 
```bash
python run_prediction.py
```



## Evaluate your model

`build_dataset.py`๋ฅผ ํตํด splitํ test ํ์ผ๊ณผ ์์ธก ๊ฒฐ๊ณผ๋ฅผ f1 score์ accuracy๋ก ํ๊ฐํฉ๋๋ค.
```bash
python run_evaluation.py
```

