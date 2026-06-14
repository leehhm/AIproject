# 🏦 Bank of Korea (BOK) Policy Credibility Analyzer
> **한국은행 기준금리 방향성 예측 및 통화정책 신뢰도 분석 모델**

## 📌 프로젝트 개요
본 프로젝트는 자연어 처리(NLP)와 시계열 예측(Time-Series Forecasting) 모델을 결합하여 한국은행 금융통화위원회의 기준금리 결정 방향성(인상, 동결, 인하)을 예측하고, 시장의 예측과 실제 결정 간의 괴리를 분석하여 **통화정책의 신뢰도(Policy Credibility)**를 지수화 및 시각화하는 프로젝트입니다.

## 🚀 주요 기능
* **의사록 텍스트 분석 (NLP):** `BERT` 모델을 활용하여 텍스트 데이터(금통위 의사록 등)에서 기준금리 인상/동결/인하 압력을 확률로 추출합니다.
* **시계열 예측 (Time-Series):** `Amazon Chronos-Bolt` 기초 모델을 사용하여 과거 데이터를 기반으로 롤링 예측(Rolling Prediction)을 수행합니다.
* **정책 신뢰도 지수 산출:** 실제 금리 결정과 모델 예측값의 차이를 기반으로 'Policy Surprise(정책 서프라이즈)'를 계산하고, 이를 종합 지수(Composite Index) 및 신뢰도 등급(Credibility Grade)으로 환산합니다.
* **대시보드 시각화:** 분석된 정책 신뢰도 트렌드와 기준금리 변동 내역을 한눈에 파악할 수 있는 대시보드를 생성합니다.

## 📂 파일 구성
* **`bert.py`**
  * 한국은행 기준금리 방향성 분석기 (v6)
  * HuggingFace `BertForSequenceClassification`을 사용한 텍스트 분류 모델 구현.
  * 불균형 데이터 해소를 위한 전환점 당월 오버샘플링(Oversampling) 및 연속성 보정 알고리즘 포함.
* **`chronos-bolt.py`**
  * `amazon/chronos-bolt-base` 모델을 활용한 시계열 예측 및 신뢰도 지수 산출 스크립트.
  * 텍스트 분석 결과(`monthly_results.json`)와 실제 기준금리 데이터를 결합하여 최종 신뢰도 평가.
* **`bok_labels.csv`**
  * 모델 학습 및 평가를 위한 한국은행 기준금리 결정 이력 데이터셋 (2016년~).
  * 컬럼: `연월`, `결정(인하/동결/인상)`, `기준금리`.
* **`credibility_dashboard.jpg`**
  * 시계열 모델링 및 BERT 분석 결과를 종합하여 출력한 정책 신뢰도 대시보드 시각화 예시 결과물.

## ⚙️ 요구 사항 (Prerequisites)
프로젝트를 실행하기 위해 아래의 파이썬 라이브러리들이 필요합니다.

```bash
pip install torch transformers pandas numpy matplotlib pdfplumber
```
*(주의: Chronos-Bolt 구동을 위해 추가적인 라이브러리 설치가 필요할 수 있습니다.)*

## 💻 실행 방법 (Usage)

**1. 방향성 확률 추출 (BERT Model)**
`bok_labels.csv`와 원본 PDF 텍스트 데이터를 기반으로 확률을 추출합니다.
```bash
python bert.py
```
*(실행 결과로 `monthly_results.json`이 생성됩니다.)*

**2. 신뢰도 지수 산출 (Chronos-Bolt Model)**
생성된 JSON 파일과 과거 기준금리 데이터를 병합하여 정책 신뢰도를 계산합니다.
```bash
python chronos-bolt.py
```
*(실행 결과로 터미널에 연월별 정책 서프라이즈 수치와 신뢰도 등급이 출력되며, 대시보드 이미지가 생성됩니다.)*

## 📊 결과물 예시
`credibility_dashboard.jpg`를 통해 시간에 따른 한국은행의 정책 신뢰도 변동 추이와 기준금리 결정 내역을 직관적으로 확인할 수 있습니다. 데이터 상 신뢰도가 하락하고 정책 서프라이즈가 발생한 지점의 심층적인 분석이 가능합니다.
