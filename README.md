# 🏦 Bank of Korea Policy Credibility Analyzer
> **한국은행 기준금리 방향성 예측 및 통화정책 신뢰도 분석 파이프라인**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface&logoColor=white)
![Chronos](https://img.shields.io/badge/Amazon-Chronos_Bolt-FF9900)

## 📌 Project Overview
본 프로젝트는 **자연어 처리(NLP)**와 **최신 시계열 예측(Time-Series Forecasting)** 모델을 결합한 이중 파이프라인을 통해 한국은행 금융통화위원회의 기준금리 결정 방향성(인상, 동결, 인하)을 예측합니다. 

단순한 금리 예측을 넘어, 시장의 예측 모델과 실제 정책 결정 간의 괴리를 분석하고 이를 정량화하여 **'통화정책의 신뢰도(Policy Credibility)'를 종합 지수화 및 시각화**하는 것을 핵심 목표로 합니다.

## 🚀 Key Features
- **🧠 Text-driven Directional Analysis (NLP)**
  - `BERT` 기반 분류 모델을 활용하여 금통위 의사록 등 원본 텍스트에서 금리 조정 압력(인상/동결/인하 확률)을 정밀하게 추출합니다.
  - 불균형 데이터 해소를 위한 연속성 보정 및 전환점 당월 오버샘플링 알고리즘이 적용되었습니다.
- **📈 Advanced Time-Series Forecasting**
  - `Amazon/chronos-bolt-base` 파운데이션 모델을 활용한 롤링 예측(Rolling Prediction)을 수행합니다.
- **📊 Policy Credibility Indexing**
  - 모델의 예측값과 실제 금리 결정 간의 '정책 서프라이즈(Policy Surprise)'를 계산하여 최종적인 신뢰도 등급(Credibility Grade)을 도출합니다.
- **📉 Interactive Visualization**
  - 시계열 트렌드와 기준금리 변동 내역, 정책 신뢰도 하락 구간을 직관적으로 파악할 수 있는 종합 대시보드를 제공합니다.

## 📂 Repository Structure
```text
.
├── bert.py                  # BERT 기반 통화정책 방향성 예측 및 확률 추출 모듈 (v6)
├── chronos-bolt.py          # Chronos-Bolt 기반 시계열 예측 및 신뢰도 지수 산출 모듈
├── bok_labels.csv           # 한국은행 기준금리 결정 이력 데이터셋 (2016~)
├── credibility_dashboard.jpg # 통화정책 신뢰도 분석 종합 시각화 대시보드
└── README.md                # 프로젝트 명세서
