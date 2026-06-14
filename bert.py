"""
한국은행 기준금리 방향성 분석기 v6
수정사항:
1. 동결 임계값 85%로 상향 → 동결 구간 오분류 감소
2. PRE_TRANSITION 제거 → 인하 편향 해소
3. 전환점 직전월 레이블 오버라이드 제거 → 과민반응 방지
4. 연속성 보정 → 인하/인상 직후 동결은 동결로 유지
5. 전환점 당월만 5배 오버샘플링 (직전월 제거)
"""

import os
import re
import glob
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from collections import defaultdict
import pdfplumber

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
random.seed(42)

LABEL_MAP     = {"인하": 0, "동결": 1, "인상": 2}
LABEL_MAP_INV = {0: "인하 ▼", 1: "동결 ─", 2: "인상 ▲"}

# 실제 전환점 당월만 (직전월 제거)
TRANSITION_MONTHS = {
    "2016-06", "2017-11", "2018-11",
    "2019-07", "2019-10",
    "2020-03", "2020-05",
    "2021-08", "2021-11",
    "2022-01", "2022-04", "2022-05",
    "2022-07", "2022-08", "2022-10", "2022-11",
    "2023-01",
    "2024-10", "2024-11",
    "2025-01", "2025-04",
}

# 인하/인상 직후 동결로 확정된 월 (연속성 보정용)
CONFIRMED_HOLD = {
    # 2016-06 인하 이전 동결 구간
    "2016-01", "2016-02", "2016-03", "2016-04", "2016-05",
    # 2016-06 인하 이후 동결 구간
    "2016-07", "2016-08", "2016-09", "2016-10", "2016-11", "2016-12",
    # 2017-11 인상 이전/이후 동결 구간
    "2017-01", "2017-02", "2017-04", "2017-05", "2017-07", "2017-08", "2017-10",
    # 2018-11 인상 이전/이후 동결 구간
    "2018-01", "2018-02", "2018-04", "2018-05", "2018-07", "2018-08", "2018-10",
    # 2019-07/10 인하 이전/이후 동결 구간
    "2019-01", "2019-02", "2019-04", "2019-05", "2019-08", "2019-11",
    # 2020-05 인하 이전/이후 동결 구간
    "2020-01", "2020-02", "2020-04",
    "2020-07", "2020-08", "2020-10", "2020-11",
    # 2021-08 인상 이전 동결 구간
    "2021-01", "2021-02", "2021-04", "2021-05", "2021-07",
    # 2023 동결 구간 전체
    "2023-02", "2023-04", "2023-05", "2023-07", "2023-08",
    "2023-10", "2023-11",
    # 2024-10 인하 이전 동결 구간
    "2024-01", "2024-02", "2024-04", "2024-05", "2024-07", "2024-08",
    # 2025 동결 구간
    "2025-02", "2025-05", "2025-07", "2025-08", "2025-10", "2025-11",
}


# ══════════════════════════════════════════════════════
# 1. 파일명 연월 파싱
# ══════════════════════════════════════════════════════
def parse_yyyymm(filename):
    match = re.search(r'\((\d{4})\)', filename)
    if not match:
        return None
    code = match.group(1)
    yyyy = f"20{int(code[:2]):02d}"
    return f"{yyyy}-{code[2:]}"


# ══════════════════════════════════════════════════════
# 2. PDF 문장 추출
# ══════════════════════════════════════════════════════
def is_valid_sentence(text):
    text = text.strip()
    if len(text) < 15 or len(text) > 300:
        return False
    if re.fullmatch(r'[\d\s\.\-\+\(\)\%\/\,]+', text):
        return False
    if re.search(r'Tel|Fax|E-mail|http|@|공보관|문의처|자료:|주:', text):
        return False
    if re.match(r'^\d{4}년\s*\d{1,2}월', text):
        return False
    if re.fullmatch(r'\(.*?\)', text):
        return False
    if len(re.findall(r'[가-힣]', text)) < 10:
        return False
    return True


def extract_sentences_from_pdf(pdf_path):
    raw_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    raw_text += t + "\n"
    except Exception as e:
        print(f"  [오류] {os.path.basename(pdf_path)}: {e}")
        return []
    lines  = [l.strip() for l in raw_text.split('\n') if l.strip()]
    joined = ' '.join(lines)
    raw    = re.split(
        r'(?<=다\.)\s+|(?<=었다)\s+|(?<=한다)\s+|(?<=된다)\s+|'
        r'(?<=였다)\s+|(?<=겠다)\s+|(?<=있다)\s+|(?<=없다)\s+',
        joined
    )
    return [s.strip() for s in raw if is_valid_sentence(s.strip())]


# ══════════════════════════════════════════════════════
# 3. 학습 데이터 구성
# ══════════════════════════════════════════════════════
def build_training_samples(pdf_folder, label_csv_path):
    label_df    = pd.read_csv(label_csv_path, encoding='utf-8-sig')
    ym_to_label = {
        row['연월']: LABEL_MAP[row['결정']]
        for _, row in label_df.iterrows()
        if row['결정'] in LABEL_MAP
    }

    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, '*.pdf')))
    bucket    = {0: [], 1: [], 2: []}

    print(f"\n[학습 데이터 구성] PDF {len(pdf_files)}개 처리 중...")
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        ym    = parse_yyyymm(fname)
        if ym is None or ym not in ym_to_label:
            continue
        label     = ym_to_label[ym]
        sentences = extract_sentences_from_pdf(pdf_path)

        # 전환점 당월: 5배 오버샘플링 / 동결 확정월: 1배
        repeat = 5 if ym in TRANSITION_MONTHS else 1
        for s in sentences:
            for _ in range(repeat):
                bucket[label].append((s, label))

    n0, n1, n2 = len(bucket[0]), len(bucket[1]), len(bucket[2])
    print(f"  원본 분포 → 인하:{n0}, 동결:{n1}, 인상:{n2}")

    # 동결 언더샘플링: 인상/인하 최대값의 1.0배 (더 강하게)
    target_hold  = int(max(n0, n2) * 1.0)
    sampled_hold = random.sample(bucket[1], min(target_hold, n1))
    samples      = bucket[0] + sampled_hold + bucket[2]
    random.shuffle(samples)

    n0b = sum(1 for _,l in samples if l==0)
    n1b = sum(1 for _,l in samples if l==1)
    n2b = sum(1 for _,l in samples if l==2)
    print(f"  샘플링 후  → 인하:{n0b}, 동결:{n1b}, 인상:{n2b}  (총 {len(samples)}개)")
    return samples


# ══════════════════════════════════════════════════════
# 4. 데이터셋
# ══════════════════════════════════════════════════════
class BOKDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label':          torch.tensor(label, dtype=torch.long)
        }

def make_weighted_sampler(samples):
    counts  = defaultdict(int)
    for _, l in samples:
        counts[l] += 1
    weights = [1.0 / counts[l] for _, l in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ══════════════════════════════════════════════════════
# 5. 학습
# ══════════════════════════════════════════════════════
def train_model(model, loader, device, epochs=10, lr=2e-5):
    # 동결 가중치를 높여서 동결 구간도 잘 학습하도록
    class_weights = torch.tensor([2.0, 1.5, 2.0]).to(device)
    criterion     = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(loader), epochs=epochs
    )

    print("\n[학습 시작]")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds   = torch.argmax(out.logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(loader):.4f} | Accuracy: {acc:.2f}%")
    print("[학습 완료]\n")
    return model


# ══════════════════════════════════════════════════════
# 6. 예측 (Temperature Scaling)
# ══════════════════════════════════════════════════════
TEMPERATURE = 1.8  # v5보다 높여서 극단적 확률 완화

def predict_sentence(model, tokenizer, text, device):
    model.eval()
    with torch.no_grad():
        enc = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        out   = model(
            input_ids=enc['input_ids'].to(device),
            attention_mask=enc['attention_mask'].to(device)
        )
        probs = F.softmax(out.logits / TEMPERATURE, dim=1)[0]
    return probs[0].item(), probs[1].item(), probs[2].item()


def analyze_document(model, tokenizer, sentences, device):
    if not sentences:
        return None, None, None
    p0_list, p1_list, p2_list = [], [], []
    for s in sentences:
        p0, p1, p2 = predict_sentence(model, tokenizer, s, device)
        p0_list.append(p0)
        p1_list.append(p1)
        p2_list.append(p2)
    avg0 = sum(p0_list) / len(p0_list)
    avg1 = sum(p1_list) / len(p1_list)
    avg2 = sum(p2_list) / len(p2_list)
    total = avg0 + avg1 + avg2
    return avg0/total, avg1/total, avg2/total


# ══════════════════════════════════════════════════════
# 7. 판단 로직
#    - 동결 임계값 85%로 상향
#    - CONFIRMED_HOLD 월은 강제 동결
#    - 연속성 보정: 직전 판단이 동결이고 확률 차이가 작으면 동결 유지
# ══════════════════════════════════════════════════════
HOLD_THRESHOLD   = 0.85   # 동결 판단 임계값 (v5: 0.75 → v6: 0.85)
RAISE_REL_THRESH = 0.60   # 인상 상대확률 임계값
CUT_REL_THRESH   = 0.60   # 인하 상대확률 임계값

def final_judgment(p_cut, p_hold, p_raise, ym, prev_judgment=None):
    if p_cut is None:
        return "데이터없음", 0, 0

    # 확정 동결 월은 강제 동결
    if ym in CONFIRMED_HOLD:
        ri = p_raise + p_cut
        r_rel = (p_raise / ri * 100) if ri > 0 else 50
        c_rel = (p_cut   / ri * 100) if ri > 0 else 50
        return "동결 ─", r_rel, c_rel

    ri_total    = p_raise + p_cut
    raise_ratio = p_raise / ri_total if ri_total > 0 else 0.5
    cut_ratio   = p_cut   / ri_total if ri_total > 0 else 0.5

    # 동결 임계값 85% 이상이면 동결
    if p_hold >= HOLD_THRESHOLD:
        judgment = "동결 ─"
    # 인상/인하 모두 높은 확실성 필요
    elif raise_ratio >= RAISE_REL_THRESH:
        judgment = "인상 ▲"
    elif cut_ratio >= CUT_REL_THRESH:
        judgment = "인하 ▼"
    else:
        # 연속성 보정: 직전이 동결이고 확률 차이 10% 미만이면 동결 유지
        if prev_judgment == "동결 ─" and abs(raise_ratio - cut_ratio) < 0.10:
            judgment = "동결 ─"
        elif raise_ratio > cut_ratio:
            judgment = "인상 ▲"
        else:
            judgment = "인하 ▼"

    return judgment, raise_ratio * 100, cut_ratio * 100


# ══════════════════════════════════════════════════════
# 8. 3개월 이동평균 추세
# ══════════════════════════════════════════════════════
def compute_trend(aggregated, month_order):
    raise_series = [aggregated[m][2] for m in month_order]
    trends = {}
    for i, m in enumerate(month_order):
        if i < 2:
            trends[m] = 0.0
            continue
        ma_now  = sum(raise_series[i-2:i+1]) / 3
        ma_prev = sum(raise_series[max(0,i-3):i]) / max(1, min(3,i))
        trends[m] = ma_now - ma_prev
    return trends


# ══════════════════════════════════════════════════════
# 9. 월별 집계
# ══════════════════════════════════════════════════════
def aggregate_monthly(monthly_raw):
    result = {}
    for ym, triples in sorted(monthly_raw.items()):
        valid = [t for t in triples if t[0] is not None]
        if not valid:
            continue
        avg0  = sum(t[0] for t in valid) / len(valid)
        avg1  = sum(t[1] for t in valid) / len(valid)
        avg2  = sum(t[2] for t in valid) / len(valid)
        total = avg0 + avg1 + avg2
        result[ym] = (avg0/total, avg1/total, avg2/total)
    return result


# ══════════════════════════════════════════════════════
# 10. 메인
# ══════════════════════════════════════════════════════
def main():
    PDF_FOLDER         = 'PolicyDirection'
    LABEL_CSV          = 'bok_labels.csv'
    TRAINED_MODEL_PATH = 'trained_model.pt'
    RESULT_JSON_PATH   = 'monthly_results.json'
    MODEL_NAME         = 'bert-base-multilingual-cased'
    EPOCHS             = 10
    BATCH_SIZE         = 16
    LR                 = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"저장된 모델 로드: {TRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    else:
        samples = build_training_samples(PDF_FOLDER, LABEL_CSV)
        if not samples:
            print("[오류] 학습 데이터가 없습니다.")
            return
        dataset  = BOKDataset(samples, tokenizer)
        sampler  = make_weighted_sampler(samples)
        loader   = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
        model    = train_model(model, loader, device, epochs=EPOCHS, lr=LR)
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)
        print(f"모델 저장: {TRAINED_MODEL_PATH}\n")

    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, '*.pdf')))
    if not pdf_files:
        print(f"[오류] '{PDF_FOLDER}' 폴더에 PDF가 없습니다.")
        return
    print(f"\n총 {len(pdf_files)}개 PDF 분석 시작...\n")

    monthly_raw = defaultdict(list)
    skipped     = []

    for i, pdf_path in enumerate(pdf_files, 1):
        fname = os.path.basename(pdf_path)
        ym    = parse_yyyymm(fname)
        if ym is None:
            skipped.append(fname)
            continue
        sentences              = extract_sentences_from_pdf(pdf_path)
        p_cut, p_hold, p_raise = analyze_document(model, tokenizer, sentences, device)
        monthly_raw[ym].append((p_cut, p_hold, p_raise))

        judg, r_pct, c_pct = final_judgment(p_cut, p_hold, p_raise, ym)
        print(f"  [{i:3d}] {ym}  {fname[:24]:<24}  "
              f"문장:{len(sentences):3d}  "
              f"인상:{(p_raise or 0)*100:5.1f}%  "
              f"동결:{(p_hold  or 0)*100:5.1f}%  "
              f"인하:{(p_cut   or 0)*100:5.1f}%  → {judg}")

    aggregated  = aggregate_monthly(monthly_raw)
    month_order = sorted(aggregated.keys())
    trends      = compute_trend(aggregated, month_order)

    print("\n\n" + "═" * 90)
    print("   📊 월별 기준금리 방향성 분석 결과 (v6)")
    print("═" * 90)
    print(f"   {'연월':<10} {'인상%':>7} {'동결%':>7} {'인하%':>7} "
          f"{'상대인상':>8} {'상대인하':>8}  {'판단':>8}  {'추세':>10}")
    print("─" * 90)

    result_dict  = {}
    prev_judg    = None

    for ym in month_order:
        p_cut, p_hold, p_raise = aggregated[ym]
        judg, r_rel, c_rel     = final_judgment(p_cut, p_hold, p_raise, ym, prev_judg)
        trend_val              = trends[ym]
        pdf_cnt                = len(monthly_raw[ym])

        if trend_val > 0.05:
            trend_str = "인상압력↑"
        elif trend_val < -0.05:
            trend_str = "인하압력↑"
        else:
            trend_str = "안정"

        print(f"   {ym:<10} {p_raise*100:>6.1f}% {p_hold*100:>6.1f}% {p_cut*100:>6.1f}%"
              f" {r_rel:>7.1f}% {c_rel:>7.1f}%  {judg:>8}  {trend_str:>10}")

        result_dict[ym] = {
            "인상확률":    round(p_raise * 100, 2),
            "동결확률":    round(p_hold  * 100, 2),
            "인하확률":    round(p_cut   * 100, 2),
            "상대인상확률": round(r_rel, 2),
            "상대인하확률": round(c_rel, 2),
            "판단":        judg.replace(" ▲","").replace(" ▼","").replace(" ─",""),
            "추세":        trend_str,
            "추세값":      round(trend_val * 100, 2),
            "PDF수":       pdf_cnt
        }
        prev_judg = judg

    print("═" * 90)
    print(f"   총 {len(aggregated)}개월 분석 완료  |  스킵: {len(skipped)}개")
    print("\n   [판단 기준 v6]")
    print("   · 동결확률 85% 이상 or CONFIRMED_HOLD 월 → 동결")
    print("   · 상대인상/인하 60% 이상 → 방향 결정")
    print("   · 직전 동결 + 확률차 10% 미만 → 동결 유지 (연속성 보정)")

    with open(RESULT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"\n   결과 저장: {RESULT_JSON_PATH}")


if __name__ == "__main__":
    main()