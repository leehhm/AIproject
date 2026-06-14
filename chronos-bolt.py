import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ══════════════════════════════════════════════════════════════
# 0.  경로 / 하이퍼파라미터 설정
# ══════════════════════════════════════════════════════════════
DATA_DIR       = Path("cleanData")          # ← CSV 파일 폴더
BERT_JSON_PATH = Path("monthly_results.json")
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Chronos-2 모델 ID (HuggingFace)
CHRONOS_MODEL_ID = "amazon/chronos-bolt-base"   # ~200M params

CONTEXT_LENGTH    = 24   # 롤링 예측 시 참조할 과거 개월 수
PREDICTION_LENGTH = 1    # 한 달 앞 예측 (t 실제값과 비교)


# ══════════════════════════════════════════════════════════════
# 1.  헬퍼 함수
# ══════════════════════════════════════════════════════════════
def load_flat_csv(path: Path, freq: str = "MS") -> pd.Series:
    """
    헤더 없이 값만 쉼표로 나열된 단일-행 CSV를 읽어
    DatetimeIndex Series 반환.
    """
    values = pd.read_csv(path, header=None).values.flatten().astype(float)
    if freq == "MS":
        idx = pd.date_range("2016-01-01", periods=len(values), freq="MS")
    else:
        idx = pd.date_range("2016-01-04", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name=path.stem)


def to_monthly_mean(s: pd.Series) -> pd.Series:
    """일별 영업일 Series → 월 평균 (MS freq)"""
    return s.resample("MS").mean()


def to_monthly_last(s: pd.Series) -> pd.Series:
    """일별 영업일 Series → 월말 마지막값 → MS 인덱스"""
    m = s.resample("M").last()
    m.index = m.index.to_period("M").to_timestamp() + pd.offsets.MonthBegin(0)
    return m


# ══════════════════════════════════════════════════════════════
# 2.  데이터 로드 & 통합 DataFrame 구성
# ══════════════════════════════════════════════════════════════
def load_all_data() -> pd.DataFrame:
    print("[1/5] 데이터 로드 & 리샘플링 중...")

    # ── 월별 (120개) ────────────────────────────────────────
    M = "MS"
    kospi_close    = load_flat_csv(DATA_DIR / "코스피_종가(Month).csv",        M)
    kosdaq_close   = load_flat_csv(DATA_DIR / "코스닥_종가(Month).csv",        M)
    us_rate        = load_flat_csv(DATA_DIR / "미국기준금리(Month).csv",        M)
    inflation_exp  = load_flat_csv(DATA_DIR / "기대인플레이션율(Month).csv",    M)
    kospi_per      = load_flat_csv(DATA_DIR / "코스피_PER(Month).csv",          M)
    kospi_mktcap   = load_flat_csv(DATA_DIR / "코스피_시가총액(Month).csv",     M)
    kospi_turnover = load_flat_csv(DATA_DIR / "코스피_회전율(Month).csv",       M)
    kospi_div      = load_flat_csv(DATA_DIR / "코스피_배당수익률(Month).csv",   M)
    kospi_tvol     = load_flat_csv(DATA_DIR / "코스피_거래량(Month).csv",       M)
    kospi_tval     = load_flat_csv(DATA_DIR / "코스피_거래대금(Month).csv",     M)
    kosdaq_mktcap  = load_flat_csv(DATA_DIR / "코스닥_시가총액(Month).csv",   M)
    kosdaq_tvol    = load_flat_csv(DATA_DIR / "코스닥_거래량(Month).csv",       M)
    kosdaq_tval    = load_flat_csv(DATA_DIR / "코스닥_거래대금(Month).csv",     M)

    # ── 일별 → 월 평균 리샘플링 ────────────────────────────
    B = "B"
    call_all_m    = to_monthly_mean(load_flat_csv(DATA_DIR / "콜금리_전체거래(Day).csv",     B))
    call_broker_m = to_monthly_mean(load_flat_csv(DATA_DIR / "콜금리_중개회사거래(Day).csv", B))
    cd91_m        = to_monthly_mean(load_flat_csv(DATA_DIR / "CD91(Month).csv",              B))
    gov3_m        = to_monthly_mean(load_flat_csv(DATA_DIR / "국고채3년(Month).csv",         B))
    koribor_m     = to_monthly_mean(load_flat_csv(DATA_DIR / "코리보3개월(Month).csv",       B))
    usdkrw_m      = to_monthly_mean(load_flat_csv(DATA_DIR / "원달러환율(Day).csv",          B))
    bok_rate_m    = to_monthly_last(load_flat_csv(DATA_DIR / "한국은행기준금리(Month).csv",  B))

    # ── BOK 공식 레이블 ─────────────────────────────────────
    label_df = pd.read_csv(DATA_DIR / "bok_labels.csv")
    label_df["연월"]   = pd.to_datetime(label_df["연월"] + "-01")
    label_df           = label_df.set_index("연월").sort_index()
    label_df["금리변화"] = label_df["기준금리"].diff().fillna(0)

    # ── 통합 ────────────────────────────────────────────────
    df = pd.DataFrame({
        "kospi":          kospi_close,
        "kosdaq":         kosdaq_close,
        "us_rate":        us_rate,
        "inflation_exp":  inflation_exp,
        "kospi_per":      kospi_per,
        "kospi_mktcap":   kospi_mktcap,
        "kospi_turnover": kospi_turnover,
        "kospi_div":      kospi_div,
        "kospi_tvol":     kospi_tvol,
        "kospi_tval":     kospi_tval,
        "kosdaq_mktcap":  kosdaq_mktcap,
        "kosdaq_tvol":    kosdaq_tvol,
        "kosdaq_tval":    kosdaq_tval,
        "call_all":       call_all_m,
        "call_broker":    call_broker_m,
        "cd91":           cd91_m,
        "gov3":           gov3_m,
        "koribor":        koribor_m,
        "usdkrw":         usdkrw_m,
        "bok_rate":       bok_rate_m,
    })
    df = df.join(label_df[["기준금리", "금리변화", "결정"]], how="left")
    df["결정"]   = df["결정"].fillna("동결")
    df["금리변화"] = df["금리변화"].fillna(0)
    df = df.loc["2016-01-01":"2025-12-01"]

    print(f"   → {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}, "
          f"{len(df)}개월, 변수 {len(df.columns)}개")
    return df


# ══════════════════════════════════════════════════════════════
# 3.  Policy Surprise 계산
# ══════════════════════════════════════════════════════════════
def compute_policy_surprise(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Policy Surprise 계산 중...")

    df["call_change"]     = df["call_all"].diff().fillna(0)
    df["actual_change"]   = df["금리변화"]
    df["policy_surprise"] = (df["actual_change"] - df["call_change"]).abs()

    ps_max = df["policy_surprise"].max()
    df["policy_surprise_norm"] = df["policy_surprise"] / ps_max if ps_max > 0 else 0.0

    top = df["policy_surprise"].nlargest(3)
    print(f"   → 평균 {df['policy_surprise'].mean():.4f}%p | "
          f"상위 3개: " + ", ".join(f"{d.strftime('%Y-%m')}({v:.3f})" for d, v in top.items()))
    return df


# ══════════════════════════════════════════════════════════════
# 4.  BERT 보조 피처 결합
# ══════════════════════════════════════════════════════════════
def merge_bert_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] BERT 피처 결합 중...")

    if not BERT_JSON_PATH.exists():
        print(f"   ⚠  {BERT_JSON_PATH} 없음 → BERT 피처 중립값으로 대체")
        df["bert_raise"]  = 0.33
        df["bert_hold"]   = 0.34
        df["bert_cut"]    = 0.33
        df["bert_signal"] = 0.0
        return df

    with open(BERT_JSON_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = [{
        "date":        pd.to_datetime(ym + "-01"),
        "bert_raise":  v.get("인상확률", 33) / 100,
        "bert_hold":   v.get("동결확률", 34) / 100,
        "bert_cut":    v.get("인하확률", 33) / 100,
        "bert_signal": (v.get("상대인상확률", 50) - v.get("상대인하확률", 50)) / 100,
    } for ym, v in raw.items()]

    bert_df = pd.DataFrame(rows).set_index("date")
    df = df.join(bert_df, how="left")
    for col in ["bert_raise", "bert_hold", "bert_cut", "bert_signal"]:
        df[col] = df[col].fillna(df[col].mean())

    print(f"   → {bert_df.index.isin(df.index).sum()}개월 BERT 피처 매칭 완료")
    return df


# ══════════════════════════════════════════════════════════════
# 5.  Chronos-2 롤링 예측 + 신뢰도 지수 산출 (시점 한 달 당김 적용)
# ══════════════════════════════════════════════════════════════
def compute_credibility_index(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Chronos-2 롤링 예측 + 신뢰도 지수 산출 중...")

    use_chronos = False
    try:
        import torch
        from chronos import BaseChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   디바이스: {device} | 모델: {CHRONOS_MODEL_ID}")
        
        pipeline = BaseChronosPipeline.from_pretrained(
            CHRONOS_MODEL_ID,
            device_map=device,
            dtype=torch.bfloat16,
        )
        use_chronos = True
        print("   Chronos-2 로드 완료 ✓")
    except Exception as e:
        print(f"   ⚠  Chronos 로드 실패({e}) → 통계 Fallback 사용")

    kospi = df["kospi"].values
    n     = len(kospi)
    rows  = []

    # [수정포인트]: 범위를 n까지 확장하고, ctx는 t 직전까지, actual은 t 시점으로 정렬합니다.
    for t in range(CONTEXT_LENGTH, n):
        ctx    = kospi[t - CONTEXT_LENGTH: t]  # t 시점 직전까지의 과거 24개월 데이터
        actual = kospi[t]                      # 예측 목표인 t 시점의 실제 KOSPI 값

        if use_chronos:
            import torch
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            
            # ChronosBoltPipeline은 num_samples 인자를 받지 않습니다.
            fc = pipeline.predict(
                inputs=ctx_tensor,
                prediction_length=PREDICTION_LENGTH,
            )
            
            if hasattr(fc, "numpy"):
                samples = fc.numpy().flatten()
            else:
                samples = fc[0].cpu().numpy().flatten()
                
            q10 = float(np.quantile(samples, 0.10))
            q50 = float(np.quantile(samples, 0.50))
            q90 = float(np.quantile(samples, 0.90))
        else:
            mu, sigma = float(np.mean(ctx)), float(np.std(ctx)) + 1e-6
            q10 = mu - 1.28 * sigma
            q50 = mu
            q90 = mu + 1.28 * sigma

        iqr = (q90 - q10) + 1e-6

        if q10 <= actual <= q90:
            deviation = 0.0
        elif actual < q10:
            deviation = (q10 - actual) / iqr
        else:
            deviation = (actual - q90) / iqr

        credibility = float(max(0.0, 100.0 * np.exp(-deviation)))

        rows.append({
            "date":         df.index[t], # [수정포인트]: t 시점의 인덱스 날짜와 직접 매핑 (한 달 당겨짐)
            "kospi_actual": actual,
            "q10":          q10,
            "q50":          q50,
            "q90":          q90,
            "deviation":    deviation,
            "credibility":  credibility,
        })

        if (t - CONTEXT_LENGTH) % 12 == 0:
            ym = df.index[t].strftime("%Y-%m")
            print(f"   {ym}  실제:{actual:>7,.0f}  "
                  f"Q10:{q10:>7,.0f}  Q90:{q90:>7,.0f}  "
                  f"이탈:{deviation:.3f}  신뢰도:{credibility:.1f}")

    result_df = pd.DataFrame(rows).set_index("date")
    return result_df


# ══════════════════════════════════════════════════════════════
# 6.  최종 통합 & 저장
# ══════════════════════════════════════════════════════════════
def integrate_and_save(df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.join(result_df[["q10", "q50", "q90", "deviation", "credibility"]], how="left")

    merged["credibility_adj"] = (
        merged["credibility"].fillna(50)
        + merged["bert_signal"].fillna(0) * 5
    ).clip(0, 100)

    merged["composite_index"] = (
        merged["credibility_adj"]
        * (1 - merged["policy_surprise_norm"].fillna(0))
    ).clip(0, 100)

    def grade(v):
        if pd.isna(v): return "-"
        if v >= 70: return "HIGH"
        if v >= 40: return "MID"
        return "LOW"
    merged["credibility_grade"] = merged["credibility_adj"].apply(grade)

    export_cols = [
        "결정", "기준금리", "금리변화",
        "kospi", "kosdaq", "usdkrw",
        "call_all", "call_change", "actual_change", "policy_surprise",
        "bert_raise", "bert_hold", "bert_cut", "bert_signal",
        "q10", "q50", "q90", "deviation",
        "credibility", "credibility_adj", "credibility_grade", "composite_index",
    ]
    export_cols = [c for c in export_cols if c in merged.columns]
    out_csv = OUTPUT_DIR / "credibility_index.csv"
    merged[export_cols].to_csv(out_csv, encoding="utf-8-sig", float_format="%.4f")
    print(f"\n   결과 CSV: {out_csv}")
    return merged


# ══════════════════════════════════════════════════════════════
# 7.  성능 지표
# ══════════════════════════════════════════════════════════════
def evaluate_metrics(merged: pd.DataFrame):
    valid = merged[["kospi", "q50"]].dropna()
    if len(valid) < 5:
        print("\n⚠  예측값 부족으로 성능 지표 생략")
        return

    actual, pred = valid["kospi"].values, valid["q50"].values
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae  = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-6))) * 100
    cov  = ((merged["kospi"] >= merged["q10"]) &
            (merged["kospi"] <= merged["q90"])).mean() * 100

    print(f"\n{'═'*42}")
    print("   📊 Chronos-2 예측 성능 (KOSPI, Q50 기준)")
    print(f"{'═'*42}")
    print(f"  RMSE   : {rmse:>10.2f}  pt")
    print(f"  MAE    : {mae:>10.2f}  pt")
    print(f"  MAPE   : {mape:>10.2f}  %")
    print(f"  Q10-Q90 커버리지: {cov:.1f}%  (이론 80%)")
    print(f"{'═'*42}")


# ══════════════════════════════════════════════════════════════
# 8.  4-패널 대시보드 시각화
# ══════════════════════════════════════════════════════════════
def visualize(merged: pd.DataFrame):
    print("\n시각화 생성 중...")

    try:
        plt.rcParams["font.family"]        = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    RAISE_COLOR = "#e05c3a"
    CUT_COLOR   = "#3a9de0"
    BAND_COLOR  = "#e05c3a"
    CRED_HIGH   = "#27ae60"
    CRED_MID    = "#f39c12"
    CRED_LOW    = "#e74c3c"

    dates  = merged.index
    fig, axes = plt.subplots(4, 1, figsize=(17, 22), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.suptitle(
        "한국은행 통화정책 신뢰도 분석 대시보드\n"
        "(Chronos-2 롤링 예측 기반 신뢰구간 이탈률 + Policy Surprise)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    def mark_decisions(ax, df):
        for d, row in df.iterrows():
            dec = row.get("결정", "동결")
            if dec == "인상":
                ax.axvline(d, color=RAISE_COLOR, alpha=0.35, lw=1.0, ls="--")
            elif dec == "인하":
                ax.axvline(d, color=CUT_COLOR, alpha=0.35, lw=1.0, ls="--")

    # ── 패널 ① : KOSPI + Chronos 신뢰구간 ────────────────────
    ax = axes[0]
    ax.plot(dates, merged["kospi"], color="#1a1a2e", lw=2.0, label="KOSPI 실제", zorder=3)
    if merged["q50"].notna().any():
        # 데이터 정렬 구조 자체를 교정했으므로 별도의 조작 없이 올바른 자리에 매칭됩니다.
        ax.plot(dates, merged["q50"], color=BAND_COLOR, lw=1.3,
                ls="--", alpha=0.85, label="Chronos-2 Q50", zorder=2)
        ax.fill_between(dates, merged["q10"], merged["q90"],
                        color=BAND_COLOR, alpha=0.13,
                        label="예측 신뢰구간 (Q10–Q90)", zorder=1)
    mark_decisions(ax, merged)
    
    from matplotlib.lines import Line2D
    legend_extra = [
        Line2D([0], [0], color=RAISE_COLOR, ls="--", lw=1, alpha=0.6, label="인상 결정월"),
        Line2D([0], [0], color=CUT_COLOR,   ls="--", lw=1, alpha=0.6, label="인하 결정월"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labels + [l.get_label() for l in legend_extra],
              fontsize=8.5, loc="upper left", ncol=2)
    ax.set_ylabel("KOSPI 지수", fontsize=10)
    ax.set_title("① KOSPI 실제값 vs Chronos-2 예측 신뢰구간", fontsize=11, pad=4)
    ax.grid(True, alpha=0.25)

    # ── 패널 ② : 통화정책 신뢰도 지수 ──────────────────────
    ax = axes[1]
    cred = merged.get("credibility_adj", merged.get("credibility", pd.Series(dtype=float)))
    bar_colors = [CRED_HIGH if v >= 70 else (CRED_MID if v >= 40 else CRED_LOW)
                  for v in cred.fillna(50)]
    ax.bar(dates, cred.fillna(0), width=25, color=bar_colors, alpha=0.85, zorder=2)
    ax.axhline(70, color=CRED_HIGH, ls="--", lw=1.1, alpha=0.7, label="HIGH 기준 (70)")
    ax.axhline(40, color=CRED_LOW,  ls="--", lw=1.1, alpha=0.7, label="LOW 기준 (40)")
    mark_decisions(ax, merged)
    ax.set_ylim(0, 108)
    ax.set_ylabel("신뢰도 지수 (0–100)", fontsize=10)
    ax.set_title("② 통화정책 신뢰도 지수 (BERT 보정 포함)", fontsize=11, pad=4)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")

    # ── 패널 ③ : Policy Surprise ────────────────────────────
    ax = axes[2]
    ps = merged["policy_surprise"].fillna(0)
    ax.bar(dates, ps, width=25, color="#8e44ad", alpha=0.78, zorder=2)
    ax.set_ylabel("Policy Surprise (|Δ실제 − Δ기대| %p)", fontsize=10)
    ax.set_title("③ Policy Surprise  (실제 기준금리 변화 − 콜금리 기대 변화)", fontsize=11, pad=4)
    for d, v in ps.nlargest(5).items():
        ax.annotate(d.strftime("%Y-%m"), xy=(d, v), xytext=(0, 5),
                    textcoords="offset points", fontsize=7.5,
                    ha="center", color="#4a235a", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    # ── 패널 ④ : BERT 신호 + 기준금리 추이 ─────────────────
    ax   = axes[3]
    ax2  = ax.twinx()
    if "bert_signal" in merged.columns:
        sig = merged["bert_signal"].fillna(0)
        ax.bar(dates, sig * 100, width=25,
               color=[RAISE_COLOR if v > 0 else CUT_COLOR for v in sig],
               alpha=0.65, label="BERT 신호 (인상 ▲ / 인하 ▼)", zorder=2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("BERT 방향 신호 (×100)", fontsize=10)
    if "기준금리" in merged.columns:
        ax2.plot(dates, merged["기준금리"], color="#1a1a2e",
                 lw=2.2, label="한국은행 기준금리 (%)", zorder=3)
        ax2.set_ylabel("기준금리 (%)", fontsize=10)
    ax.set_title("④ BERT 방향성 신호 + 한국은행 기준금리 추이", fontsize=11, pad=4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    out_path = OUTPUT_DIR / "credibility_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.993])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   대시보드: {out_path}")


# ══════════════════════════════════════════════════════════════
# 9.  요약 리포트 콘솔 출력
# ══════════════════════════════════════════════════════════════
def print_summary(merged: pd.DataFrame):
    print("\n" + "═" * 72)
    print("   📋 월별 통화정책 신뢰도 지수 요약")
    print("═" * 72)
    print(f"   {'연월':<10} {'결정':^6} {'기준금리':>7} {'PS':>7} "
          f"{'신뢰도':>7} {'복합지수':>9} {'등급':>6}")
    print("─" * 72)
    for d, row in merged.iterrows():
        dec   = row.get("결정", "-")
        rate  = row.get("기준금리", float("nan"))
        ps    = row.get("policy_surprise", 0.0)
        cred  = row.get("credibility_adj", row.get("credibility", float("nan")))
        comp  = row.get("composite_index", float("nan"))
        grade = row.get("credibility_grade", "-")
        flag  = " ◀ 주목" if (not pd.isna(cred) and cred < 40 and ps > 0.05) else ""
        print(f"   {d.strftime('%Y-%m'):<10} {dec:^6} {rate:>7.2f}%  "
              f"{ps:>6.3f}  {cred:>7.1f}  {comp:>9.1f}  {grade:>6}{flag}")
    print("═" * 72)

    vc = merged["credibility"].dropna()
    if len(vc):
        print(f"\n   신뢰도 평균: {vc.mean():.1f} | "
              f"최저: {vc.min():.1f} ({vc.idxmin().strftime('%Y-%m')}) | "
              f"최고: {vc.max():.1f} ({vc.idxmax().strftime('%Y-%m')})\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("   한국은행 통화정책 신뢰도 지수 파이프라인  (Chronos-2 / chronos-bolt-base)")
    print("=" * 72 + "\n")

    df         = load_all_data()
    df         = compute_policy_surprise(df)
    df         = merge_bert_features(df)
    result_df  = compute_credibility_index(df)
    merged     = integrate_and_save(df, result_df)
    evaluate_metrics(merged)
    print_summary(merged)
    visualize(merged)

    print("\n✅ 파이프라인 완료")
    print(f"   출력 폴더: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()