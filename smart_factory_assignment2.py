
import math
from typing import Dict, Tuple, Optional

import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.stats import shapiro, boxcox
import plotly.io as pio
pio.renderers.default = 'svg'

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False



# 0. 기본 설정
st.set_page_config(
    page_title="Smart Manufacturing SPC App",
    page_icon="🏭",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.2rem;}
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #38bdf8 100%);
        padding: 26px 30px;
        border-radius: 22px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.25);
    }
    .hero h1 {margin: 0; font-size: 34px;}
    .hero p {margin-top: 8px; font-size: 16px; opacity: 0.92;}
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        min-height: 120px;
    }
    .metric-title {color: #64748b; font-size: 13px; font-weight: 800;}
    .metric-value {color: #0f172a; font-size: 30px; font-weight: 900; margin-top: 8px;}
    .metric-desc {font-size: 13px; margin-top: 6px;}
    .good {color: #047857; font-weight: 800;}
    .warn {color: #b45309; font-weight: 800;}
    .bad {color: #b91c1c; font-weight: 800;}
    .box {
        background: white;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# 1. 불편화 상수 CSV 불러오기


from pathlib import Path

APP_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def load_constant_file(candidate_names):
    """
    불편화 상수 CSV 파일을 현재 app.py와 같은 폴더에서 찾는다.
    여러 파일명을 후보로 두어 다운로드/업로드 과정에서 이름이 조금 달라도 읽을 수 있게 한다.
    """
    for name in candidate_names:
        path = APP_DIR / name
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        "불편화 상수 CSV 파일을 찾을 수 없습니다. "
        f"다음 파일 중 하나가 app1.py와 같은 폴더에 있어야 합니다: {candidate_names}"
    )


# 관리도용 불편화 상수: m, A2, A3, d2, D3, D4, B3, B4
control_const_df = load_constant_file([
    "unbiased_control_chart.csv"
])

control_const_df.columns = [
    str(c).strip()
    for c in control_const_df.columns
]

control_const_df = control_const_df.rename(
    columns={
        "m": "n",
        "M": "n"
    }
)

control_const_df["n"] = control_const_df["n"].astype(int)
control_const_df = control_const_df.set_index("n")


# 공정능력분석용 불편화 상수: N, d2, d3, d4
capability_const_df = load_constant_file([
    "unbiased_capability_analysis.csv",
    "unbiased_capability_analysis(1).csv",
    "unbiased_capability_analysis(2).csv"
])

capability_const_df.columns = [
    str(c).strip()
    for c in capability_const_df.columns
]

capability_const_df = capability_const_df.rename(
    columns={
        "N": "n",
        "n": "n"
    }
)

capability_const_df["n"] = capability_const_df["n"].astype(int)
capability_const_df = capability_const_df.set_index("n")


def series_to_dict(df, col):
    """상수 테이블의 특정 컬럼을 {n: 값} 딕셔너리로 변환"""
    return {
        int(idx): float(value)
        for idx, value in df[col].dropna().items()
    }



A2 = series_to_dict(control_const_df, "A2")
A3 = series_to_dict(control_const_df, "A3")
B3 = series_to_dict(control_const_df, "B3")
B4 = series_to_dict(control_const_df, "B4")
D3 = series_to_dict(control_const_df, "D3")
D4 = series_to_dict(control_const_df, "D4")


D2 = series_to_dict(capability_const_df, "d2")


def c4_value(n: int) -> float:
    """
    c4 불편화 상수 계산.
    제공된 CSV에는 c4가 없으므로 표준식으로 계산하여 사용한다.
    c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)
    """
    n = int(round(n))

    if n <= 1:
        return np.nan

    return (
        np.sqrt(2 / (n - 1))
        * math.gamma(n / 2)
        / math.gamma((n - 1) / 2)
    )



C4 = {
    n: c4_value(n)
    for n in range(2, 51)
}


def get_coef(table: Dict[int, float], n: int) -> float:
    """불편화 계수 계산: n이 표 범위를 벗어나면 가장 가까운 값을 사용"""
    n = int(round(n))

    if n in table:
        return table[n]

    if n < min(table.keys()):
        return table[min(table.keys())]

    return table[max(table.keys())]



def make_sample_data(seed: int = 42, n_lot: int = 30, subgroup_size: int = 5) -> pd.DataFrame:
    """1) 데이터 준비: 임의 제조공정 데이터 생성"""
    np.random.seed(seed)
    rows = []

    for lot in range(1, n_lot + 1):
        # 일부 lot 이후 평균이 살짝 이동하도록 만들어 이상 탐지 가능하게 함
        shift = 0.00 if lot <= int(n_lot * 0.65) else 0.08

        for i in range(subgroup_size):
            thickness = np.random.normal(loc=10.00 + shift, scale=0.045)
            sample_size = 200
            defects = np.random.binomial(n=sample_size, p=0.015 + (0.006 if lot > int(n_lot * 0.75) else 0))
            rows.append({
                "Lot": lot,
                "Sample_No": i + 1,
                "Thickness": thickness,
                "Sample_Size": sample_size,
                "Defects": defects
            })

    return pd.DataFrame(rows)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """2) 데이터 전처리: 중복 컬럼 제거, 숫자형 변환, 결측치 제거 준비"""
    df = df.copy()

    # 같은 이름의 컬럼이 있으면 groupby 오류가 나므로 제거
    df = df.loc[:, ~df.columns.duplicated()]

    # 컬럼명 앞뒤 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    return df


# 분석 대상 컬럼 자동 판별


def normalize_col(col):
    """
    컬럼명을 비교하기 쉽도록 정규화
    Sample_Size -> samplesize
    Sample Size -> samplesize
    sampleSize -> samplesize
    """
    return (
        str(col)
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


def is_meta_column(col):
    """
    분석 대상이 아닌 메타데이터 컬럼 판별
    """

    name = normalize_col(col)

    keywords = [
        "id",
        "index",
        "lot",
        "batch",
        "wafer",
        "sampleno",
        "samplesize",
        "serial",
        "number"
    ]

    return any(keyword in name for keyword in keywords)

def is_subgroup_column(col):

    name = normalize_col(col)

    keywords = [
        "lot",
        "batch",
        "group",
        "shift",
        "line",
        "wafer"
    ]

    return any(
        keyword in name
        for keyword in keywords
    )

def normality_test(values: pd.Series) -> Dict[str, object]:
    """3) 정규성 검정: Shapiro-Wilk 또는 D'Agostino 검정"""
    x = pd.to_numeric(values, errors="coerce").dropna()

    if len(x) < 3:
        return {"method": "검정 불가", "p_value": np.nan, "is_normal": False}

    if not SCIPY_AVAILABLE:
        return {"method": "scipy 없음", "p_value": np.nan, "is_normal": True}

    # 표본이 5000개 이하면 Shapiro, 크면 normaltest 사용
    if len(x) <= 5000:
        stat, p = stats.shapiro(x)
        method = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(x)
        method = "D'Agostino K²"

    return {
        "method": method,
        "p_value": float(p),
        "is_normal": bool(p >= 0.05)
    }


def calculate_unbiased_constants(n: int) -> Dict[str, float]:
    """4) 불편화 상수 계산: CSV에서 불러온 상수를 n에 맞게 반환"""
    return {
        "A2": get_coef(A2, n),
        "A3": get_coef(A3, n),
        "B3": get_coef(B3, n),
        "B4": get_coef(B4, n),
        "d2": get_coef(D2, n),
        "D3": get_coef(D3, n),
        "D4": get_coef(D4, n),
        "C4": c4_value(n)
    }


def sigma_within(df: pd.DataFrame, value_col: str, subgroup_col: Optional[str]) -> float:
    """군내변동 sigma_within 계산"""
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()

    if subgroup_col is None or subgroup_col == "없음":
        mr = values.diff().abs().dropna()
        mr_bar = mr.mean()
        return mr_bar / get_coef(D2, 2)

    # 안전한 groupby: 중복 컬럼 문제 방지
    temp = pd.DataFrame({
        "subgroup": df[subgroup_col],
        "value": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna()

    grouped = temp.groupby("subgroup")["value"]
    sizes = grouped.size()

    if sizes.empty:
        return np.nan

    # 부분군 크기가 1이면 I-MR 방식
    if sizes.max() <= 1:
        v = temp["value"].reset_index(drop=True)
        mr = v.diff().abs().dropna()
        return mr.mean() / get_coef(D2, 2)

    # 부분군 크기가 같으면 Rbar / d2 방식
    if sizes.nunique() == 1:
        n = int(sizes.iloc[0])
        ranges = grouped.max() - grouped.min()
        return ranges.mean() / get_coef(D2, n)

    # 부분군 크기가 다르면 pooled standard deviation / c4 근사
    ss = 0
    denom = 0

    for _, g in grouped:
        arr = g.dropna().values
        if len(arr) >= 2:
            ss += np.sum((arr - arr.mean()) ** 2)
            denom += len(arr) - 1

    if denom <= 0:
        return np.nan

    sp = math.sqrt(ss / denom)
    n_avg = int(round(sizes.mean()))
    return sp / c4_value(n_avg)


def sigma_overall(values: pd.Series) -> float:
    """전체변동 sigma_overall 계산"""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    return x.std(ddof=1) / c4_value(len(x))


def calculate_capability_indices(
    df: pd.DataFrame,
    value_col: str,
    subgroup_col: Optional[str],
    lsl: float,
    usl: float
) -> Dict[str, float]:
    """5) 공정능력지수 계산: Cp, Cpk, Pp, Ppk"""
    x = pd.to_numeric(df[value_col], errors="coerce").dropna()
    mu = x.mean()

    sig_w = sigma_within(df, value_col, subgroup_col)
    sig_o = sigma_overall(x)

    cp = (usl - lsl) / (6 * sig_w) if sig_w > 0 else np.nan
    cpk = min((usl - mu) / (3 * sig_w), (mu - lsl) / (3 * sig_w)) if sig_w > 0 else np.nan

    pp = (usl - lsl) / (6 * sig_o) if sig_o > 0 else np.nan
    ppk = min((usl - mu) / (3 * sig_o), (mu - lsl) / (3 * sig_o)) if sig_o > 0 else np.nan

    ppm_lsl = (x < lsl).mean() * 1_000_000
    ppm_usl = (x > usl).mean() * 1_000_000

    return {
        "Mean": mu,
        "Sigma_within": sig_w,
        "Sigma_overall": sig_o,
        "Cp": cp,
        "Cpk": cpk,
        "Pp": pp,
        "Ppk": ppk,
        "PPM_LSL": ppm_lsl,
        "PPM_USL": ppm_usl,
        "PPM_Total": ppm_lsl + ppm_usl
    }


def handle_non_normal_data(values: pd.Series) -> Tuple[pd.Series, str]:
    """6) 정규성 불만족시 처리 방법: 양수 데이터면 log 변환 예시 제공"""
    x = pd.to_numeric(values, errors="coerce").dropna()

    if (x <= 0).any():
        return x, "0 이하 값이 포함되어 log 변환은 적용하지 않았습니다."

    x_log = np.log(x)
    return pd.Series(x_log, index=x.index), "정규성 불만족 가능성이 있어 log 변환 데이터를 추가 검토할 수 있습니다."



def make_variable_control_chart(
    df: pd.DataFrame,
    value_col: str,
    subgroup_col: Optional[str],
    chart_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """2) 계량형 관리도 생성: Xbar-R, Xbar-S, I-MR"""
    if chart_type == "I-MR" or subgroup_col is None or subgroup_col == "없음":
        x = pd.to_numeric(df[value_col], errors="coerce").dropna().reset_index(drop=True)
        center = x.mean()
        mr = x.diff().abs()
        mr_bar = mr.dropna().mean()
        sigma = mr_bar / get_coef(D2, 2)

        chart_df = pd.DataFrame({
            "Point": np.arange(1, len(x) + 1),
            "Statistic": x,
            "CL": center,
            "UCL": center + 3 * sigma,
            "LCL": center - 3 * sigma
        })

        mr_df = pd.DataFrame({
            "Point": np.arange(1, len(mr) + 1),
            "Statistic": mr,
            "CL": mr_bar,
            "UCL": get_coef(D4, 2) * mr_bar,
            "LCL": 0
        }).dropna()

        chart_df["Chart"] = "I"
        mr_df["Chart"] = "MR"
        return chart_df, mr_df

    temp = pd.DataFrame({
        "subgroup": df[subgroup_col],
        "value": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna()

    grouped = temp.groupby("subgroup")["value"]
    summary = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary["range"] = summary["max"] - summary["min"]

    n = int(summary["count"].mode().iloc[0])
    xbarbar = summary["mean"].mean()

    if chart_type == "Xbar-S":
        sbar = summary["std"].mean()
        chart_df = pd.DataFrame({
            "Point": summary["subgroup"],
            "Statistic": summary["mean"],
            "CL": xbarbar,
            "UCL": xbarbar + get_coef(A3, n) * sbar,
            "LCL": xbarbar - get_coef(A3, n) * sbar
        })

        sub_df = pd.DataFrame({
            "Point": summary["subgroup"],
            "Statistic": summary["std"],
            "CL": sbar,
            "UCL": get_coef(B4, n) * sbar,
            "LCL": get_coef(B3, n) * sbar
        })
        chart_df["Chart"] = "Xbar"
        sub_df["Chart"] = "S"
        return chart_df, sub_df

    rbar = summary["range"].mean()
    chart_df = pd.DataFrame({
        "Point": summary["subgroup"],
        "Statistic": summary["mean"],
        "CL": xbarbar,
        "UCL": xbarbar + get_coef(A2, n) * rbar,
        "LCL": xbarbar - get_coef(A2, n) * rbar
    })

    sub_df = pd.DataFrame({
        "Point": summary["subgroup"],
        "Statistic": summary["range"],
        "CL": rbar,
        "UCL": get_coef(D4, n) * rbar,
        "LCL": get_coef(D3, n) * rbar
    })
    chart_df["Chart"] = "Xbar"
    sub_df["Chart"] = "R"
    return chart_df, sub_df


def make_attribute_control_chart(
    df: pd.DataFrame,
    count_col: str,
    sample_col: Optional[str],
    subgroup_col: Optional[str],
    chart_type: str
) -> pd.DataFrame:
    """3) 계수형 관리도 생성: NP, P, C, U"""
    if subgroup_col and subgroup_col != "없음":
        if sample_col and sample_col != "없음":
            temp = df.groupby(subgroup_col).agg(
                count=(count_col, "sum"),
                n=(sample_col, "sum")
            ).reset_index()
        else:
            temp = df.groupby(subgroup_col).agg(
                count=(count_col, "sum")
            ).reset_index()
            temp["n"] = 1

        point = temp[subgroup_col]
    else:
        temp = pd.DataFrame()
        temp["count"] = pd.to_numeric(df[count_col], errors="coerce")
        temp["n"] = pd.to_numeric(df[sample_col], errors="coerce") if sample_col and sample_col != "없음" else 1
        point = np.arange(1, len(temp) + 1)

    count = pd.to_numeric(temp["count"], errors="coerce").fillna(0)
    n = pd.to_numeric(temp["n"], errors="coerce").fillna(1).replace(0, 1)

    if chart_type == "P":
        pbar = count.sum() / n.sum()
        stat = count / n
        cl = pbar
        ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / n)
        lcl = np.maximum(0, pbar - 3 * np.sqrt(pbar * (1 - pbar) / n))

    elif chart_type == "NP":
        nbar = n.mean()
        pbar = count.sum() / n.sum()
        stat = count
        cl = nbar * pbar
        ucl = cl + 3 * np.sqrt(nbar * pbar * (1 - pbar))
        lcl = max(0, cl - 3 * np.sqrt(nbar * pbar * (1 - pbar)))

    elif chart_type == "C":
        cbar = count.mean()
        stat = count
        cl = cbar
        ucl = cbar + 3 * np.sqrt(cbar)
        lcl = max(0, cbar - 3 * np.sqrt(cbar))

    else:  # U
        ubar = count.sum() / n.sum()
        stat = count / n
        cl = ubar
        ucl = ubar + 3 * np.sqrt(ubar / n)
        lcl = np.maximum(0, ubar - 3 * np.sqrt(ubar / n))

    return pd.DataFrame({
        "Point": point,
        "Statistic": stat,
        "CL": cl,
        "UCL": ucl,
        "LCL": lcl,
        "Chart": chart_type
    })


def detect_outliers_basic(chart_df: pd.DataFrame) -> pd.DataFrame:
    """관리한계 이탈점 탐지"""
    result = chart_df.copy()
    result["Outlier"] = (result["Statistic"] > result["UCL"]) | (result["Statistic"] < result["LCL"])
    return result


def nelson_rule_1(chart_df: pd.DataFrame) -> pd.DataFrame:
    """Nelson Rule 1: ±3σ 관리한계 밖 점"""
    result = detect_outliers_basic(chart_df)
    return result[result["Outlier"]][["Point", "Statistic", "CL", "UCL", "LCL"]]


def remove_outliers_and_recalculate(
    df: pd.DataFrame,
    chart_df: pd.DataFrame,
    subgroup_col: Optional[str],
    value_col: str,
    chart_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """5) 이상치 제거 후 관리도 재작성하기"""
    outlier_points = chart_df.loc[
        (chart_df["Statistic"] > chart_df["UCL"]) | (chart_df["Statistic"] < chart_df["LCL"]),
        "Point"
    ].tolist()

    if len(outlier_points) == 0:
        new_chart, new_sub = make_variable_control_chart(df, value_col, subgroup_col, chart_type)
        return df, new_chart, new_sub

    if subgroup_col and subgroup_col != "없음" and chart_type != "I-MR":
        filtered = df[~df[subgroup_col].isin(outlier_points)].copy()
    else:
        # I-MR인 경우 point는 행 순서이므로 index 기준 제거
        remove_idx = [int(p) - 1 for p in outlier_points]
        filtered = df.drop(df.index[remove_idx]).copy()

    new_chart, new_sub = make_variable_control_chart(filtered, value_col, subgroup_col, chart_type)
    return filtered, new_chart, new_sub


def plot_control_chart(chart_df: pd.DataFrame, title: str) -> go.Figure:
    """4) 관리도 시각화"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df["Point"],
        y=chart_df["Statistic"],
        mode="lines+markers",
        name="Statistic"
    ))

    fig.add_trace(go.Scatter(
        x=chart_df["Point"],
        y=chart_df["CL"],
        mode="lines",
        name="CL"
    ))

    fig.add_trace(go.Scatter(
        x=chart_df["Point"],
        y=chart_df["UCL"],
        mode="lines",
        name="UCL",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=chart_df["Point"],
        y=chart_df["LCL"],
        mode="lines",
        name="LCL",
        line=dict(dash="dash")
    ))

    outliers = chart_df[
        (chart_df["Statistic"] > chart_df["UCL"]) |
        (chart_df["Statistic"] < chart_df["LCL"])
    ]

    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            x=outliers["Point"],
            y=outliers["Statistic"],
            mode="markers",
            name="Outlier",
            marker=dict(size=14, symbol="x")
        ))

    fig.update_layout(
        title=title,
        height=430,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=30, t=60, b=30)
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

    return fig


def plot_capability_histogram(values: pd.Series, lsl: float, usl: float, target: float) -> go.Figure:
    """공정능력분석 시각화: 히스토그램 + 규격선"""
    x = pd.to_numeric(values, errors="coerce").dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x,
        name="Data",
        opacity=0.75
    ))

    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
    fig.add_vline(x=target, line_dash="dot", annotation_text="Target")

    fig.update_layout(
        title="Process Capability Histogram",
        height=430,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=60, b=30)
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

    return fig


def metric_card(title: str, value: str, desc: str, level: str = "good") -> None:
    """UI 카드"""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-desc {level}">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def capability_level(cpk: float) -> Tuple[str, str]:
    """공정능력 판정"""
    if pd.isna(cpk):
        return "계산 불가", "bad"
    if cpk >= 1.33:
        return "양호", "good"
    if cpk >= 1.00:
        return "주의", "warn"
    return "부족", "bad"

def recommend_control_chart(
    df,
    analysis_col,
    subgroup_col=None,
    sample_size_col=None
):
    """
    강의록 기준 관리도 자동 추천
    """

    series = pd.to_numeric(
        df[analysis_col],
        errors="coerce"
    ).dropna()

    
    # 계량형 판정
    

    decimal_exist = (
        series % 1 != 0
    ).any()

    if decimal_exist:

        if subgroup_col is None or subgroup_col == "없음":

            return {
                "data_type": "계량형",
                "chart": "I-MR",
                "reason":
                "부분군이 없으므로 I-MR 관리도 사용"
            }

        subgroup_size = int(
            df.groupby(subgroup_col)
              .size()
              .mode()
              .iloc[0]
        )

        if subgroup_size < 10:

            return {
                "data_type": "계량형",
                "chart": "Xbar-R",
                "reason":
                f"부분군 크기={subgroup_size}, \n"
                '\n'
                "강의록 기준 Xbar-R 사용"
            }

        else:

            return {
                "data_type": "계량형",
                "chart": "Xbar-S",
                "reason":
                f"부분군 크기={subgroup_size},\n"
                '\n'
                "강의록 기준 Xbar-S 사용"
            }

    
    # 계수형 판정
    

    else:

        if sample_size_col is not None:

            sample_values = df[
                sample_size_col
            ]

            if sample_values.nunique() == 1:

                return {
                    "data_type": "계수형",
                    "chart": "P",
                    "reason":
                    "불량률 데이터"
                }

            else:

                return {
                    "data_type": "계수형",
                    "chart": "U",
                    "reason":
                    "단위당 결점수 데이터"
                }

        return {
            "data_type": "계수형",
            "chart": "C",
            "reason":
            "결점 개수 데이터"
        }



# 공정능력 등급


def capability_grade(cpk):

    if pd.isna(cpk):
        return "N/A"

    elif cpk >= 1.67:
        return "S"

    elif cpk >= 1.33:
        return "A"

    elif cpk >= 1.00:
        return "B"

    else:
        return "C"



# 공정 상태 진단


def process_status(cpk):

    if pd.isna(cpk):
        return (
            "⚪ 계산 불가",
            "공정능력을 계산할 수 없습니다."
        )

    elif cpk >= 1.33:

        return (
            "🟢 양호",
            "현재 공정은 규격을 안정적으로 만족하고 있습니다."
        )

    elif cpk >= 1.00:

        return (
            "🟡 주의",
            "공정능력은 확보되었으나 지속적인 모니터링이 필요합니다."
        )

    else:

        return (
            "🔴 개선 필요",
            "규격 이탈 가능성이 높으므로 개선이 필요합니다."
        )


# 자동 분석 리포트


def generate_report(
    cap_result,
    normal_result,
    outlier_count
):

    report = []

    # 정규성

    if normal_result["is_normal"]:

        report.append(
            "정규성 검정을 만족합니다."
        )

    else:

        report.append(
            "정규성 불만족 → Box-Cox 변환 수행"
            )

    # 공정능력

    cpk = cap_result["Cpk"]

    if cpk >= 1.33:

        report.append(
            f"Cpk={cpk:.3f}로 공정능력이 양호합니다."
        )

    elif cpk >= 1.00:

        report.append(
            f"Cpk={cpk:.3f}로 공정능력은 보통 수준입니다."
        )

    else:

        report.append(
            f"Cpk={cpk:.3f}로 공정 개선이 필요합니다."
        )

    # 이상점

    if outlier_count == 0:

        report.append(
            "관리도 이상점이 발견되지 않았습니다."
        )

    else:

        report.append(
            f"관리도 이상점 {outlier_count}개가 발견되었습니다."
        )

    return report

def plot_distribution(data):

    fig = px.histogram(
        x=pd.to_numeric(
            data,
            errors="coerce"
        ).dropna(),
        nbins=20,
        title="데이터 분포",
        marginal="box"
    )

    fig.update_layout(
        xaxis_title="측정값",
        yaxis_title="빈도"
    )

    return fig

def plot_qq(data):

    values = pd.to_numeric(
        data,
        errors="coerce"
    ).dropna()

    
    z_value = stats.zscore(values)

    (x, y), reg_line = stats.probplot(
        z_value,
        dist="norm"
    )

    fig = px.scatter(
        x=x,
        y=y,
        title="Q-Q Plot",
        labels={
            "x": "Theoretical Quantiles",
            "y": "Sample Quantiles"
        }
    )

    xmin = min(x)
    xmax = max(x)

    # 기준선
    fig.add_shape(
        type="line",
        x0=xmin,
        y0=xmin,
        x1=xmax,
        y1=xmax,
        line=dict(
            color="red",
            width=2
        )
    )

    fig.update_layout(
        height=450
    )

    return fig

def plot_boxcox_compare(
    original,
    transformed,
    lam
):

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "원본 데이터",
            f"Box-Cox 변환 (λ={lam:.2f})"
        )
    )

    fig.add_trace(
        go.Histogram(
            x=original,
            nbinsx=20,
            name="원본"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Histogram(
            x=transformed,
            nbinsx=20,
            name="변환"
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title="Box-Cox 변환 전후 비교",
        height=450,
        showlegend=False
    )

    return fig

def apply_boxcox(data):

    values = pd.to_numeric(
        data,
        errors="coerce"
    ).dropna()

    # Box-Cox는 양수만 가능

    if (values <= 0).any():
        shift = abs(values.min()) + 1
        values = values + shift

    transformed, lam = boxcox(values)

    if lam < -5 or lam > 5:
        return(
            pd.Series(transformed),
            lam,
            False
        )

    return (
        pd.Series(transformed),
        lam,
        True
    )


# Streamlit 화면 구성


st.markdown(
    """
    <div class="hero">
        <h1>🏭 Smart Manufacturing SPC Web App</h1>
        <p>공정능력분석 + 통계적공정관리 기반 제조공정 분석 대시보드</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ 설정")

uploaded_file = st.sidebar.file_uploader("CSV 데이터 업로드", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.sidebar.success(
        f"파일 업로드 완료\n{uploaded_file.name}"
    )

else:
    raw_df = make_sample_data()

    st.sidebar.info(
        "샘플 데이터를 사용 중입니다."
    )

df = preprocess_data(raw_df)

st.sidebar.subheader("데이터 편집")
use_editor = st.sidebar.checkbox("웹에서 데이터 직접 수정", value=True)

if use_editor:
    df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
    df = preprocess_data(df)

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# 메타 컬럼 자동 제거
numeric_cols = [
    col
    for col in numeric_cols
    if not is_meta_column(col)
]

all_cols = df.columns.tolist()


if len(numeric_cols) == 0:
    st.error("숫자형 컬럼이 필요합니다.")
    st.stop()

st.sidebar.subheader("분석 컬럼")
value_col = st.sidebar.selectbox(
    "분석 대상 컬럼",
    numeric_cols,
    index=numeric_cols.index("Thickness")
    if "Thickness" in numeric_cols else 0
)

subgroup_candidates = [
    col
    for col in all_cols
    if is_subgroup_column(col)
]

subgroup_options = (
    ["없음"]
    + subgroup_candidates
)

subgroup_col = st.sidebar.selectbox(
    "부분군 컬럼",
    subgroup_options,
    index=subgroup_options.index("Lot")
    if "Lot" in subgroup_options else 0
)


recommendation = recommend_control_chart(
    df=df,
    analysis_col=value_col,
    subgroup_col=None
    if subgroup_col == "없음"
    else subgroup_col
)

st.sidebar.markdown("---")

st.sidebar.success(
    f"""
데이터 유형 : {recommendation['data_type']}

추천 관리도 :
{recommendation['chart']}
"""
)

st.sidebar.info(
    recommendation["reason"]
)

st.sidebar.subheader("규격 입력")
default_center = float(pd.to_numeric(df[value_col], errors="coerce").mean())
lsl = st.sidebar.number_input("LSL", value=9.90, step=0.01, format="%.4f")
usl = st.sidebar.number_input("USL", value=10.10, step=0.01, format="%.4f")
target = st.sidebar.number_input("Target", value=10.00, step=0.01, format="%.4f")


# 관리도 선택


st.sidebar.subheader("관리도 선택")

data_type = st.sidebar.radio(
    "데이터 유형",
    ["계량형", "계수형"]
)

# 계량형 관리도


if data_type == "계량형":

    count_col = None
    sample_col = None

    if subgroup_col == "없음":

        chart_type = "I-MR"

        st.sidebar.success(
            "자동 선택 : I-MR\n(부분군 없음)"
        )

    else:

        subgroup_size = int(
            df.groupby(subgroup_col).size().mode().iloc[0]
        )

        if subgroup_size < 10:

            chart_type = "Xbar-R"

            st.sidebar.success(
                f"자동 선택 : Xbar-R\n(n={subgroup_size})"
            )

        else:

            chart_type = "Xbar-S"

            st.sidebar.success(
                f"자동 선택 : Xbar-S\n(n={subgroup_size})"
            )

    st.sidebar.info(
        "강의록 기준\n"
        '\n'
        "n=1 → I-MR\n"
        '\n'
        "2≤n<10 → Xbar-R\n"
        '\n'
        "n≥10 → Xbar-S"
    )


# 계수형 관리도


else:

    attribute_mode = st.sidebar.radio(
        "계수형 데이터 종류",
        [
            "불량 데이터",
            "결점 데이터"
        ]
    )

    count_col = st.sidebar.selectbox(
        "불량/결점 개수 컬럼",
        numeric_cols,
        index=numeric_cols.index("Defects")
        if "Defects" in numeric_cols else 0
    )

    sample_col = st.sidebar.selectbox(
        "표본 크기 컬럼",
        ["없음"] + numeric_cols,
        index=(["없음"] + numeric_cols).index("Sample_Size")
        if "Sample_Size" in numeric_cols else 0
    )


    # 불량 데이터


    if attribute_mode == "불량 데이터":

        if sample_col != "없음":

            sample_values = df[sample_col]

            if sample_values.nunique() == 1:

                chart_type = "P"

                st.sidebar.success(
                    "추천 관리도 : P\n(불량률 데이터)"
                )

                st.sidebar.info(
                    "표본 크기가 일정하므로\n"
                    '\n'
                    "NP 관리도도 사용 가능합니다."
                )

            else:

                chart_type = "P"

                st.sidebar.success(
                    "추천 관리도 : P\n(표본 크기 변동)"
                )

        else:

            chart_type = "NP"

            st.sidebar.success(
                "추천 관리도 : NP"
            )


    # 결점 데이터


    else:

        if sample_col != "없음":

            if df[sample_col].nunique() == 1:

                chart_type = "C"

                st.sidebar.success(
                    "추천 관리도 : C\n(결점수 관리)"
                )

            else:

                chart_type = "U"

                st.sidebar.success(
                    "추천 관리도 : U\n(단위당 결점수)"
                )

        else:

            chart_type = "C"

            st.sidebar.success(
                "추천 관리도 : C"
            )

    st.sidebar.info(
        "강의록 기준\n"
        '\n'
        "불량률 → P\n"
        '\n'
        "불량수 → NP\n"
        '\n'
        "결점수 → C\n"
        '\n'
        "단위당 결점수 → U"
    )



# 분석 실행


normal_result = normality_test(df[value_col])
cap_result = calculate_capability_indices(
    df=df,
    value_col=value_col,
    subgroup_col=None if subgroup_col == "없음" else subgroup_col,
    lsl=lsl,
    usl=usl
)

judgement, level = capability_level(cap_result["Cpk"])

grade = capability_grade(
    cap_result["Cpk"]
)

status_title, status_msg = process_status(
    cap_result["Cpk"]
)


# 다운로드용 결과 데이터


report_df = pd.DataFrame({

    "항목": [
        "Cp",
        "Cpk",
        "Pp",
        "Ppk",
        "정규성 p-value",
        "공정등급",
        "공정상태"
    ],

    "값": [
        round(cap_result["Cp"], 3),
        round(cap_result["Cpk"], 3),
        round(cap_result["Pp"], 3),
        round(cap_result["Ppk"], 3),
        round(normal_result["p_value"], 4),
        grade,
        status_title
    ]
})

if data_type == "계량형":
    main_chart, sub_chart = make_variable_control_chart(
        df=df,
        value_col=value_col,
        subgroup_col=None if subgroup_col == "없음" else subgroup_col,
        chart_type=chart_type
    )
    outlier_df = nelson_rule_1(main_chart)
else:
    main_chart = make_attribute_control_chart(
        df=df,
        count_col=count_col,
        sample_col=None if sample_col == "없음" else sample_col,
        subgroup_col=None if subgroup_col == "없음" else subgroup_col,
        chart_type=chart_type
    )
    sub_chart = None
    outlier_df = nelson_rule_1(main_chart)



# KPI 영역


from datetime import datetime

# 공정 상태 / 등급

grade = capability_grade(
    cap_result["Cpk"]
)

status_title, status_msg = process_status(
    cap_result["Cpk"]
)

# 다운로드용 결과표

report_df = pd.DataFrame({

    "항목": [
        "Cp",
        "Cpk",
        "Pp",
        "Ppk",
        "정규성 p-value",
        "공정등급",
        "공정상태"
    ],

    "값": [
        round(cap_result["Cp"], 3),
        round(cap_result["Cpk"], 3),
        round(cap_result["Pp"], 3),
        round(cap_result["Ppk"], 3),
        round(normal_result["p_value"], 4)
        if not pd.isna(normal_result["p_value"])
        else "N/A",
        grade,
        status_title
    ]
})

# KPI 카드

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    metric_card(
        "Cp",
        f"{cap_result['Cp']:.3f}",
        "단기 산포 기준",
        level
    )

with c2:
    metric_card(
        "Cpk",
        f"{cap_result['Cpk']:.3f}",
        judgement,
        level
    )

with c3:

    ppk_level = (
        "good"
        if cap_result["Ppk"] >= 1.33
        else "warn"
        if cap_result["Ppk"] >= 1.0
        else "bad"
    )

    metric_card(
        "Ppk",
        f"{cap_result['Ppk']:.3f}",
        "장기 공정성능",
        ppk_level
    )

with c4:

    norm_level = (
        "good"
        if normal_result["is_normal"]
        else "warn"
    )

    metric_card(
        "정규성 p-value",
        f"{normal_result['p_value']:.4f}"
        if not pd.isna(normal_result["p_value"])
        else "N/A",
        normal_result["method"],
        norm_level
    )

with c5:

    out_level = (
        "good"
        if len(outlier_df) == 0
        else "bad"
    )

    metric_card(
        "이상점",
        f"{len(outlier_df)}개",
        "관리한계 이탈",
        out_level
    )

st.markdown("<br>", unsafe_allow_html=True)

# 공정 상태 + 등급

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
### {status_title}

{status_msg}
"""
    )

with col2:

    st.info(
        f"""
### 🏆 공정능력 등급

현재 등급 : **{grade}**
"""
    )

st.markdown("<br>", unsafe_allow_html=True)

# 다운로드 버튼

today = datetime.now().strftime(
    "%Y%m%d_%H%M"
)

csv = report_df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="📥 분석 결과 다운로드",
    data=csv,
    file_name=f"process_report_{today}.csv",
    mime="text/csv"
)

st.markdown("<br>", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "① 데이터",
    "② 공정능력분석",
    "③ 정규성 검정",
    "④ 관리도",
    "⑤ 이상치 제거 후 재작성",
    "⑥ 자동 분석 리포트",
    "⑦ 함수 결과표"
])


with tab1:
    st.markdown("### 1) 데이터 준비 및 2) 데이터 전처리")
    st.write("샘플 데이터 또는 업로드된 CSV 데이터를 사용합니다. 웹에서 데이터를 직접 수정하면 아래 분석 결과가 즉시 바뀝니다.")
    st.dataframe(df, use_container_width=True)
    st.write("데이터 크기:", df.shape)
    st.write("컬럼:", df.columns.tolist())


with tab2:
    st.markdown("### 5) 공정능력지수 계산")
    left, right = st.columns([1.3, 1])

    with left:
        st.plotly_chart(
            plot_capability_histogram(df[value_col], lsl, usl, target),
            use_container_width=True
        )

    with right:
        cap_table = pd.DataFrame({
            "항목": list(cap_result.keys()),
            "값": list(cap_result.values())
        })
        st.dataframe(cap_table, use_container_width=True, hide_index=True)

    st.info(
        f"현재 Cpk는 {cap_result['Cpk']:.3f}이며 공정능력 판정은 '{judgement}'입니다. "
        "Cpk는 평균의 치우침과 산포를 동시에 고려합니다."
    )


with tab3:
    st.markdown(
    "### 3) 정규성 검정"
    )

    st.write(
        pd.DataFrame([normal_result])
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### 데이터 분포"
        )

        st.plotly_chart(
            plot_distribution(
                df[value_col]
            ),
            use_container_width=True
        )

    with col2:

        st.markdown(
            "#### Q-Q Plot"
        )

        st.plotly_chart(
            plot_qq(
                df[value_col]
            ),
            use_container_width=True
        )

    if normal_result['is_normal']:
        st.success(
            'p-value가 0.05 이상이므로 정규성을 크게 위배하지 않았습니다.'
        )

    else:
        st.warning(
            'p-value가 0.05미만이므로 정규성을 불만족 가능성이 있습니다.'
        )

        boxcox_data, lam, boxcox_valid = apply_boxcox(
            df[value_col]
        )

        st.warning(
            '정규성 불만족 -> Box-Cox 변환 수행'
        )

        st.write(
            f"Lambda = {lam:.4f}"
        )

        if not boxcox_valid:
            st.error(
                "Box-Cox λ 값이 너무 극단적입니다."
                "이 데이터에는 Box-Cox 변환 결과를 공정능력 계산에 사용하는 것이 적절하지 않습니다."
            )

        else:
            boxcox_normal = normality_test(
                boxcox_data
            )

            st.write(
                f"변환 후 p-value = {boxcox_normal['p_value']:.4f}"
            )

            if boxcox_normal['is_normal']:
                st.success(
                    "Box-Cox 변환 후 정규성 만족"
                )

            else:
                st.error(
                    "Box-Cox 변환 후에도 정규성 불만족"
                )

            st.plotly_chart(
                plot_boxcox_compare(
                    pd.to_numeric(
                        df[value_col],
                        errors='coerce'
                    ).dropna(),

                    boxcox_data,
                    lam
                ),
                use_container_width=True
            )

            st.dataframe(
                pd.DataFrame({
                    "Original":

                    pd.to_numeric(
                        df[value_col],
                        errors='coerce'
                    ).dropna().head(10).values,

                    "Transformed":
                    boxcox_data.head(10).values
                })
            )

            temp_df = df.copy()

            valid_idx = pd.to_numeric(
                temp_df[value_col],
                errors='coerce'
            ).dropna().index

            temp_df = temp_df.loc[
                valid_idx
            ].copy()

            temp_df[value_col] = (
                boxcox_data.values
            )

            boxcox_lsl = (
                boxcox_data.mean()
                -3 * boxcox_data.std(ddof=1)
            )

            boxcox_usl = (
                boxcox_data.mean()
                +3 * boxcox_data.std(ddof=1)
            )

            boxcox_cap = calculate_capability_indices(
                df = temp_df,
                value_col = value_col,

                subgroup_col = 
                None
                if subgroup_col == '없음'
                else subgroup_col,

                lsl = boxcox_lsl,
                usl = boxcox_usl
            )

            st.markdown(
                "### Box-Cox 변환 후 공정능력지수"
            )

            st.dataframe(

                pd.DataFrame({
                    "항목": [
                        "Cp",
                        "Cpk",
                        "Pp",
                        "Ppk",
                        "Box-Cox LSL",
                        "Box-Cox USL"
                    ],
                    "값": [
                        boxcox_cap["Cp"],
                        boxcox_cap["Cpk"],
                        boxcox_cap["Pp"],
                        boxcox_cap["Ppk"],
                        boxcox_lsl,
                        boxcox_usl
                    ]
                }),

                hide_index=True,

                use_container_width = True
            )



with tab4:
    st.markdown("### 2) 계량형 관리도 또는 3) 계수형 관리도 생성 + 4) 관리도 시각화")
    st.plotly_chart(
        plot_control_chart(main_chart, f"{chart_type} Main Control Chart"),
        use_container_width=True
    )

    if sub_chart is not None:
        st.plotly_chart(
            plot_control_chart(sub_chart, f"{chart_type} Sub Control Chart"),
            use_container_width=True
        )

    st.markdown("### 관리도 계산 결과")
    st.dataframe(main_chart, use_container_width=True, hide_index=True)

    if len(outlier_df) == 0:
        st.success("관리한계 밖 이상점이 없습니다.")
    else:
        st.error("관리한계 밖 이상점이 발견되었습니다.")
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)


with tab5:
    st.markdown("### 5) 이상치 제거 후 관리도 재작성하기")

    if data_type != "계량형":
        st.info("현재 예시는 계량형 관리도에 대해 이상치 제거 후 재작성 기능을 제공합니다.")
    else:
        filtered_df, new_main_chart, new_sub_chart = remove_outliers_and_recalculate(
            df=df,
            chart_df=main_chart,
            subgroup_col=None if subgroup_col == "없음" else subgroup_col,
            value_col=value_col,
            chart_type=chart_type
        )

        st.write("이상치 제거 전 데이터 수:", len(df))
        st.write("이상치 제거 후 데이터 수:", len(filtered_df))

        st.plotly_chart(
            plot_control_chart(new_main_chart, f"{chart_type} Recalculated Main Control Chart"),
            use_container_width=True
        )

        if new_sub_chart is not None:
            st.plotly_chart(
                plot_control_chart(new_sub_chart, f"{chart_type} Recalculated Sub Control Chart"),
                use_container_width=True
            )

        new_cap = calculate_capability_indices(
            filtered_df,
            value_col,
            None if subgroup_col == "없음" else subgroup_col,
            lsl,
            usl
        )

        st.markdown("### 이상치 제거 후 공정능력")
        st.dataframe(pd.DataFrame({
            "항목": list(new_cap.keys()),
            "값": list(new_cap.values())
        }), use_container_width=True, hide_index=True)

with tab6:

    report = generate_report(
        cap_result,
        normal_result,
        len(outlier_df)
    )

    st.markdown(
        "## 자동 분석 리포트"
    )

    for idx, item in enumerate(report, start=1):

        st.write(
            f"{idx}. {item}"
        )

with tab7:
    st.markdown("### 함수별 결과 확인")
    st.write("#### 불편화 상수")
    subgroup_n = 2
    if subgroup_col != "없음":
        subgroup_n = int(df.groupby(subgroup_col).size().mode().iloc[0])
    st.dataframe(pd.DataFrame([calculate_unbiased_constants(subgroup_n)]), use_container_width=True)

    st.write("#### 관리도용 불편화 상수 CSV")
    st.dataframe(control_const_df.reset_index(), use_container_width=True)

    st.write("#### 공정능력분석용 불편화 상수 CSV")
    st.dataframe(capability_const_df.reset_index(), use_container_width=True)

    st.write("#### 정규성 검정 결과")
    st.dataframe(pd.DataFrame([normal_result]), use_container_width=True)

    st.write("#### 공정능력지수 결과")
    st.dataframe(pd.DataFrame([cap_result]), use_container_width=True)

    st.write("#### 관리도 결과")
    st.dataframe(main_chart, use_container_width=True)
