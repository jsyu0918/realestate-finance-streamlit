# =============================================================
# 부동산금융 수업용 Streamlit 예제 앱
# - 실행: 터미널에서  streamlit run app.py
# - 구성: 왼쪽 사이드바(대출 조건 입력) + 3개 탭
#   ① 대출 상환 계산기  ② DSR 점검  ③ 임대 투자수익률
# - 금액 단위: 화면 입력은 억원/만원, 내부 계산은 모두 "만원"
# =============================================================

import numpy as np                     # 숫자 계산
import pandas as pd                    # 표(데이터프레임)
import plotly.express as px            # 간단한 차트
import plotly.graph_objects as go      # 세밀한 차트
from plotly.subplots import make_subplots  # 보조 y축이 있는 차트
import streamlit as st                 # 웹 앱 화면

# -------------------------------------------------------------
# 0. 페이지 기본 설정 (앱 맨 위에 한 번만 호출)
# -------------------------------------------------------------
st.set_page_config(page_title="부동산금융 실습", page_icon="🏠", layout="wide")

EOK = 10_000  # 1억원 = 10,000만원


# -------------------------------------------------------------
# 1. 계산 함수들 (화면과 분리해 두면 테스트하기 쉬움)
# -------------------------------------------------------------
def fmt_money(manwon: float) -> str:
    """만원 단위 숫자를 '3억 2,000만원' 형태의 글자로 바꿈"""
    sign = "-" if manwon < 0 else ""
    manwon = abs(round(manwon))
    eok, man = divmod(manwon, EOK)
    if eok and man:
        return f"{sign}{eok:,}억 {man:,}만원"
    if eok:
        return f"{sign}{eok:,}억원"
    return f"{sign}{man:,}만원"


def loan_schedule(principal: float, annual_rate: float, years: int, method: str) -> pd.DataFrame:
    """월별 상환 스케줄 표를 만듦
    principal   : 대출금액(만원)
    annual_rate : 연 금리(%)  예) 4.0
    years       : 만기(년)
    method      : '원리금균등' / '원금균등' / '만기일시'
    """
    n = years * 12                 # 총 상환 개월 수
    r = annual_rate / 100 / 12     # 월 금리
    balance = principal            # 남은 대출 잔액
    rows = []

    # 원리금균등: 매달 같은 금액(원금+이자)을 냄 → 연금(annuity) 공식
    if method == "원리금균등":
        pmt = principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)

    for m in range(1, n + 1):
        interest = balance * r                 # 이번 달 이자 = 잔액 × 월금리
        if method == "원리금균등":
            principal_paid = pmt - interest    # 낸 돈에서 이자를 뺀 나머지가 원금
        elif method == "원금균등":
            principal_paid = principal / n     # 원금을 매달 똑같이 나눠서 갚음
        else:  # 만기일시: 매달 이자만, 마지막 달에 원금 전부
            principal_paid = principal if m == n else 0.0
        balance -= principal_paid
        rows.append({
            "월": m,
            "연차": (m - 1) // 12 + 1,
            "원금": principal_paid,
            "이자": interest,
            "월상환액": principal_paid + interest,
            "잔액": max(balance, 0.0),
        })
    return pd.DataFrame(rows)


def yearly_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """월별 표를 연도별로 합계 (잔액은 그 해 말 기준)"""
    return schedule.groupby("연차").agg(
        원금=("원금", "sum"), 이자=("이자", "sum"), 잔액=("잔액", "last")
    ).reset_index()


def first_year_payment(principal: float, annual_rate: float, years: int, method: str) -> float:
    """첫 1년(12개월) 동안 내는 원리금 합계(만원) — DSR·현금수익률 계산에 사용"""
    sch = loan_schedule(principal, annual_rate, years, method)
    return float(sch.loc[sch["월"] <= 12, "월상환액"].sum())


def calc_dsr(annual_debt_service: float, other_debt_service: float, annual_income: float) -> float:
    """DSR(%) = (모든 대출의 연간 원리금 상환액) ÷ 연소득 × 100  (교육용 단순 계산)"""
    if annual_income <= 0:
        return float("nan")
    return (annual_debt_service + other_debt_service) / annual_income * 100


def rental_returns(price: float, annual_rent: float, vacancy: float, opex_ratio: float,
                   loan: float, debt_service: float) -> dict:
    """임대 투자 지표 계산 (금액: 만원, 비율: %)
    - NOI(순영업소득) = 연 임대료 × (1-공실률) × (1-운영비용률)
    - Cap rate      = NOI ÷ 매입가
    - 자기자본수익률 = (NOI - 연 원리금 상환액) ÷ 자기자본   (cash-on-cash)
    """
    egi = annual_rent * (1 - vacancy / 100)        # 유효총소득(공실 반영)
    noi = egi * (1 - opex_ratio / 100)             # 운영비용을 뺀 순영업소득
    equity = price - loan                          # 자기자본 = 매입가 - 대출
    cap_rate = noi / price * 100 if price > 0 else float("nan")
    cash_flow = noi - debt_service                 # 세전 현금흐름
    coc = cash_flow / equity * 100 if equity > 0 else float("nan")
    return {"EGI": egi, "NOI": noi, "자기자본": equity, "Cap rate": cap_rate,
            "현금흐름": cash_flow, "자기자본수익률": coc}


def sensitivity_table(price, annual_rent, opex_ratio, loan, years, method,
                      rates=None, vacancies=None) -> pd.DataFrame:
    """금리 × 공실률 조합별 자기자본수익률(%) 표"""
    rates = np.arange(2.0, 7.01, 0.5) if rates is None else rates
    vacancies = np.arange(0, 20.01, 2.5) if vacancies is None else vacancies
    table = {}
    for rate in rates:
        ds = first_year_payment(loan, rate, years, method)
        table[round(float(rate), 1)] = [
            rental_returns(price, annual_rent, v, opex_ratio, loan, ds)["자기자본수익률"]
            for v in vacancies
        ]
    df = pd.DataFrame(table, index=[round(float(v), 1) for v in vacancies])
    df.index.name = "공실률(%)"
    df.columns.name = "대출금리(%)"
    return df


# -------------------------------------------------------------
# 2. 사이드바: 모든 탭에서 같이 쓰는 대출 조건
# -------------------------------------------------------------
st.sidebar.header("🏦 대출 조건 입력")
st.sidebar.caption("여기서 바꾼 값이 세 탭 모두에 반영됩니다.")

price_eok = st.sidebar.number_input("주택가격 (억원)", min_value=0.5, max_value=100.0,
                                    value=10.0, step=0.5)
ltv = st.sidebar.slider("LTV (%)", min_value=0, max_value=90, value=50, step=5)
rate = st.sidebar.slider("대출금리 (연 %)", min_value=0.5, max_value=10.0, value=4.0, step=0.1)
years = st.sidebar.slider("만기 (년)", min_value=5, max_value=40, value=30, step=5)
method = st.sidebar.radio("상환방식", ["원리금균등", "원금균등", "만기일시"])

st.sidebar.divider()
st.sidebar.caption("단국대학교 도시계획·부동산학부 · 부동산금융 실습용 예제")

# 공통 계산 (탭 1·2·3에서 재사용)
price = price_eok * EOK                     # 주택가격(만원)
loan = price * ltv / 100                    # 대출금액(만원)
schedule = loan_schedule(loan, rate, years, method)
yearly = yearly_summary(schedule)
annual_ds = first_year_payment(loan, rate, years, method)  # 첫해 연간 원리금

# -------------------------------------------------------------
# 3. 본문: 제목과 3개 탭
# -------------------------------------------------------------
st.title("🏠 부동산금융 실습 계산기")
st.write("왼쪽에서 대출 조건을 바꾸면 아래 결과가 바로 다시 계산됩니다.")

tab1, tab2, tab3 = st.tabs(["① 대출 상환 계산기", "② DSR 점검", "③ 임대 투자수익률"])

# ---------------- 탭 1: 대출 상환 계산기 ----------------
with tab1:
    st.subheader("대출 상환 계산기")

    # 핵심 숫자 3개를 카드(metric)로 보여줌
    c1, c2, c3 = st.columns(3)
    c1.metric("대출금액", fmt_money(loan), help="주택가격 × LTV")
    c2.metric("첫 달 월상환액", fmt_money(schedule["월상환액"].iloc[0]) if loan > 0 else "0만원")
    c3.metric("총이자", fmt_money(schedule["이자"].sum()))

    # 연도별 원금/이자(막대) + 잔액(선) 차트
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=yearly["연차"], y=yearly["원금"], name="원금 상환")
    fig.add_bar(x=yearly["연차"], y=yearly["이자"], name="이자 상환")
    fig.add_scatter(x=yearly["연차"], y=yearly["잔액"], name="대출 잔액(연말)",
                    mode="lines+markers", secondary_y=True)
    fig.update_layout(barmode="stack", height=420, title="연도별 원금·이자 상환액과 대출 잔액",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="연차")
    fig.update_yaxes(title_text="연간 상환액 (만원)", tickformat=",", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="잔액 (만원)", tickformat=",.0f", rangemode="tozero", showgrid=False, secondary_y=True)
    st.plotly_chart(fig)

    # 표로도 확인 (펼쳐서 보기)
    with st.expander("연도별 상환표 보기"):
        st.dataframe(yearly.round(0), hide_index=True)

    with st.expander("💬 수업 활용 질문"):
        st.markdown(
            "1. 같은 금리·만기에서 **원리금균등**과 **원금균등**의 총이자가 다른 이유는 무엇일까요?\n"
            "2. **만기일시** 상환은 초기 부담이 작지만 위험이 큽니다. 차주와 은행 입장에서 각각 어떤 위험이 있을까요?"
        )

# ---------------- 탭 2: DSR 점검 ----------------
with tab2:
    st.subheader("DSR(총부채원리금상환비율) 점검")

    c1, c2, c3 = st.columns(3)
    income = c1.number_input("연소득 (만원)", min_value=0, value=9000, step=500)
    other_ds = c2.number_input("기타 대출 연상환액 (만원)", min_value=0, value=300, step=100)
    limit = c3.slider("규제 한도 (%)", min_value=20, max_value=70, value=40, step=5)

    dsr = calc_dsr(annual_ds, other_ds, income)

    m1, m2, m3 = st.columns(3)
    m1.metric("이번 주택대출 연상환액(첫해)", fmt_money(annual_ds))
    m2.metric("총 연간 원리금", fmt_money(annual_ds + other_ds))
    m3.metric("DSR", "계산 불가" if np.isnan(dsr) else f"{dsr:.1f}%",
              delta=None if np.isnan(dsr) else f"{dsr - limit:+.1f}%p (한도 대비)",
              delta_color="inverse")

    # 통과/초과 메시지
    if np.isnan(dsr):
        st.warning("연소득을 입력해 주세요.")
    elif dsr <= limit:
        st.success(f"✅ 통과: DSR {dsr:.1f}% ≤ 규제 한도 {limit}%")
    else:
        # 한도 안에서 받을 수 있는 이 대출의 연상환액 여유분
        room = income * limit / 100 - other_ds
        st.error(f"❌ 한도 초과: DSR {dsr:.1f}% > 규제 한도 {limit}% "
                 f"(이 대출의 연상환액이 {fmt_money(max(room, 0))} 이하여야 통과)")

    # 게이지처럼 보이는 간단한 막대 차트
    bar = go.Figure(go.Bar(x=[0 if np.isnan(dsr) else dsr], y=["DSR"], orientation="h",
                           marker_color="crimson" if (not np.isnan(dsr) and dsr > limit) else "seagreen"))
    bar.add_vline(x=limit, line_dash="dash", annotation_text=f"한도 {limit}%")
    bar.update_layout(height=180, xaxis=dict(range=[0, max(100, (0 if np.isnan(dsr) else dsr) + 10)],
                                             title="%"), margin=dict(t=20, b=40))
    st.plotly_chart(bar)

    st.caption("※ 교육용 단순 계산입니다. 이번 대출의 첫해 원리금과 기타 대출 상환액을 연소득으로 나눈 값이며, "
               "실제 금융권의 공식 DSR·스트레스 DSR(가산금리, 대출 종류별 만기 산정 방식 등)과는 다릅니다.")

    with st.expander("💬 수업 활용 질문"):
        st.markdown(
            "1. 상환방식을 **만기일시**로 바꾸면 이 계산의 DSR은 낮아집니다. 실제 규제에서는 이런 '착시'를 어떻게 막을까요?\n"
            "2. 금리를 1%p 올려 보세요. 스트레스 DSR이 금리 상승 위험을 미리 반영하려는 이유는 무엇일까요?"
        )

# ---------------- 탭 3: 임대 투자수익률 ----------------
with tab3:
    st.subheader("임대 투자수익률 분석")
    st.caption("대출은 사이드바의 LTV·금리·만기·상환방식을 그대로 적용합니다 (대출금액 = 매입가 × LTV).")

    c1, c2, c3, c4 = st.columns(4)
    buy_eok = c1.number_input("매입가 (억원)", min_value=0.5, max_value=100.0,
                              value=float(price_eok), step=0.5)
    rent = c2.number_input("연 임대료 (만원)", min_value=0, value=6000, step=100)
    vacancy = c3.slider("공실률 (%)", min_value=0, max_value=50, value=5)
    opex = c4.slider("운영비용률 (%)", min_value=0, max_value=60, value=20,
                     help="유효임대료 대비 관리비·세금·보험 등 운영비 비율")

    buy_price = buy_eok * EOK
    inv_loan = buy_price * ltv / 100                         # 투자용 대출금액
    inv_ds = first_year_payment(inv_loan, rate, years, method)  # 첫해 연 원리금
    res = rental_returns(buy_price, rent, vacancy, opex, inv_loan, inv_ds)

    # 첫째 줄: 자금 구조 / 둘째 줄: 수익률 지표
    m1, m2, m3 = st.columns(3)
    m1.metric("자기자본", fmt_money(res["자기자본"]), help="매입가 - 대출금액")
    m2.metric("대출금액", fmt_money(inv_loan), help="매입가 × LTV")
    m3.metric("첫해 연 원리금", fmt_money(inv_ds))
    m4, m5, m6 = st.columns(3)
    m4.metric("NOI (순영업소득)", fmt_money(res["NOI"]), help="연 임대료 × (1-공실률) × (1-운영비용률)")
    m5.metric("Cap rate", f"{res['Cap rate']:.2f}%", help="NOI ÷ 매입가")
    m6.metric("자기자본수익률 (cash-on-cash)",
              "계산 불가" if np.isnan(res["자기자본수익률"]) else f"{res['자기자본수익률']:.2f}%")

    # 레버리지 효과 한 줄 설명
    if res["Cap rate"] > rate:
        st.info(f"📈 **정(+)의 레버리지**: Cap rate({res['Cap rate']:.2f}%)가 대출금리({rate:.1f}%)보다 높으면 "
                "빚을 낼수록 자기자본수익률이 올라갑니다.")
    else:
        st.warning(f"📉 **부(−)의 레버리지**: Cap rate({res['Cap rate']:.2f}%)가 대출금리({rate:.1f}%)보다 낮으면 "
                   "빚을 낼수록 자기자본수익률이 떨어집니다.")
    # 대출상수 K = 연 원리금 ÷ 대출금액 (원금 상환까지 포함한 '실질 대출 비용률')
    k = inv_ds / inv_loan * 100 if inv_loan > 0 else float("nan")
    st.caption("※ 자기자본수익률은 NOI에서 첫해 원리금(원금 포함)을 뺀 현금흐름 기준입니다. "
               "원금까지 갚는 방식에서는 금리 대신 대출상수 K(= 연 원리금 ÷ 대출금"
               + ("" if np.isnan(k) else f", 현재 {k:.2f}%")
               + ")가 기준이 되어, Cap rate > K 일 때 자기자본수익률이 Cap rate보다 높아집니다.")

    # 민감도 분석: 금리 × 공실률
    st.markdown("#### 민감도 분석: 대출금리 × 공실률 → 자기자본수익률(%)")
    sens = sensitivity_table(buy_price, rent, opex, inv_loan, years, method)
    heat = px.imshow(sens.round(1), text_auto=True, aspect="auto",
                     color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                     labels=dict(x="대출금리 (%)", y="공실률 (%)", color="수익률(%)"))
    heat.update_xaxes(dtick=0.5, ticksuffix="%", side="bottom")
    heat.update_yaxes(dtick=2.5, ticksuffix="%")
    heat.update_layout(height=440)
    st.plotly_chart(heat)

    with st.expander("민감도 표(숫자) 보기"):
        st.dataframe(sens.round(2))

    with st.expander("💬 수업 활용 질문"):
        st.markdown(
            "1. 표에서 자기자본수익률이 0% 아래로 떨어지는 금리·공실률 조합은 어디인가요? 이 경계가 의미하는 바는?\n"
            "2. LTV를 30%와 70%로 바꿔 보세요. 정(+)의 레버리지와 부(−)의 레버리지에서 결과가 어떻게 달라지나요?"
        )
