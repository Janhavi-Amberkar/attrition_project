"""
Workforce Attrition Patterns & Risk Hotspot Analysis  |  Palo Alto Networks
Diagnostic (descriptive) dashboard built with Streamlit + Plotly.

Run:  streamlit run app.py
Data: cleaned_attrition.csv (or the raw "Palo Alto Networks.csv") in the same folder.
All derived columns are rebuilt here, so either file works.
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Attrition Risk Intelligence | Palo Alto Networks",
                   page_icon=":bar_chart:", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ palette
NAVY, BLUE, TEAL = "#0B2545", "#2E6F95", "#1B998B"
RED, AMBER, GREY, INK = "#C0392B", "#E08E0B", "#8A94A6", "#1B2A41"
GRID = "#E6EBF1"
HEAT = [[0.0, "#F2F6FA"], [0.45, "#F4CE98"], [0.75, "#E07B39"], [1.0, "#A93226"]]
STATUS_COLORS = {"Above overall": RED, "At / below overall": BLUE}
STATUS_ORDER = ["Above overall", "At / below overall"]

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px;}}
  .hero {{background: linear-gradient(120deg, {NAVY} 0%, #13315C 60%, {BLUE} 100%);
          padding: 1.25rem 1.6rem; border-radius: 12px; color: #fff; margin-bottom: 1rem;}}
  .hero h1 {{font-size: 1.65rem; margin: 0; font-weight: 700; color:#fff; padding:0;}}
  .hero p {{margin: .25rem 0 0 0; color: #C9D6E8; font-size: .92rem;}}
  .kpi {{background: #fff; border: 1px solid {GRID}; border-left: 5px solid {BLUE};
         border-radius: 10px; padding: .8rem 1rem; height: 100%;
         box-shadow: 0 1px 3px rgba(11,37,69,.06);}}
  .kpi.bad {{border-left-color: {RED};}}
  .kpi.good {{border-left-color: {TEAL};}}
  .kpi.warn {{border-left-color: {AMBER};}}
  .kpi-l {{font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: #5B6B82; font-weight: 600;}}
  .kpi-v {{font-size: 1.8rem; font-weight: 700; color: {NAVY}; line-height: 1.25;}}
  .kpi-s {{font-size: .8rem; color: #5B6B82;}}
  .card {{background: #F7F9FC; border: 1px solid {GRID}; border-radius: 10px; padding: .9rem 1.1rem;}}
  .card h4 {{margin: 0 0 .4rem 0; color: {NAVY}; font-size: 1rem;}}
  .note {{font-size: .8rem; color: #5B6B82;}}
  div[data-baseweb="tab-list"] {{gap: .25rem;}}
  button[data-baseweb="tab"] {{font-weight: 600;}}
  h2, h3 {{color: {NAVY};}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def _stretch(fn):
    """Version-proof 'full width' kwargs (Streamlit changed this API)."""
    params = inspect.signature(fn).parameters
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


_PC = _stretch(st.plotly_chart)
_DF = _stretch(st.dataframe)
_counter = [0]


def show(fig):
    _counter[0] += 1
    st.plotly_chart(fig, key=f"pc{_counter[0]}", **_PC)


def table(data, **cfg):
    st.dataframe(data, hide_index=True, column_config=cfg or None, **_DF)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, returned in %."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    n_safe = np.where(n == 0, np.nan, n)
    p = k / n_safe
    d = 1 + z * z / n_safe
    c = (p + z * z / (2 * n_safe)) / d
    h = z * np.sqrt(p * (1 - p) / n_safe + z * z / (4 * n_safe ** 2)) / d
    return (c - h) * 100, (c + h) * 100


def style(fig, title=None, height=380):
    fig.update_layout(
        template="plotly_white", height=height, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        title=dict(text=title, x=0, font=dict(size=15, color=NAVY)) if title else None,
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", y=-0.18, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def kpi(label, value, sub="", tone=""):
    return (f'<div class="kpi {tone}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load():
    here = Path(__file__).parent
    for name in ["cleaned_attrition.csv", "Palo Alto Networks.csv", "Palo_Alto_Networks.csv"]:
        if (here / name).exists():
            d = pd.read_csv(here / name)
            break
    else:
        st.error("Data file not found. Put cleaned_attrition.csv next to app.py.")
        st.stop()

    for c in d.select_dtypes(include=["object", "string"]).columns:
        d[c] = d[c].str.strip()
    if d["Attrition"].dtype == object:
        d["Attrition"] = d["Attrition"].map({"Yes": 1, "No": 0})
    d["Attrition"] = d["Attrition"].astype(int)

    d["BusinessTravel"] = d["BusinessTravel"].replace(
        {"Travel_Rarely": "Travel Rarely", "Travel_Frequently": "Travel Frequently"})
    d["Department"] = d["Department"].replace(
        {"Research & Development": "R&D", "Human Resources": "HR"})

    d["AttritionLabel"] = d["Attrition"].map({0: "Retained", 1: "Exited"})
    d["EducationLabel"] = pd.Categorical(
        d["Education"].map({1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}),
        ["Below College", "College", "Bachelor", "Master", "Doctor"], ordered=True)
    d["JobLevelLabel"] = pd.Categorical("Level " + d["JobLevel"].astype(str),
                                        [f"Level {i}" for i in range(1, 6)], ordered=True)

    def cut(col, bins, labels):
        return pd.cut(d[col], bins=bins, labels=labels)

    d["AgeGroup"] = cut("Age", [17, 25, 35, 45, 55, 100], ["18-25", "26-35", "36-45", "46-55", "56+"])
    d["TenureBand"] = cut("YearsAtCompany", [-1, 1, 2, 5, 10, 20, 100],
                          ["0-1", "2", "3-5", "6-10", "11-20", "21+"])
    d["CareerStage"] = cut("TotalWorkingYears", [-1, 5, 15, 100],
                           ["Early (0-5 yrs)", "Mid (6-15 yrs)", "Senior (16+ yrs)"])
    d["PromotionGap"] = cut("YearsSinceLastPromotion", [-1, 0, 2, 5, 100],
                            ["Promoted this year", "1-2 yrs", "3-5 yrs", "6+ yrs"])
    d["DistanceBand"] = cut("DistanceFromHome", [0, 5, 10, 20, 100],
                            ["1-5", "6-10", "11-20", "21+"])
    d["BusinessTravel"] = pd.Categorical(d["BusinessTravel"],
                                         ["Non-Travel", "Travel Rarely", "Travel Frequently"], ordered=True)
    expo = ((d["OverTime"] == "Yes").astype(int) +
            (d["BusinessTravel"] == "Travel Frequently").astype(int))
    d["WorkloadExposure"] = pd.Categorical(
        expo.map({0: "Neither", 1: "One factor", 2: "Both (OT + frequent travel)"}),
        ["Neither", "One factor", "Both (OT + frequent travel)"], ordered=True)
    return d


df = load()
N_ALL, EX_ALL = len(df), int(df["Attrition"].sum())
BASE = EX_ALL / N_ALL * 100          # fixed organisation-wide benchmark

# ------------------------------------------------------------------ sidebar
SB = st.sidebar
SB.markdown(f"<h3 style='margin:0;color:{NAVY}'>Filters</h3>", unsafe_allow_html=True)

all_depts = sorted(df["Department"].unique())
all_roles = sorted(df["JobRole"].unique())
tmin, tmax = int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())
amin, amax = int(df["Age"].min()), int(df["Age"].max())
travel_all = ["Non-Travel", "Travel Rarely", "Travel Frequently"]
edu_all = list(df["EducationLabel"].cat.categories)

DEFAULTS = {"f_dept": all_depts, "f_role": all_roles, "f_tenure": (tmin, tmax), "f_ot": "All",
            "f_travel": travel_all, "f_age": (amin, amax), "f_gender": ["Female", "Male"],
            "f_edu": edu_all, "f_early": 2, "f_min": 15, "f_ci": True, "_prev_depts": all_depts}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


SB.button("Reset all filters", on_click=reset_filters, use_container_width=True)

SB.markdown("**Workforce**")
depts = SB.multiselect("Department", all_depts, key="f_dept")
role_opts = sorted(df[df["Department"].isin(depts)]["JobRole"].unique())
if st.session_state["_prev_depts"] != depts:          # department change -> offer that department's roles
    st.session_state["f_role"] = role_opts
    st.session_state["_prev_depts"] = list(depts)
else:
    st.session_state["f_role"] = [r for r in st.session_state["f_role"] if r in role_opts]
roles = SB.multiselect("Job role", role_opts, key="f_role")
tenure_rng = SB.slider("Tenure range (years at company)", tmin, tmax, key="f_tenure")

SB.markdown("**Workload**")
ot_choice = SB.radio("Overtime", ["All", "Yes", "No"], horizontal=True, key="f_ot")
travel = SB.multiselect("Business travel", travel_all, key="f_travel")

with SB.expander("Demographic filters"):
    age_rng = st.slider("Age", amin, amax, key="f_age")
    genders = st.multiselect("Gender", ["Female", "Male"], key="f_gender")
    edus = st.multiselect("Education level", edu_all, key="f_edu")

with SB.expander("Analysis settings"):
    EARLY_X = st.slider("Early-tenure cutoff X (years)", 1, 5, key="f_early",
                        help="Early-tenure KPI = attrition among employees with YearsAtCompany <= X.")
    MIN_N = st.slider("Minimum group size", 5, 50, key="f_min",
                      help="Heatmap cells and hotspot rows below this size are flagged / masked.")
    SHOW_CI = st.toggle("Show 95% confidence intervals", key="f_ci")

f = df[df["Department"].isin(depts) & df["JobRole"].isin(roles)
       & df["YearsAtCompany"].between(*tenure_rng)
       & df["BusinessTravel"].isin(travel)
       & df["Age"].between(*age_rng)
       & df["Gender"].isin(genders) & df["EducationLabel"].isin(edus)]
if ot_choice != "All":
    f = f[f["OverTime"] == ot_choice]
SB.success(f"**{len(f):,}** of {N_ALL:,} employees selected")


# ------------------------------------------------------------------ analytics helpers
def rate_table(data, col):
    t = (data.groupby(col, observed=True)
             .agg(Employees=("Attrition", "size"), Exited=("Attrition", "sum")).reset_index())
    t["AttritionRate"] = (t.Exited / t.Employees * 100).round(2)
    lo, hi = wilson(t.Exited, t.Employees)
    t["CI_low"], t["CI_high"] = np.round(lo, 2), np.round(hi, 2)
    t["vs_overall_pp"] = (t.AttritionRate - BASE).round(2)
    t["Share_of_exits_%"] = (t.Exited / max(int(data["Attrition"].sum()), 1) * 100).round(1)
    return t


def tier(r):
    if r.Employees < MIN_N:
        return "Small sample"
    if r.CI_low > BASE:
        return "High risk"
    if r.CI_high < BASE:
        return "Low risk"
    return "Near baseline"


def bar_rate(t, col, title, horizontal=False, height=360):
    t = t.copy()
    if horizontal:
        t = t.sort_values("AttritionRate")
    t["Status"] = np.where(t.AttritionRate > BASE, STATUS_ORDER[0], STATUS_ORDER[1])
    kw = dict(color="Status", color_discrete_map=STATUS_COLORS, category_orders={"Status": STATUS_ORDER},
              text="AttritionRate",
              hover_data={"Employees": True, "Exited": True, "AttritionRate": ":.1f", "Status": False})
    top = t.CI_high if SHOW_CI else t.AttritionRate
    err = dict(type="data", symmetric=False, array=t.CI_high - t.AttritionRate,
               arrayminus=t.AttritionRate - t.CI_low, color=INK, thickness=1.2, width=4) if SHOW_CI else None
    lab = [f"{v:.1f}%" for v in t.AttritionRate]
    ghost = dict(mode="markers+text", text=lab, marker=dict(size=1, color="rgba(0,0,0,0)"),
                 showlegend=False, hoverinfo="skip", textfont=dict(size=12, color=INK))
    if horizontal:
        kw.pop("text")
        fig = px.bar(t, x="AttritionRate", y=col, orientation="h", **kw)
        fig.add_vline(x=BASE, line_dash="dash", line_color=NAVY,
                      annotation_text=f"Overall {BASE:.1f}%", annotation_position="top")
        if SHOW_CI:
            fig.add_trace(go.Scatter(x=t.AttritionRate, y=t[col].astype(str), mode="markers",
                                     marker=dict(size=1, color="rgba(0,0,0,0)"), showlegend=False,
                                     hoverinfo="skip", error_x=err))
        fig.add_trace(go.Scatter(x=top, y=t[col].astype(str), textposition="middle right", **ghost))
        fig.update_xaxes(title="Attrition rate (%)", range=[0, float(top.max()) * 1.18])
    else:
        kw.pop("text")
        fig = px.bar(t, x=col, y="AttritionRate", **kw)
        fig.add_hline(y=BASE, line_dash="dash", line_color=NAVY,
                      annotation_text=f"Overall {BASE:.1f}%", annotation_position="top right")
        if SHOW_CI:
            fig.add_trace(go.Scatter(x=t[col].astype(str), y=t.AttritionRate, mode="markers",
                                     marker=dict(size=1, color="rgba(0,0,0,0)"), showlegend=False,
                                     hoverinfo="skip", error_y=err))
        fig.add_trace(go.Scatter(x=t[col].astype(str), y=top, textposition="top center", **ghost))
        fig.update_xaxes(title=None, categoryorder="array", categoryarray=[str(v) for v in t[col]])
        fig.update_yaxes(title="Attrition rate (%)", range=[0, float(top.max()) * 1.18])
    return style(fig, title, height)


def heat(data, rows, cols, title, height=380):
    p = data.pivot_table(index=rows, columns=cols, values="Attrition", aggfunc="mean", observed=True) * 100
    n = data.pivot_table(index=rows, columns=cols, values="Attrition", aggfunc="size", observed=True)
    p = p.where(n >= MIN_N).round(1)
    fig = px.imshow(p, text_auto=".1f", aspect="auto", color_continuous_scale=HEAT, zmin=0)
    fig.update_coloraxes(colorbar=dict(title="%", thickness=12))
    fig.update_xaxes(title=None, side="top")
    fig.update_yaxes(title=None)
    return style(fig, title, height)


# ------------------------------------------------------------------ header
st.markdown(f"""
<div class="hero">
  <h1>Workforce Attrition Patterns &amp; Risk Hotspot Analysis</h1>
  <p>Palo Alto Networks &nbsp;|&nbsp; Diagnostic analysis of {N_ALL:,} employees &nbsp;|&nbsp;
     Organisation-wide baseline attrition: <b>{BASE:.2f}%</b> ({EX_ALL} exits)</p>
</div>
""", unsafe_allow_html=True)

if len(f) == 0:
    st.warning("No employees match the current filters. Adjust the sidebar or press 'Reset all filters'.")
    st.stop()

tabs = st.tabs(["Attrition Overview", "Department & Role Heatmaps", "Demographic Explorer",
                "Tenure & Workload Analysis", "Methodology & Data"])

# ================================================================== TAB 1
with tabs[0]:
    n, ex = len(f), int(f["Attrition"].sum())
    rt = ex / n * 100
    lo, hi = wilson(ex, n)
    early = f[f["YearsAtCompany"] <= EARLY_X]
    early_rt = early["Attrition"].mean() * 100 if len(early) else np.nan
    early_share = early["Attrition"].sum() / ex * 100 if ex else np.nan
    wl_both = f[f["WorkloadExposure"] == "Both (OT + frequent travel)"]
    wai = (wl_both["Attrition"].mean() * 100 / BASE * 100) if len(wl_both) else np.nan

    tone = "bad" if rt > BASE + 1 else ("good" if rt < BASE - 1 else "warn")
    c = st.columns(5)
    c[0].markdown(kpi("Employees in view", f"{n:,}", f"{n / N_ALL * 100:.0f}% of workforce"), unsafe_allow_html=True)
    c[1].markdown(kpi("Exited", f"{ex:,}", f"{ex / EX_ALL * 100:.0f}% of all exits", "bad"), unsafe_allow_html=True)
    c[2].markdown(kpi("Attrition rate", f"{rt:.2f}%",
                      f"{rt - BASE:+.2f} pp vs overall | 95% CI {lo:.1f}-{hi:.1f}%", tone), unsafe_allow_html=True)
    c[3].markdown(kpi(f"Early-tenure (<= {EARLY_X} yrs)",
                      "n/a" if np.isnan(early_rt) else f"{early_rt:.1f}%",
                      "n/a" if np.isnan(early_share) else f"{early_share:.0f}% of exits in view", "warn"),
                  unsafe_allow_html=True)
    c[4].markdown(kpi("Workload Attrition Index",
                      "n/a" if np.isnan(wai) else f"{wai:.0f}",
                      "OT + frequent travel (100 = baseline)", "bad"), unsafe_allow_html=True)
    st.write("")

    a, b = st.columns([1, 1.5])
    with a:
        pie = f["AttritionLabel"].value_counts().reindex(["Retained", "Exited"]).reset_index()
        pie.columns = ["Status", "Employees"]
        fig = px.pie(pie, names="Status", values="Employees", hole=0.62, color="Status",
                     color_discrete_map={"Retained": BLUE, "Exited": RED})
        fig.update_traces(textinfo="percent+value", sort=False)
        fig.add_annotation(text=f"<b>{rt:.1f}%</b><br>attrition", showarrow=False,
                           font=dict(size=18, color=NAVY))
        show(style(fig, "Retained vs exited", 340))
    with b:
        dep = rate_table(f, "Department")
        show(bar_rate(dep, "Department", "Attrition rate by department (%)", height=340))

    # --- auto-generated findings for the current selection
    def findings(d):
        out = []
        r = rate_table(d, "JobRole")
        r = r[r.Employees >= MIN_N].sort_values("AttritionRate", ascending=False)
        if len(r):
            x = r.iloc[0]
            out.append(f"Highest-attrition role: **{x.JobRole}** at **{x.AttritionRate:.1f}%** "
                       f"({int(x.Exited)} of {int(x.Employees)}), {x.AttritionRate / BASE:.1f}x the baseline.")
        r2 = rate_table(d, "JobRole").sort_values("Exited", ascending=False)
        if len(r2):
            x = r2.iloc[0]
            out.append(f"Largest source of exits: **{x.JobRole}** with **{int(x.Exited)}** exits "
                       f"({x['Share_of_exits_%']:.0f}% of exits in view).")
        t = rate_table(d, "TenureBand").sort_values("AttritionRate", ascending=False)
        t = t[t.Employees >= MIN_N]
        if len(t):
            x = t.iloc[0]
            out.append(f"Riskiest tenure band: **{x.TenureBand} years** at **{x.AttritionRate:.1f}%**.")
        o = rate_table(d, "OverTime").set_index("OverTime")
        if {"Yes", "No"} <= set(o.index) and o.loc["No", "AttritionRate"] > 0:
            out.append(f"Overtime workers leave at **{o.loc['Yes', 'AttritionRate']:.1f}%** vs "
                       f"**{o.loc['No', 'AttritionRate']:.1f}%** without overtime "
                       f"(**{o.loc['Yes', 'AttritionRate'] / o.loc['No', 'AttritionRate']:.1f}x**).")
        tr = rate_table(d, "BusinessTravel").set_index("BusinessTravel")
        if "Travel Frequently" in tr.index and "Non-Travel" in tr.index and tr.loc["Non-Travel", "AttritionRate"] > 0:
            out.append(f"Frequent travellers: **{tr.loc['Travel Frequently', 'AttritionRate']:.1f}%** vs "
                       f"**{tr.loc['Non-Travel', 'AttritionRate']:.1f}%** for non-travellers.")
        return out

    def bold(s):
        parts = s.split("**")
        return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))

    st.markdown(f'<div class="card"><h4>Key findings for the current selection</h4>'
                + "".join(f"<div style='margin:.2rem 0'>&bull; {bold(x)}</div>" for x in findings(f))
                + "</div>", unsafe_allow_html=True)

# ================================================================== TAB 2
with tabs[1]:
    role_t = rate_table(f, "JobRole")
    role_t["Risk tier"] = role_t.apply(tier, axis=1)

    a, b = st.columns(2)
    with a:
        show(bar_rate(rate_table(f, "Department"), "Department", "Department attrition rate (%)", height=380))
    with b:
        show(bar_rate(role_t, "JobRole", "Role attrition rate (%)", horizontal=True, height=380))

    a, b = st.columns(2)
    with a:   # Pareto of exits
        p = role_t.sort_values("Exited", ascending=False).copy()
        p["Cum"] = p.Exited.cumsum() / max(p.Exited.sum(), 1) * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=p.JobRole, y=p.Exited, name="Exits", marker_color=BLUE,
                             text=p.Exited, textposition="outside"), secondary_y=False)
        fig.add_trace(go.Scatter(x=p.JobRole, y=p.Cum, name="Cumulative % of exits", mode="lines+markers",
                                 line=dict(color=RED, width=2.5)), secondary_y=True)
        fig.update_yaxes(title="Exits", secondary_y=False, gridcolor=GRID)
        fig.update_yaxes(title="Cumulative %", range=[0, 105], tickvals=[0, 25, 50, 75, 100], secondary_y=True, showgrid=False)
        fig.update_xaxes(tickangle=-30, automargin=True)
        show(style(fig, "Pareto: where the exits come from", 420))
    with b:   # treemap
        tm = (f.groupby(["Department", "JobRole"], observed=True)
                .agg(Employees=("Attrition", "size"), Exited=("Attrition", "sum")).reset_index())
        tm["AttritionRate"] = (tm.Exited / tm.Employees * 100).round(1)
        fig = px.treemap(tm, path=[px.Constant("All"), "Department", "JobRole"], values="Employees",
                         color="AttritionRate", color_continuous_scale=HEAT,
                         range_color=[0, max(40, tm.AttritionRate.max())],
                         hover_data={"Exited": True, "AttritionRate": ":.1f"})
        fig.update_traces(textinfo="label+value", root_color="#F2F6FA")
        fig.update_coloraxes(colorbar=dict(title="Attr. %", thickness=12))
        show(style(fig, "Workforce size (area) vs attrition intensity (colour)", 420))

    show(heat(f, "JobRole", "Department", f"Attrition intensity: job role x department (%, cells < {MIN_N} employees masked)", 420))

    st.subheader("Risk hotspot ranking")
    st.caption("Tier uses the 95% confidence interval: **High risk** = whole interval above the overall rate; "
               "**Low risk** = whole interval below it; otherwise **Near baseline**. "
               "Groups under the minimum size are marked **Small sample**.")
    rank = role_t.sort_values("AttritionRate", ascending=False)[
        ["JobRole", "Employees", "Exited", "AttritionRate", "CI_low", "CI_high", "vs_overall_pp",
         "Share_of_exits_%", "Risk tier"]]
    table(rank,
          AttritionRate=st.column_config.ProgressColumn("Attrition %", format="%.1f", min_value=0, max_value=50),
          CI_low=st.column_config.NumberColumn("CI low", format="%.1f"),
          CI_high=st.column_config.NumberColumn("CI high", format="%.1f"),
          vs_overall_pp=st.column_config.NumberColumn("vs overall (pp)", format="%+.1f"),
          **{"Share_of_exits_%": st.column_config.NumberColumn("% of exits", format="%.1f")})
    st.download_button("Download ranking (CSV)", rank.to_csv(index=False).encode(), "role_risk_ranking.csv", "text/csv")

# ================================================================== TAB 3
with tabs[2]:
    dims = {"Age group": "AgeGroup", "Gender": "Gender", "Marital status": "MaritalStatus",
            "Education level": "EducationLabel", "Education field": "EducationField",
            "Job level": "JobLevelLabel", "Department": "Department"}
    a, b = st.columns([1, 3])
    pick = a.selectbox("Explore attrition by", list(dims))
    dim = dims[pick]
    t = rate_table(f, dim)
    with a:
        st.markdown(f'<div class="note">Bars above the dashed line exceed the organisation-wide '
                    f'{BASE:.1f}%. Whiskers are 95% confidence intervals: wide whiskers mean a small group.</div>',
                    unsafe_allow_html=True)
    with b:
        show(bar_rate(t, dim, f"Attrition rate by {pick.lower()} (%)", height=360))
    table(t[[dim, "Employees", "Exited", "AttritionRate", "CI_low", "CI_high", "vs_overall_pp", "Share_of_exits_%"]],
          AttritionRate=st.column_config.NumberColumn("Attrition %", format="%.1f"),
          vs_overall_pp=st.column_config.NumberColumn("vs overall (pp)", format="%+.1f"))

    st.subheader("Two-way explorer")
    c1, c2 = st.columns(2)
    r_name = c1.selectbox("Rows", list(dims), index=0)
    c_name = c2.selectbox("Columns", list(dims), index=2)
    if r_name == c_name:
        st.info("Pick two different dimensions.")
    else:
        show(heat(f, dims[r_name], dims[c_name],
                  f"Attrition rate (%): {r_name} x {c_name}", 400))

    a, b = st.columns(2)
    with a:
        fig = px.histogram(f, x="Age", color="AttritionLabel", nbins=30, barmode="overlay", opacity=0.75,
                           color_discrete_map={"Retained": BLUE, "Exited": RED},
                           category_orders={"AttritionLabel": ["Retained", "Exited"]})
        fig.update_yaxes(title="Employees")
        show(style(fig, "Age distribution: retained vs exited", 340))
    with b:
        g = rate_table(f, "Gender")
        e = rate_table(f, "EducationLabel")
        e["Dimension"] = "Education"; e = e.rename(columns={"EducationLabel": "Group"})
        g["Dimension"] = "Gender"; g = g.rename(columns={"Gender": "Group"})
        ge = pd.concat([g, e])
        ge["Status"] = np.where(ge.AttritionRate > BASE, STATUS_ORDER[0], STATUS_ORDER[1])
        fig = px.bar(ge, x="Group", y="AttritionRate", color="Status", color_discrete_map=STATUS_COLORS,
                     category_orders={"Status": STATUS_ORDER}, text="AttritionRate",
                     hover_data={"Employees": True, "Exited": True, "Status": False})
        fig.add_hline(y=BASE, line_dash="dash", line_color=NAVY)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_xaxes(title=None); fig.update_yaxes(title="Attrition rate (%)", rangemode="tozero")
        show(style(fig, "Gender and education level (no strong pattern)", 340))

# ================================================================== TAB 4
with tabs[3]:
    st.markdown("##### Tenure and career stage")
    a, b = st.columns(2)
    with a:
        show(bar_rate(rate_table(f, "TenureBand"), "TenureBand", "Attrition by tenure band (years at company)", height=360))
    with b:
        by = (f[f["YearsAtCompany"] <= 20].groupby("YearsAtCompany")
                .agg(Employees=("Attrition", "size"), Exited=("Attrition", "sum")).reset_index())
        by["Rate"] = by.Exited / by.Employees * 100
        by["lo"], by["hi"] = wilson(by.Exited, by.Employees)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=by.YearsAtCompany, y=by.hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=by.YearsAtCompany, y=by.lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(46,111,149,.15)", name="95% CI",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=by.YearsAtCompany, y=by.Rate, mode="lines+markers", name="Attrition %",
                                 line=dict(color=BLUE, width=2.5), marker=dict(size=by.Employees.clip(upper=100) / 8 + 4),
                                 customdata=by[["Employees", "Exited"]],
                                 hovertemplate="Year %{x}<br>%{y:.1f}%<br>%{customdata[0]} employees, "
                                               "%{customdata[1]} exits<extra></extra>"))
        fig.add_hline(y=BASE, line_dash="dash", line_color=NAVY, annotation_text=f"Overall {BASE:.1f}%")
        fig.update_xaxes(title="Years at company"); fig.update_yaxes(title="Attrition rate (%)", rangemode="tozero")
        show(style(fig, "Attrition by exact year (marker size = headcount)", 360))

    a, b, c = st.columns(3)
    with a:
        show(bar_rate(rate_table(f, "CareerStage"), "CareerStage", "Career stage (total experience)", height=340))
    with b:
        show(bar_rate(rate_table(f, "JobLevelLabel"), "JobLevelLabel", "Job level", height=340))
    with c:
        show(bar_rate(rate_table(f, "PromotionGap"), "PromotionGap", "Years since last promotion", height=340))

    early = f[f["YearsAtCompany"] <= EARLY_X]
    later = f[f["YearsAtCompany"] > EARLY_X]
    if len(early) and len(later) and later["Attrition"].mean() > 0:
        st.markdown(
            f'<div class="card"><h4>Early-tenure attrition (first {EARLY_X} years)</h4>'
            f'<b>{early["Attrition"].mean() * 100:.1f}%</b> of employees with up to {EARLY_X} years leave, against '
            f'<b>{later["Attrition"].mean() * 100:.1f}%</b> afterwards '
            f'(<b>{early["Attrition"].mean() / later["Attrition"].mean():.2f}x</b>). '
            f'They are {len(early) / len(f) * 100:.0f}% of staff in view but '
            f'{early["Attrition"].sum() / max(f["Attrition"].sum(), 1) * 100:.0f}% of exits.</div>',
            unsafe_allow_html=True)

    st.markdown("##### Workload and mobility")
    a, b, c = st.columns(3)
    with a:
        show(bar_rate(rate_table(f, "OverTime"), "OverTime", "Overtime vs no overtime", height=340))
    with b:
        show(bar_rate(rate_table(f, "BusinessTravel"), "BusinessTravel", "Business travel frequency", height=340))
    with c:
        show(bar_rate(rate_table(f, "DistanceBand"), "DistanceBand", "Distance from home (band)", height=340))

    a, b = st.columns(2)
    with a:
        wi = rate_table(f, "WorkloadExposure")
        wi["WAI"] = (wi.AttritionRate / BASE * 100).round(0)
        wi["Status"] = np.where(wi.WAI > 100, STATUS_ORDER[0], STATUS_ORDER[1])
        fig = px.bar(wi, x="WorkloadExposure", y="WAI", color="Status", color_discrete_map=STATUS_COLORS,
                     category_orders={"Status": STATUS_ORDER,
                                      "WorkloadExposure": ["Neither", "One factor", "Both (OT + frequent travel)"]},
                     text="WAI",
                     hover_data={"Employees": True, "Exited": True, "AttritionRate": ":.1f", "Status": False})
        fig.add_hline(y=100, line_dash="dash", line_color=NAVY, annotation_text="Baseline = 100")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(title=None); fig.update_yaxes(title="Index", rangemode="tozero")
        show(style(fig, "Workload Attrition Index (group rate / overall rate x 100)", 380))
    with b:
        show(heat(f, "OverTime", "BusinessTravel", f"Overtime x business travel (%, cells < {MIN_N} masked)", 380))

    ft = f.assign(Tenure=np.where(f.YearsAtCompany <= EARLY_X, f"Early (<= {EARLY_X} yrs)", f"Later (> {EARLY_X} yrs)"))
    ot_t = (ft.groupby(["Tenure", "OverTime"]).agg(Employees=("Attrition", "size"), Exited=("Attrition", "sum"))
              .reset_index())
    ot_t["AttritionRate"] = (ot_t.Exited / ot_t.Employees * 100).round(1)
    fig = px.bar(ot_t, x="Tenure", y="AttritionRate", color="OverTime", barmode="group", text="AttritionRate",
                 color_discrete_map={"No": BLUE, "Yes": RED}, hover_data=["Employees", "Exited"])
    fig.add_hline(y=BASE, line_dash="dash", line_color=NAVY)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_xaxes(title=None); fig.update_yaxes(title="Attrition rate (%)", rangemode="tozero")
    show(style(fig, "Does overtime matter after accounting for tenure?", 360))

# ================================================================== TAB 5
with tabs[4]:
    a, b = st.columns([1.2, 1])
    with a:
        st.subheader("KPI definitions")
        st.markdown(f"""
| KPI | Definition |
|---|---|
| **Attrition rate** | Employees who left / total employees. Baseline = {EX_ALL} / {N_ALL:,} = **{BASE:.2f}%** |
| **Department / role attrition rate** | Same ratio within each department or job role |
| **Early-tenure attrition** | Attrition among employees with `YearsAtCompany` <= X (X set in sidebar, default 2) |
| **Workload Attrition Index (WAI)** | (Attrition rate of a workload group / overall rate) x 100. Workload exposure counts overtime and frequent travel: Neither, One factor, Both |
| **Risk tier** | High / Low risk when the 95% Wilson interval lies wholly above / below the baseline |
""")
        st.subheader("How to read the charts")
        st.markdown("""
- **Colour:** red bars exceed the organisation-wide rate, blue bars do not.
- **Whiskers:** 95% confidence interval. Wide whiskers mean the group is small, so treat the rate cautiously.
- **Masked heatmap cells:** combinations with fewer employees than the minimum group size are hidden.
- **Scope:** the data is a single snapshot, so the findings describe **associations, not causes**.
""")
    with b:
        st.subheader("Dataset summary")
        q = pd.DataFrame({
            "Check": ["Rows", "Columns", "Missing values", "Duplicate rows", "Exited / retained"],
            "Result": [f"{N_ALL:,}", f"{df.shape[1]}", f"{int(df.isna().sum().sum())}",
                       f"{int(df.duplicated().sum())}", f"{EX_ALL} / {N_ALL - EX_ALL}"]})
        table(q)
        st.subheader("Export")
        st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode(),
                           "filtered_attrition.csv", "text/csv")
        with st.expander("Preview filtered data"):
            table(f.head(200))
