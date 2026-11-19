"""ACDC member explorer: interactive EDA dashboard for the Data Mining project (Deliverable 1 bonus).

Run from the repository root:  python dashboard/app.py   then open http://127.0.0.1:8050
The app reads data/processed/acdc_dash.parquet, created by notebook/03_dash_features.ipynb.
"""

# %% Imports and paths
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from lifelines import KaplanMeierFitter

# The notebook has no __file__, so it falls back to the working directory (notebook/).
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().parent / "dashboard"
ROOT = HERE.parent
DATA_FILE = ROOT / "data" / "processed" / "acdc_dash.parquet"

# %% Labels, order and colours
PLAN_LABEL = {"hot_desk": "Hot desk", "flex": "Flex", "dedicated_desk": "Dedicated desk",
              "full_time": "Full time", "private_office": "Private office"}
STATE_ORDER = ["New", "Active", "Dormant", "Churned"]

# Plan palette (charcoal, violet, amber, burgundy, coral); each plan keeps its colour on every chart.
PLAN_COLOR = dict(zip(PLAN_LABEL.values(), ["#1d212a", "#6b5bbf", "#ffc000", "#5a0e24", "#ee6a4d"]))
STATE_COLOR = {"New": "#5a0e24", "Active": "#ffc000", "Dormant": "#1d212a", "Churned": "#d6d0c8"}
ACCENT = "#a11a5b"  # deep berry from the rocket ramp, for single-series charts
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
# Sequential: seaborn "rocket" (reversed, cream to plum, darkest end trimmed). Diverging: teal - cream - berry.
ROCKET = [[0.0, "#faebdd"], [0.125, "#f6c19f"], [0.25, "#f5946b"], [0.375, "#f16244"], [0.5, "#de2e44"], [0.625, "#b91657"], [0.75, "#891e5b"], [0.875, "#5b1e51"], [1.0, "#30173a"]]
DIVERGING = [[0, "#1d6b63"], [0.25, "#7fbfb3"], [0.5, "#f5efe6"], [0.75, "#e0708a"], [1, "#8a1650"]]

INDUSTRY_LABEL = {"information technology": "Information technology", "consulting": "Consulting",
                  "design & creative": "Design & creative", "marketing & pr": "Marketing & PR",
                  "finance": "Finance", "legal": "Legal", "education": "Education", "health": "Health",
                  "real estate": "Real estate", "other": "Other"}
ADDONS = {"uses_meeting_rooms": "Meeting rooms", "uses_event_space": "Event space",
          "uses_guest_passes": "Guest passes", "uses_print": "Print", "uses_mail_handling": "Mail handling",
          "uses_phone_booths": "Phone booths", "uses_day_passes": "Day passes", "uses_storage": "Storage",
          "uses_bike_locker": "Bike locker"}
FLAGS = {"flag_days_code_999": "Days since visit = 999 (code)",
         "flag_last_period_above_total": "Last-period visits above total",
         "flag_colleagues": "Colleagues = -1 (code)", "flag_negative_value": "Negative visits, renewals or value",
         "flag_start_after_end": "Start date after end date", "flag_age": "Age outside 16-100",
         "flag_swipe_outside": "Swipe outside membership", "flag_plan_days": "Plan days 0 or 9",
         "flag_swipe_after_end": "Swipe after end date (kept)", "flag_negative_addon": "Negative add-on hours",
         "flag_entitled_visits": "Entitled visits = 0", "flag_extreme_value": "Monthly value > 10x plan median (kept)"}
NUMERIC = {"monthly_value": "Monthly value (EUR)", "utilization_rate": "Utilization of entitled visits",
           "tenure_months": "Tenure (months)", "avg_weekly_visits": "Average weekly visits",
           "visits_per_plan_day": "Visits per plan day", "recency_days": "Days since last visit",
           "addon_count": "Add-ons used", "lifetime_value": "Lifetime value (EUR)",
           "total_visits": "Total visits", "n_renewals": "Renewals", "age": "Age",
           "meeting_room_hours": "Meeting-room hours", "n_colleagues_on_account": "Colleagues on account"}
SPLITS = {"plan": "Plan", "member_state": "Member state", "value_band": "Value band", "team": "Account type",
          "gender_l": "Gender", "country_l": "Country", "industry_l": "Industry"}
SKEWED = {"monthly_value", "lifetime_value", "total_visits", "meeting_room_hours", "recency_days", "tenure_months"}

# %% Load data
def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read the parquet file and add display labels used by the charts."""
    df = pd.read_parquet(path)
    df["plan"] = pd.Categorical(df["plan_tier"].map(PLAN_LABEL), categories=list(PLAN_LABEL.values()), ordered=True)
    df["member_state"] = pd.Categorical(df["member_state"], categories=STATE_ORDER, ordered=True)
    df["industry_l"] = df["industry"].map(INDUSTRY_LABEL).fillna("Not given")
    df["country_l"] = df["country"].str.title().fillna("Not given")
    df["gender_l"] = df["gender"].str.capitalize().fillna("Not given")
    df["team"] = np.where(df["is_team_account"], "Team account", "Individual")
    df["value_band"] = df["value_band"].fillna("No value data")
    return df


DF = load_data()
COHORTS = sorted(DF["start_cohort"].dropna().unique())
N_ALL = len(DF)

# %% Formatting and chart helpers
def eur(x) -> str:
    return "n/a" if pd.isna(x) else f"€{x:,.0f}"


def pct(x, digits=0) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def base_layout(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    """Shared, quiet chart styling: light grid, no chart junk, legend on top."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, size=12)),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text="", font_size=11,
                                        traceorder="normal"),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title_font_size=11)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title_font_size=11)
    return fig


def two_line(labels) -> list:
    """'Dedicated desk' -> 'Dedicated<br>desk', so heatmap columns stay narrow and upright."""
    return [str(x).replace(" ", "<br>", 1) for x in labels]


def empty_fig(message: str = "No members match the current filters.", height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base_layout(fig, height, legend=False)


def card(title: str, subtitle: str, body, wide: bool = False) -> dbc.Col:
    """A chart card: a title that states the question, a short note on how to read it, then the chart."""
    return dbc.Col(html.Div([
        html.H3(title, className="card-title"),
        html.P(subtitle, className="card-sub"),
        body,
    ], className="panel"), lg=12 if wide else 6, className="mb-4")


TABLE_STYLE = dict(
    style_as_list_view=True,
    style_header={"fontWeight": 600, "backgroundColor": SURFACE, "color": INK, "fontFamily": FONT,
                  "borderBottom": f"1px solid {AXIS}"},
    style_cell={"fontFamily": FONT, "fontSize": 13, "padding": "7px 10px", "color": INK_2, "backgroundColor": SURFACE,
                "border": "none", "borderBottom": f"1px solid {GRID}", "fontVariantNumeric": "tabular-nums"},
)


def graph(id_: str) -> dcc.Loading:
    return dcc.Loading(dcc.Graph(id=id_, config={"displaylogo": False, "displayModeBar": "hover"}),
                       type="dot", color=ACCENT)


def insight(id_: str) -> html.Div:
    return html.Div(id=id_, className="insight")


# %% Filtering
FILTER_IDS = ["f-plan", "f-state", "f-country", "f-industry", "f-cohort", "f-team", "f-clean"]
FILTERS = [Input(i, "value") for i in FILTER_IDS]


def filter_frame(plans, states, countries, industries, cohort_range, team, clean, keep_flagged=False):
    """Apply the sidebar filters. keep_flagged ignores the 'exclude flagged rows' switch (data-quality tab)."""
    d = DF
    if plans:
        d = d[d["plan"].isin(plans)]
    if states:
        d = d[d["member_state"].isin(states)]
    if countries:
        d = d[d["country_l"].isin(countries)]
    if industries:
        d = d[d["industry_l"].isin(industries)]
    if cohort_range:
        lo, hi = COHORTS[cohort_range[0]], COHORTS[cohort_range[1]]
        d = d[d["start_cohort"].between(lo, hi)]
    if team in ("Individual", "Team account"):
        d = d[d["team"] == team]
    if clean and not keep_flagged:
        d = d[d["quality_issues"] == 0]
    return d


# %% Layout: sidebar
def sidebar() -> html.Div:
    plans = list(PLAN_LABEL.values())
    countries = sorted(DF["country_l"].unique(), key=lambda c: (c == "Not given", c))
    industries = sorted(DF["industry_l"].unique(), key=lambda c: (c in ("Other", "Not given"), c))
    years = {i: c[:4] for i, c in enumerate(COHORTS) if c.endswith("Q1")}
    return html.Div([
        html.Div([html.Span("ACDC", className="brand-mark"), html.Span("Member explorer", className="brand-name")],
                 className="brand"),
        html.P("Filters apply to every tab.", className="side-note"),
        html.Label("Plan", className="side-label"),
        dbc.Checklist(id="f-plan", options=plans, value=[], className="side-check"),
        html.Label("Member state", className="side-label"),
        dbc.Checklist(id="f-state", options=STATE_ORDER, value=[], className="side-check"),
        html.Label("Start cohort", className="side-label"),
        html.Div(dcc.RangeSlider(id="f-cohort", min=0, max=len(COHORTS) - 1, step=1,
                                 value=[0, len(COHORTS) - 1], marks=years, allowCross=False,
                                 allow_direct_input=False), className="side-slider"),
        html.Div(id="cohort-text", className="side-hint"),
        html.Label("Country", className="side-label"),
        dcc.Dropdown(id="f-country", options=countries, multi=True, placeholder="All countries"),
        html.Label("Industry", className="side-label"),
        dcc.Dropdown(id="f-industry", options=industries, multi=True, placeholder="All industries"),
        html.Label("Account type", className="side-label"),
        dbc.RadioItems(id="f-team", options=["All", "Individual", "Team account"], value="All",
                       className="side-check"),
        html.Hr(className="side-rule"),
        dbc.Switch(id="f-clean", label="Exclude rows flagged by cleaning", value=False, className="side-switch"),
        dbc.Button("Reset filters", id="reset", color="link", className="side-reset"),
        html.Div([
            html.P("Data: ACDC members 2021-2023, extracted 1 Jan 2024."),
            html.P("Cleaning and features: notebook 03."),
        ], className="side-foot"),
    ], className="sidebar")


# %% Layout: tabs
def kpi(id_: str, label: str) -> dbc.Col:
    return dbc.Col(html.Div([html.Div(label, className="kpi-label"), html.Div(id=id_, className="kpi-value"),
                             html.Div(id=f"{id_}-sub", className="kpi-sub")], className="kpi"), xs=6, lg=True)


tab_overview = [
    insight("ov-insight"),
    dbc.Row([kpi("k-members", "Members"), kpi("k-active", "Still active"), kpi("k-value", "Median monthly value"),
             kpi("k-util", "Median utilization"), kpi("k-dormant", "Dormant among active")], className="g-3 mb-4"),
    dbc.Row([
        card("Where members are in their lifecycle", "Share of each plan by member state. Click a bar segment "
             "to list those members in the Explorer tab.", graph("ov-state")),
        card("Monthly value rises with plan size", "Monthly value = lifetime value / months of membership. "
             "Box shows the middle 50%, line shows the median. Log scale.", graph("ov-value")),
        card("New members per quarter", "Members by quarter of joining, stacked by plan. The intake fell "
             "through 2022-2023 and recovered in the last quarter.", graph("ov-intake"), wide=True),
    ]),
]

tab_distributions = [
    dbc.Row([
        dbc.Col([html.Label("Variable", className="ctl-label"),
                 dcc.Dropdown(id="d-var", options=[{"label": v, "value": k} for k, v in NUMERIC.items()],
                              value="monthly_value", clearable=False)], md=4),
        dbc.Col([html.Label("Compare by", className="ctl-label"),
                 dcc.Dropdown(id="d-split", options=[{"label": v, "value": k} for k, v in SPLITS.items()],
                              value="plan", clearable=False)], md=4),
        dbc.Col([html.Label("Scale", className="ctl-label"),
                 dbc.RadioItems(id="d-log", options=[{"label": "Linear", "value": False},
                                                     {"label": "Log", "value": True}],
                                value=True, inline=True, className="pt-1")], md=4),
    ], className="controls mb-4"),
    insight("d-insight"),
    dbc.Row([
        card("Distribution", "Histogram of all selected members. Missing values are left out and counted below.",
             graph("d-hist")),
        card("Compared across groups", "Box shows the middle 50% and the median. Groups are sorted by median.",
             graph("d-box")),
        card("Summary by group", "Missing values are counted per group. Values come from the cleaned data "
             "and are not imputed.", html.Div(id="d-table"), wide=True),
    ]),
]

tab_value = [
    insight("v-insight"),
    dbc.Row([
        card("Who pays, and who comes in", "Each dot is a member: utilization of entitled visits (x) against "
             "monthly value (y, log scale). The cross marks each plan's median.", graph("v-scatter"), wide=True),
        card("Plan use, by plan", "Median share of entitled visits used in the last period. "
             "The dotted line marks full use.", graph("v-util")),
        card("Value band against recency", "Members in each value band by days since their last visit. "
             "Click a cell to list those members.", graph("v-heat")),
    ]),
]

tab_lifecycle = [
    insight("l-insight"),
    dbc.Row([
        card("How long members stay", "Kaplan-Meier estimate of the share still a member after each month. "
             "Active members count as censored, so recent joiners do not bias the curve.", graph("l-km"), wide=True),
        card("Still active, by start cohort and plan", "Share of each cohort that is still active on 1 Jan 2024. Cells "
             "with fewer than 20 members are blank. Click a cell to list those members.", graph("l-cohort")),
        card("Plan mix of each cohort", "Share of each start cohort by plan. Larger plans almost disappeared "
             "from the 2023 intake and came back in 2023 Q4.", graph("l-mix")),
    ]),
]

tab_addons = [
    insight("a-insight"),
    dbc.Row([
        card("Add-on use by plan", "Share of members on each plan who have used the add-on. "
             "Click a cell to list those members.", graph("a-heat")),
        card("Which add-ons go together", "Lift = how much more often two add-ons are used together than "
             "chance predicts. Berry: used together more often; teal: less often.", graph("a-lift")),
        card("Value by number of add-ons", "Median monthly value by the number of add-ons a member uses. "
             "Bar labels show how many members are in each group.", graph("a-count"), wide=True),
    ]),
]

tab_quality = [
    insight("q-insight"),
    dbc.Row([
        card("Rows flagged by each cleaning rule", "Counted on the selected members, before the 'exclude' switch. "
             "One row can break more than one rule.", graph("q-flags")),
        card("Missing values after cleaning", "Share of selected members with no value. These gaps are "
             "kept for the imputation step in Deliverable 2.", graph("q-missing")),
        card("Cleaning rules", "Agreed in the group and applied in notebook 03.", dcc.Markdown("""
| Issue | Rule |
|---|---|
| Duplicate rows | Dropped (15 rows) |
| Age outside 16-100 | Missing |
| Plan days 0 or 9 | Standard days of the plan |
| Entitled visits = 0 | Plan median |
| Days since last visit = 999 | Missing (code) |
| Colleagues = -1 | 0 if individual, missing if team |
| Negative add-on hours | 0 if add-on unused, else missing |
| Negative visits, renewals, value | Missing |
| Start after end | End rebuilt from visits and weekly rate |
| Swipe outside membership | Rebuilt from reference date and days since visit |
| Last-period visits above total | Capped at total visits |
| Monthly value above 10x plan median | Flagged only |
""", className="rules"), wide=True),
    ]),
]

HEADERS = {"member_id": "Member", "plan": "Plan", "member_state": "State", "start_cohort": "Cohort",
           "tenure_months": "Tenure (months)", "monthly_value": "Monthly value (€)", "utilization_rate": "Utilization",
           "recency_days": "Days since visit", "addon_count": "Add-ons", "country_l": "Country",
           "industry_l": "Industry", "team": "Account", "quality_issues": "Cleaning flags"}
COLUMNS = list(HEADERS)

tab_explorer = [
    html.Div([
        html.Div(id="drill-text", className="drill-text"),
        dbc.Button("Clear selection", id="drill-clear", color="link", size="sm", className="drill-clear"),
        dbc.Button("Download CSV", id="download-btn", color="dark", size="sm", outline=True, className="ms-auto"),
        dcc.Download(id="download"),
    ], className="drill-bar"),
    dash_table.DataTable(
        id="table", columns=[
            {"name": HEADERS[c], "id": c, "type": "numeric",
             "format": {"specifier": ".0%"} if c == "utilization_rate" else {"specifier": ",.1~f"}}
            if c in NUMERIC or c == "quality_issues" else {"name": HEADERS[c], "id": c} for c in COLUMNS],
        page_size=15, sort_action="native", filter_action="native", style_table={"overflowX": "auto"},
        **TABLE_STYLE,
        style_filter={"backgroundColor": "#f4f3ef"},
    ),
]

TABS = [("overview", "Overview", tab_overview), ("dist", "Distributions", tab_distributions),
        ("value", "Value and usage", tab_value), ("life", "Lifecycle", tab_lifecycle),
        ("addons", "Add-ons", tab_addons), ("quality", "Data quality", tab_quality),
        ("explorer", "Member explorer", tab_explorer)]

# %% App and page layout
app = Dash(__name__, title="ACDC member explorer", assets_folder=str(HERE / "assets"),
           suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Store(id="drill", data={}),
    sidebar(),
    html.Main([
        html.Header([
            html.Div([html.H1("Who are ACDC's members, and how do they use the space?"),
                      html.P(id="scope-text", className="scope")]),
        ], className="page-head"),
        dbc.Tabs([dbc.Tab(children, label=label, tab_id=tid) for tid, label, children in TABS],
                 id="tabs", active_tab="overview", className="tabs"),
    ], className="main"),
], className="shell")

# %% Callbacks: filters and scope line
@app.callback(Output("f-plan", "value"), Output("f-state", "value"), Output("f-country", "value"),
              Output("f-industry", "value"), Output("f-cohort", "value"), Output("f-team", "value"),
              Output("f-clean", "value"), Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [0, len(COHORTS) - 1], "All", False


@app.callback(Output("scope-text", "children"), Output("cohort-text", "children"), *FILTERS)
def scope(*f):
    d = filter_frame(*f)
    lo, hi = f[4]
    return f"Showing {len(d):,} of {N_ALL:,} members ({len(d) / N_ALL:.0%}).", f"{COHORTS[lo]} to {COHORTS[hi]}"


# %% Callbacks: overview
@app.callback(Output("ov-insight", "children"),
              Output("k-members", "children"), Output("k-members-sub", "children"),
              Output("k-active", "children"), Output("k-active-sub", "children"),
              Output("k-value", "children"), Output("k-value-sub", "children"),
              Output("k-util", "children"), Output("k-util-sub", "children"),
              Output("k-dormant", "children"), Output("k-dormant-sub", "children"),
              Output("ov-state", "figure"), Output("ov-value", "figure"), Output("ov-intake", "figure"), *FILTERS)
def overview(*f):
    d = filter_frame(*f)
    if d.empty:
        return ("", "0", "", "-", "", "-", "", "-", "", "-", "", empty_fig(), empty_fig(), empty_fig(height=300))
    active = d[d["is_active"]]
    dormant_share = (active["member_state"] == "Dormant").mean() if len(active) else np.nan

    # Lifecycle mix by plan (100% bars), clickable.
    mix = pd.crosstab(d["plan"], d["member_state"], normalize="index").reindex(columns=STATE_ORDER, fill_value=0)
    counts = pd.crosstab(d["plan"], d["member_state"]).reindex(columns=STATE_ORDER, fill_value=0)
    fig_state = go.Figure()
    for s in STATE_ORDER:
        fig_state.add_bar(y=mix.index.astype(str), x=mix[s], name=s, orientation="h", marker_color=STATE_COLOR[s],
                          marker_line=dict(color=SURFACE, width=2), customdata=np.c_[mix.index.astype(str),
                          [s] * len(mix), counts[s]], hovertemplate="%{customdata[0]} · %{customdata[1]}<br>"
                          "%{x:.0%} of the plan (%{customdata[2]:,} members)<extra></extra>",
                          text=[f"{v:.0%}" if v >= 0.08 else "" for v in mix[s]], textposition="inside",
                          insidetextanchor="middle", textfont=dict(color=INK if s in ("Active", "Churned") else "white", size=11))
    fig_state.update_layout(barmode="stack")
    fig_state.update_xaxes(tickformat=".0%", range=[0, 1])
    fig_state.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")

    # Monthly value by plan.
    fig_value = go.Figure()
    for p in d["plan"].cat.categories:
        v = d.loc[(d["plan"] == p) & (d["monthly_value"] > 0), "monthly_value"]
        if len(v):
            fig_value.add_box(y=v, name=p, marker_color=PLAN_COLOR[p], boxpoints=False, line_width=1.5,
                              hovertemplate=f"{p}<br>%{{y:,.0f}} €/month<extra></extra>")
    fig_value.update_yaxes(type="log", title="€ per month", tickprefix="€")

    # Intake by quarter.
    intake = pd.crosstab(d["start_cohort"], d["plan"]).reindex(columns=d["plan"].cat.categories, fill_value=0)
    fig_intake = go.Figure()
    for p in intake.columns:
        fig_intake.add_bar(x=intake.index, y=intake[p], name=p, marker_color=PLAN_COLOR[p],
                           marker_line=dict(color=SURFACE, width=1),
                           hovertemplate=f"{p}<br>%{{x}}: %{{y:,}} new members<extra></extra>")
    fig_intake.update_layout(barmode="stack")
    fig_intake.update_yaxes(title="New members")

    # One-sentence reading, recomputed for the current filters.
    by_plan = d.groupby("plan", observed=True).agg(value=("monthly_value", "median"),
                                                   util=("utilization_rate", "median"))
    text = ""
    if len(by_plan) >= 2:
        top = by_plan["value"].idxmax()
        low_util = by_plan["util"].idxmin()
        text = f"{top} members bring the most value per month ({eur(by_plan.loc[top, 'value'])})"
        if low_util == top:
            text += f" yet use the smallest share of the visits they pay for ({pct(by_plan.loc[top, 'util'])})."
        else:
            text += (f". {low_util} members use the smallest share of the visits they pay for "
                     f"({pct(by_plan.loc[low_util, 'util'])}).")
    return (text,
            f"{len(d):,}", f"{len(d) / N_ALL:.0%} of all members",
            f"{len(active):,}", f"{len(active) / len(d):.0%} of selected",
            eur(d["monthly_value"].median()), "per member and month",
            pct(d["utilization_rate"].median()), "of entitled visits, last period",
            pct(dormant_share), f"no visit for {60}+ days",
            base_layout(fig_state, 300), base_layout(fig_value, 300, legend=False), base_layout(fig_intake, 300))


# %% Callbacks: distributions
@app.callback(Output("d-insight", "children"), Output("d-hist", "figure"), Output("d-box", "figure"),
              Output("d-table", "children"), Input("d-var", "value"), Input("d-split", "value"),
              Input("d-log", "value"), *FILTERS)
def distributions(var, split, log, *f):
    d = filter_frame(*f)
    v = d[var].dropna()
    if v.empty:
        return "", empty_fig(), empty_fig(), ""
    use_log = bool(log) and var in SKEWED
    label = NUMERIC[var]
    plot_v = np.log10(v[v > 0]) if use_log else v

    fig_hist = go.Figure(go.Histogram(x=plot_v, nbinsx=50, marker_color=ACCENT,
                                      marker_line=dict(color=SURFACE, width=1),
                                      hovertemplate="%{y:,} members<extra></extra>"))
    if use_log:  # show real values on a log axis
        ticks = [t for t in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000] if plot_v.min() - 0.3 <= np.log10(t) <= plot_v.max() + 0.3]
        fig_hist.update_xaxes(tickvals=np.log10(ticks), ticktext=[f"{t:,}" for t in ticks])
    fig_hist.update_xaxes(title=label + (" · log scale, zeros left out" if use_log else ""))
    fig_hist.update_yaxes(title="Members")

    groups = d.groupby(split, observed=True)[var]
    order = groups.median().sort_values(ascending=False).index[:12]
    fig_box = go.Figure()
    for g in order:
        gv = d.loc[d[split] == g, var].dropna()
        if use_log:
            gv = gv[gv > 0]
        color = PLAN_COLOR.get(str(g), STATE_COLOR.get(str(g), ACCENT))
        fig_box.add_box(x=gv, name=str(g), orientation="h", marker_color=color, boxpoints=False, line_width=1.5)
    fig_box.update_xaxes(type="log" if use_log else "linear", title=label)
    fig_box.update_yaxes(autorange="reversed")

    summary = d.groupby(split, observed=True)[var].agg(
        members="size", missing=lambda s: s.isna().sum(), median="median",
        p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), mean="mean").reindex(order).reset_index()
    summary.columns = [SPLITS[split], "Members", "Missing", "Median", "25th pct", "75th pct", "Mean"]
    table = dash_table.DataTable(
        data=summary.to_dict("records"), columns=[{"name": c, "id": c, "type": "numeric",
        "format": {"specifier": ",.2f"} if c in ("Median", "25th pct", "75th pct", "Mean") else {"specifier": ","}}
        if i else {"name": c, "id": c} for i, c in enumerate(summary.columns)],
        **TABLE_STYLE)

    skew = v.skew()
    shape = "strongly right-skewed" if skew > 2 else "right-skewed" if skew > 0.5 else \
        "left-skewed" if skew < -0.5 else "roughly symmetric"
    text = (f"{label}: median {v.median():,.2f}, middle half between {v.quantile(.25):,.2f} and "
            f"{v.quantile(.75):,.2f}. The distribution is {shape} (skewness {skew:.1f}); "
            f"{d[var].isna().mean():.1%} of the selected members have no value.")
    return text, base_layout(fig_hist, 320, legend=False), base_layout(fig_box, 320, legend=False), table


# %% Callbacks: value and usage
@app.callback(Output("v-insight", "children"), Output("v-scatter", "figure"), Output("v-util", "figure"),
              Output("v-heat", "figure"), *FILTERS)
def value_usage(*f):
    d = filter_frame(*f)
    d = d[d["plan"].notna()]
    if d.empty:
        return "", empty_fig(height=360), empty_fig(), empty_fig()
    plans = [p for p in PLAN_LABEL.values() if p in set(d["plan"])]

    # Small multiples: one panel per plan, same axes, so no colour has to carry the comparison alone.
    fig_sc = make_subplots(rows=1, cols=len(plans), shared_yaxes=True, subplot_titles=plans, horizontal_spacing=0.02)
    for i, p in enumerate(plans, start=1):
        s = d[(d["plan"] == p) & (d["monthly_value"] > 0)]
        fig_sc.add_scattergl(x=s["utilization_rate"].clip(upper=1.25), y=s["monthly_value"], mode="markers",
                             marker=dict(color=PLAN_COLOR[p], size=4, opacity=0.35), name=p, row=1, col=i,
                             customdata=s["member_id"], hovertemplate="%{customdata}<br>Utilization %{x:.0%}"
                             "<br>%{y:,.0f} €/month<extra></extra>")
        fig_sc.add_scatter(x=[s["utilization_rate"].median()], y=[s["monthly_value"].median()], mode="markers",
                           marker=dict(symbol="cross-thin", size=18, line=dict(width=2.5, color=INK)),
                           hovertemplate=f"{p} median<br>Utilization %{{x:.0%}}<br>%{{y:,.0f}} €/month<extra></extra>",
                           row=1, col=i)
    fig_sc.update_yaxes(type="log", tickprefix="€")
    fig_sc.update_yaxes(title_text="€ per month", row=1, col=1)
    fig_sc.update_xaxes(tickformat=".0%", range=[0, 1.25], tickvals=[0, 0.5, 1])
    fig_sc.update_annotations(font=dict(size=12, color=INK))
    base_layout(fig_sc, 380, legend=False)
    fig_sc.update_layout(margin=dict(t=28))

    util = d.groupby("plan", observed=True)["utilization_rate"].median().reindex(plans)
    fig_u = go.Figure(go.Bar(x=util.values, y=util.index, orientation="h", marker_color=[PLAN_COLOR[p] for p in plans],
                             text=[pct(u) for u in util.values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:.0%} of entitled visits<extra></extra>"))
    fig_u.add_vline(x=1, line=dict(color=MUTED, dash="dot", width=1))
    fig_u.update_xaxes(tickformat=".0%", range=[0, 1.1])
    fig_u.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")

    bands = ["High", "Mid", "Low"]
    rec = ["0-7 days", "8-30 days", "31-60 days", "61-180 days", "180+ days"]
    heat = pd.crosstab(d["value_band"], d["recency_band"]).reindex(index=bands, columns=rec, fill_value=0)
    share = heat.div(heat.sum(axis=1).replace(0, np.nan), axis=0)
    fig_h = go.Figure(go.Heatmap(z=share.values, x=rec, y=bands, colorscale=ROCKET, zmin=0, xgap=2, ygap=2,
                                 customdata=heat.values, text=[[pct(v) for v in row] for row in share.values],
                                 texttemplate="%{text}", hovertemplate="%{y} value · %{x}<br>%{customdata:,} members "
                                 "(%{z:.0%} of the band)<extra></extra>", colorbar=dict(tickformat=".0%", thickness=10)))
    fig_h.update_xaxes(showgrid=False, title="Days since last visit")
    fig_h.update_yaxes(showgrid=False, title="Value band")

    text = ""
    if "Private office" in plans:
        po = d[d["plan"] == "Private office"]
        text = (f"Private-office members pay a median {eur(po['monthly_value'].median())} per month but use only "
                f"{pct(po['utilization_rate'].median())} of their entitled visits: they pay for space they rarely use.")
    return text, fig_sc, base_layout(fig_u, 300, legend=False), base_layout(fig_h, 300, legend=False)


# %% Callbacks: lifecycle
@app.callback(Output("l-insight", "children"), Output("l-km", "figure"), Output("l-cohort", "figure"),
              Output("l-mix", "figure"), *FILTERS)
def lifecycle(*f):
    d = filter_frame(*f)
    d = d[d["plan"].notna()]
    if d.empty:
        return "", empty_fig(height=360), empty_fig(), empty_fig()
    plans = [p for p in PLAN_LABEL.values() if p in set(d["plan"])]

    fig_km = go.Figure()
    medians = {}
    for p in plans:
        s = d[d["plan"] == p]
        km = KaplanMeierFitter().fit(s["tenure_months"], event_observed=~s["is_active"])
        sf = km.survival_function_.iloc[:, 0]
        ci = km.confidence_interval_survival_function_
        medians[p] = km.median_survival_time_
        fig_km.add_scatter(x=list(ci.index) + list(ci.index[::-1]),
                           y=list(ci.iloc[:, 1]) + list(ci.iloc[::-1, 0]), fill="toself", fillcolor=PLAN_COLOR[p],
                           opacity=0.12, line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=p)
        med = "not reached" if np.isinf(medians[p]) else f"{medians[p]:.0f} months"
        fig_km.add_scatter(x=sf.index, y=sf.values, mode="lines", line=dict(color=PLAN_COLOR[p], width=2, shape="hv"),
                           name=f"{p} (median {med})", legendgroup=p,
                           hovertemplate=f"{p}<br>After %{{x:.0f}} months: %{{y:.0%}} still members<extra></extra>")
    fig_km.add_hline(y=0.5, line=dict(color=MUTED, dash="dot", width=1))
    fig_km.update_xaxes(title="Months since joining", range=[0, 36])
    fig_km.update_yaxes(title="Still a member", tickformat=".0%", range=[0, 1.02])

    cohort = d.pivot_table(index="start_cohort", columns="plan", values="is_active", aggfunc="mean",
                           observed=True).reindex(columns=plans)
    sizes = d.pivot_table(index="start_cohort", columns="plan", values="member_id", aggfunc="count",
                          observed=True).reindex(index=cohort.index, columns=plans).fillna(0)
    cohort = cohort.where(sizes >= 20)  # shares from fewer than 20 members are not shown
    fig_c = go.Figure(go.Heatmap(z=cohort.values, x=plans, y=cohort.index, colorscale=ROCKET, zmin=0, zmax=1,
                                 xgap=2, ygap=2, customdata=sizes.values,
                                 text=[[pct(v) if pd.notna(v) else "" for v in row] for row in cohort.values],
                                 texttemplate="%{text}", textfont_size=10,
                                 hovertemplate="%{y} · %{x}<br>%{z:.0%} still active of %{customdata:,} members"
                                 "<extra></extra>", colorbar=dict(tickformat=".0%", thickness=10)))
    fig_c.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig_c.update_xaxes(showgrid=False, side="top", tickvals=plans, ticktext=two_line(plans))

    mix = pd.crosstab(d["start_cohort"], d["plan"], normalize="index").reindex(columns=plans, fill_value=0)
    fig_m = go.Figure()
    for p in plans:
        fig_m.add_bar(x=mix.index, y=mix[p], name=p, marker_color=PLAN_COLOR[p], marker_line=dict(color=SURFACE, width=1),
                      hovertemplate=f"{p}<br>%{{x}}: %{{y:.0%}} of the cohort<extra></extra>")
    fig_m.update_layout(barmode="stack")
    fig_m.update_yaxes(tickformat=".0%", range=[0, 1])

    finite = {p: m for p, m in medians.items() if np.isfinite(m)}
    text = ""
    if finite:
        short = min(finite, key=finite.get)
        text = f"Half of {short.lower()} members leave within {finite[short]:.0f} months. "
        long = [p for p, m in medians.items() if not np.isfinite(m) or m > 2 * finite[short]]
        if long:
            text += "Larger plans stay much longer: " + ", ".join(
                f"{p.lower()} {'median not reached' if not np.isfinite(medians[p]) else f'{medians[p]:.0f} months'}"
                for p in long) + "."
    base_layout(fig_km, 400)
    fig_km.update_layout(margin=dict(t=56), legend=dict(y=1.04, yanchor="bottom"))  # legend sits above the 100% line
    return text, fig_km, base_layout(fig_c, 380, legend=False), base_layout(fig_m, 380)


# %% Callbacks: add-ons
@app.callback(Output("a-insight", "children"), Output("a-heat", "figure"), Output("a-lift", "figure"),
              Output("a-count", "figure"), *FILTERS)
def addons(*f):
    d = filter_frame(*f)
    d = d[d["plan"].notna()]
    if d.empty:
        return "", empty_fig(), empty_fig(), empty_fig(height=280)
    plans = [p for p in PLAN_LABEL.values() if p in set(d["plan"])]
    cols, names = list(ADDONS), list(ADDONS.values())

    rate = d.groupby("plan", observed=True)[cols].mean().reindex(plans).T
    rate.index = names
    fig_h = go.Figure(go.Heatmap(z=rate.values, x=plans, y=names, colorscale=ROCKET, zmin=0, zmax=max(0.6, rate.values.max()),
                                 xgap=2, ygap=2, text=[[pct(v) for v in row] for row in rate.values],
                                 texttemplate="%{text}", textfont_size=11,
                                 hovertemplate="%{x} · %{y}<br>%{z:.0%} have used it<extra></extra>",
                                 colorbar=dict(tickformat=".0%", thickness=10)))
    fig_h.update_xaxes(showgrid=False, side="top", tickvals=plans, ticktext=two_line(plans), tickangle=0)
    fig_h.update_yaxes(showgrid=False, autorange="reversed")

    x = d[cols].astype(float).values
    p_single = x.mean(axis=0)
    together = (x.T @ x) / len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = together / np.outer(p_single, p_single)
    log_lift = np.log2(lift)
    np.fill_diagonal(log_lift, np.nan)
    fig_l = go.Figure(go.Heatmap(z=log_lift, x=names, y=names, colorscale=DIVERGING, zmid=0, zmin=-1.5, zmax=1.5,
                                 xgap=2, ygap=2, customdata=lift,
                                 hovertemplate="%{y} + %{x}<br>Used together %{customdata:.2f}x as often as chance"
                                 "<extra></extra>", colorbar=dict(thickness=10, tickvals=[-1, 0, 1],
                                                                 ticktext=["½x", "1x", "2x"])))
    fig_l.update_xaxes(showgrid=False, tickangle=-40)
    fig_l.update_yaxes(showgrid=False, autorange="reversed")

    by_n = d.groupby("addon_count").agg(value=("monthly_value", "median"), n=("member_id", "size"))
    by_n = by_n[by_n["n"] >= 20]  # very small groups are hidden
    fig_n = go.Figure(go.Bar(x=by_n.index, y=by_n["value"], marker_color=ACCENT,
                             text=[f"n={n:,}" for n in by_n["n"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x} add-ons<br>Median %{y:,.0f} €/month<extra></extra>"))
    fig_n.update_xaxes(title="Add-ons used", dtick=1, showgrid=False)
    fig_n.update_yaxes(title="Median € per month", tickprefix="€")

    upper = np.where(np.triu(np.ones_like(lift, dtype=bool), 1) & np.isfinite(lift), lift, np.nan)
    text = ""
    if np.isfinite(upper).any():  # tiny selections can have no add-on pairs at all
        i, j = np.unravel_index(np.nanargmax(upper), upper.shape)
        top_plan = rate.loc["Mail handling"].idxmax()
        text = (f"{names[i]} and {names[j]} are used together {upper[i, j]:.1f} times as often as chance would "
                f"predict. Mail handling is mostly a resident service: {pct(rate.loc['Mail handling', top_plan])} "
                f"of {top_plan.lower()} members use it.")
    return (text, base_layout(fig_h, 360, legend=False), base_layout(fig_l, 360, legend=False),
            base_layout(fig_n, 280, legend=False))


# %% Callbacks: data quality
@app.callback(Output("q-insight", "children"), Output("q-flags", "figure"), Output("q-missing", "figure"), *FILTERS)
def quality(*f):
    d = filter_frame(*f, keep_flagged=True)
    if d.empty:
        return "", empty_fig(), empty_fig()
    flags = d[list(FLAGS)].sum().rename(FLAGS).sort_values()
    fig_f = go.Figure(go.Bar(x=flags.values, y=flags.index, orientation="h", marker_color=ACCENT,
                             text=[f"{v:,}" for v in flags.values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}<br>%{x:,} rows<extra></extra>"))
    fig_f.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig_f.update_xaxes(title="Rows", range=[0, flags.max() * 1.15])

    shown = ["age", "gender", "industry", "country", "plan_tier", "n_colleagues_on_account", "membership_end",
             "last_badge_swipe", "days_since_last_visit", "total_visits", "lifetime_value", "n_renewals",
             "meeting_room_hours", "guest_passes_issued"]
    miss = d[shown].isna().mean()
    miss.loc["membership_end"] = d.loc[~d["is_active"], "membership_end"].isna().mean()  # active members have no end date by design
    miss = miss[miss > 0].sort_values()
    fig_m = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color="#b9b1a6",
                             text=[pct(v, 1) for v in miss.values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:.1%} missing<extra></extra>"))
    fig_m.update_xaxes(tickformat=".0%", title="Share of members", range=[0, miss.max() * 1.18])
    fig_m.update_yaxes(gridcolor="rgba(0,0,0,0)")

    flagged = (d["quality_issues"] > 0).mean()
    text = (f"{flagged:.1%} of the selected members have at least one cleaning flag. Turn on "
            f"'Exclude rows flagged by cleaning' to check that a finding does not depend on them.")
    return text, base_layout(fig_f, 360, legend=False), base_layout(fig_m, 360, legend=False)


# %% Callbacks: drill-down and member explorer
@app.callback(Output("drill", "data"), Output("tabs", "active_tab"),
              Input("ov-state", "clickData"), Input("v-heat", "clickData"), Input("l-cohort", "clickData"),
              Input("a-heat", "clickData"), Input("drill-clear", "n_clicks"), prevent_initial_call=True)
def drill(state_click, value_click, cohort_click, addon_click, _):
    trigger = ctx.triggered_id
    if trigger == "drill-clear":
        return {}, no_update
    point = (ctx.triggered[0]["value"] or {}).get("points", [{}])[0]
    if trigger == "ov-state":
        sel = {"plan": point["customdata"][0], "member_state": point["customdata"][1]}
    elif trigger == "v-heat":
        sel = {"value_band": point["y"], "recency_band": point["x"]}
    elif trigger == "l-cohort":
        sel = {"start_cohort": point["y"], "plan": point["x"]}
    else:
        addon = {v: k for k, v in ADDONS.items()}[point["y"]]
        sel = {"plan": point["x"], addon: True}
    return sel, "explorer"


def apply_drill(d: pd.DataFrame, sel: dict) -> pd.DataFrame:
    for col, val in (sel or {}).items():
        d = d[d[col].astype(object) == val] if not isinstance(val, bool) else d[d[col] == val]
    return d


@app.callback(Output("table", "data"), Output("drill-text", "children"), Input("drill", "data"), *FILTERS)
def explorer(sel, *f):
    d = apply_drill(filter_frame(*f), sel)
    if sel:
        names = {"plan": "Plan", "member_state": "State", "value_band": "Value band",
                 "recency_band": "Last visit", "start_cohort": "Cohort"}
        parts = [f"{ADDONS[k]} used" if k in ADDONS else f"{names[k]}: {v}" for k, v in sel.items()]
        label = "Selection: " + " · ".join(parts) + f" · {len(d):,} members"
    else:
        label = f"All filtered members · {len(d):,}. Click a bar or heatmap cell on another tab to narrow the list."
    out = d[COLUMNS].copy()
    out["plan"], out["member_state"] = out["plan"].astype(str), out["member_state"].astype(str)
    return out.to_dict("records"), label


@app.callback(Output("download", "data"), Input("download-btn", "n_clicks"), State("drill", "data"), *[
    State(i, "value") for i in FILTER_IDS], prevent_initial_call=True)
def download(_, sel, *f):
    d = apply_drill(filter_frame(*f), sel)
    return dcc.send_data_frame(d[COLUMNS].rename(columns=HEADERS).to_csv, "acdc_members_selection.csv", index=False)


# %% Run
if __name__ == "__main__":
    app.run(debug=False)
