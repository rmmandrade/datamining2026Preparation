"""Rebuild notebook/04_dash_app.ipynb from dashboard/app.py: python tools/build_dash_notebook.py (then re-run the notebook)."""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")

# Split the script on its "# %% <title>" markers; the module docstring is left out.
parts = re.split(r"^# %% (.+)$", src, flags=re.M)
blocks = [(parts[i].strip(), parts[i + 1].strip("\n")) for i in range(1, len(parts), 2)]

NOTES = {
    "Imports and paths": "The app reads the parquet file written by notebook 03. Paths are relative to the repository, so the code runs on any machine.",
    "Labels, order and colours": "Each plan keeps one colour on every chart. The five plan colours come from a colour-blind-checked categorical palette, and member states use a separate set so the two are never confused.",
    "Load data": "Display labels are added once at load time, so the callbacks only filter and aggregate.",
    "Formatting and chart helpers": "One shared chart style: light grid, no 3D, no dual axes, legends above the plot. Every chart sits in a card whose title states the question it answers.",
    "Filtering": "All sidebar filters go through one function, so every tab shows the same selection.",
    "Layout: sidebar": None,
    "Layout: tabs": "Seven tabs. Each one opens with a sentence that is recomputed for the current filters, followed by the charts that support it.",
    "App and page layout": "`suppress_callback_exceptions` is needed because some components live in tabs that are not visible at start-up.",
    "Callbacks: filters and scope line": None,
    "Callbacks: overview": None,
    "Callbacks: distributions": "The distribution tab covers the guideline requirement for distribution charts. The summary table reports missing values per group, since nothing is imputed.",
    "Callbacks: value and usage": "Small multiples (one panel per plan, same axes) replace a single scatter with five colours, which would be hard to read.",
    "Callbacks: lifecycle": "Retention uses the Kaplan-Meier estimator from `lifelines`. Active members are treated as censored: we know they stayed at least until 1 January 2024, not when they will leave. Ignoring this would make recent plans look worse than they are.",
    "Callbacks: add-ons": "Lift compares how often two add-ons are used together with how often that would happen by chance (P(A and B) / P(A)P(B)). The colour scale is centred on a lift of 1 and uses log2, so 'twice as often' and 'half as often' are the same distance from the centre.",
    "Callbacks: data quality": "This tab ignores the 'exclude flagged rows' switch, so it always shows the full extent of the cleaning.",
    "Callbacks: drill-down and member explorer": "Clicking a bar segment or heatmap cell stores the selection and opens the Member explorer tab with those members. The list can be sorted, filtered and downloaded as CSV.",
}

cells = [nbf.v4.new_markdown_cell("""# 04 · Interactive EDA dashboard (Dash)

This notebook builds the dashboard submitted for the Deliverable 1 bonus option *Interactive EDA Dashboard*. It reads `data/processed/acdc_dash.parquet` (created by notebook 03) and lets a reader filter the members, compare distributions and drill down to individual members.

The same code is saved as `dashboard/app.py`, which is the recommended way to run it:

```
python dashboard/app.py
```

and then open http://127.0.0.1:8050 in a browser. The last cell of this notebook can also start it from Jupyter.

## Why Dash, what it adds, and the challenges

**Why Dash.** We compared Plotly Dash, Streamlit and a static HTML page. Dash was chosen because its callbacks link charts to each other: a click on one chart can filter another view, which is what drill-down needs. It also gives full control over layout and styling, and it is plain Python, so it reuses the pandas code from the EDA. Streamlit is faster to build but reruns the whole script on each interaction and makes cross-chart clicks harder. A static HTML page needs no server but would require custom JavaScript for any interaction.

**What it adds for stakeholders.** The notebooks show fixed views chosen by us. The dashboard lets a reader ask their own questions: for example, how dormant dedicated-desk members differ from active ones, or whether a pattern holds once the rows flagged by cleaning are excluded. Every chart responds to the same filters, and any group seen in a chart can be listed and exported member by member.

**Challenges.**
- *Dirty data.* The dashboard runs on cleaned data, but about one row in five carries at least one cleaning flag. We kept a flag for every rule, added a switch that excludes flagged rows, and gave data quality its own tab.
- *Tenure bias.* Totals such as lifetime value grow with time as a member. We show rates (value per month, visits per plan day) and use a survival curve for retention, so recent members are not penalised.
- *Too many categories for colour.* Five plans in one scatter plot are hard to tell apart, so we used small multiples. Colours were checked for colour-blind separation, and every chart has labels or a legend so colour is never the only cue.
- *Keeping the notebook and the app the same.* The notebook is generated from `app.py`, so there is only one version of the code.
""")]

for title, code in blocks:
    if title == "Run":
        continue
    note = NOTES.get(title)
    cells.append(nbf.v4.new_markdown_cell(f"## {title}" + (f"\n\n{note}" if note else "")))
    cells.append(nbf.v4.new_code_cell(code))

cells.append(nbf.v4.new_markdown_cell("""## Smoke test

Each callback is called once with the default filters (no selection) and once with a selection that matches no members. This checks that every tab renders, including the empty case, without starting the server."""))
cells.append(nbf.v4.new_code_cell('''DEFAULT = ([], [], [], [], [0, len(COHORTS) - 1], "All", False)
EMPTY = (["Private office"], ["New"], ["Ireland"], ["Legal"], [0, 0], "Team account", True)

for filters in (DEFAULT, EMPTY):
    overview(*filters); value_usage(*filters); lifecycle(*filters); addons(*filters); quality(*filters)
    for var in NUMERIC:
        distributions(var, "plan", True, *filters)

rows, label = explorer({"plan": "Flex", "member_state": "Dormant"}, *DEFAULT)
print(label)
print("All callbacks ran without errors.")'''))

cells.append(nbf.v4.new_markdown_cell("""## Start the dashboard

`jupyter_mode="external"` starts the server in the background and prints the address to open. Set `START_SERVER = True` to run it; it is off by default so the notebook can be executed from top to bottom without leaving a server running."""))
cells.append(nbf.v4.new_code_cell('''START_SERVER = False

if START_SERVER:
    app.run(jupyter_mode="external", port=8050)'''))

cells.append(nbf.v4.new_markdown_cell("""## Screenshots

Taken from the running app with all members selected.

![Overview](../docs/dashboard/overview.png)

![Value and usage](../docs/dashboard/value_usage.png)

![Lifecycle](../docs/dashboard/lifecycle.png)

![Add-ons](../docs/dashboard/addons.png)"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
(ROOT / "notebook" / "04_dash_app.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print(f"written with {len(cells)} cells")
