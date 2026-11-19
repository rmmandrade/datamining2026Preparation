# Dashboard data dictionary

Engineered and flag columns in `data/processed/acdc_dash.parquet`, created by `notebook/03_dash_features.ipynb`.
Original columns keep the meaning given in the project guidelines.

| Column | Definition |
|---|---|
| `is_active` | True when the raw end date was the 1970-01-01 placeholder |
| `tenure_days` | Days from start to end date, or to 2024-01-01 for active members |
| `tenure_months` | tenure_days / 30.44 |
| `start_cohort` | Quarter of membership_start, e.g. 2023Q4 |
| `monthly_value` | lifetime_value / tenure_months, with tenure floored at one month |
| `value_band` | Low / Mid / High by terciles of monthly_value |
| `utilization_rate` | actual_visits_last_period / entitled_visits_last_period |
| `visits_per_plan_day` | avg_weekly_visits / plan_days_per_week |
| `recency_days` | days_since_last_visit, negatives set to 0 |
| `recency_band` | recency_days grouped into 0-7, 8-30, 31-60, 61-180 and 180+ days |
| `addon_count` | Number of the nine add-ons the member has used |
| `member_state` | Churned (has end date), New (active, tenure <= 90 days), Dormant (active, > 60 days since last visit), Active (other active members) |
| `quality_issues` | Number of flag_* columns that are True |
| `flag_age` | True when this cleaning rule changed or questioned the row |
| `flag_plan_days` | True when this cleaning rule changed or questioned the row |
| `flag_entitled_visits` | True when this cleaning rule changed or questioned the row |
| `flag_days_code_999` | True when this cleaning rule changed or questioned the row |
| `flag_colleagues` | True when this cleaning rule changed or questioned the row |
| `flag_negative_addon` | True when this cleaning rule changed or questioned the row |
| `flag_negative_value` | True when this cleaning rule changed or questioned the row |
| `flag_start_after_end` | True when this cleaning rule changed or questioned the row |
| `flag_swipe_outside` | True when this cleaning rule changed or questioned the row |
| `flag_last_period_above_total` | True when this cleaning rule changed or questioned the row |
| `flag_swipe_after_end` | True when this cleaning rule changed or questioned the row |
| `flag_extreme_value` | True when this cleaning rule changed or questioned the row |
