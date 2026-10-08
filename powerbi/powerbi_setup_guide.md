# Module C: Power BI Integration & Dashboard Architecture Guide

## 1. Executive Summary & Objective
The Power BI Institutional Prospecting Dashboard transforms predictive lead probability scores and generative AI talking points into an interactive executive cockpit. Strategy and sales development teams can seamlessly filter top-tier accounts, analyze corporate creditworthiness against revenue velocity, and instantly access hyper-personalized outreach dossiers.

---

## 2. Power Query Ingestion (M-Script)
To ingest the exported pipeline data directly into Power BI Desktop:

1. Open **Power BI Desktop** -> **Home** -> **Get Data** -> **Blank Query**.
2. Click **Advanced Editor** and paste the following M script:

```powerquery
let
    Source = Csv.Document(File.Contents("C:\Users\USER\Desktop\Data Analyst\AI-Powered Institutional Prospecting Engine\data\institutional_prospects_powerbi.csv"), [Delimiter=",", Columns=30, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"company_id", type text},
        {"company_name", type text},
        {"ticker", type text},
        {"sector", type text},
        {"sub_sector", type text},
        {"headquarters", type text},
        {"employee_count", Int64.Type},
        {"annual_revenue_m", type number},
        {"revenue_velocity_yoy", type number},
        {"revenue_cagr_3yr", type number},
        {"ebitda_m", type number},
        {"net_margin", type number},
        {"debt_to_equity", type number},
        {"current_ratio", type number},
        {"quick_ratio", type number},
        {"altman_z_score", type number},
        {"credit_risk_tier", type text},
        {"market_cap_m", type number},
        {"market_cap_tier", type text},
        {"event_type", type text},
        {"headline", type text},
        {"signal_sentiment_score", type number},
        {"lead_probability_score", type number},
        {"priority_tier", type text},
        {"qualification_status", type text},
        {"prospecting_summary", type text},
        {"market_signals", type text},
        {"talking_points", type text},
        {"pitch_angle", type text},
        {"objection_handler", type text}
    })
in
    #"Changed Type"
```

Rename the query to `Prospects`.

---

## 3. Data Model Schema
The dataset functions as a consolidated high-performance Star-Schema reporting entity:

| Column Name | Data Type | Role | Business Definition |
| :--- | :--- | :--- | :--- |
| `company_id` | Text | Primary Key | Unique corporate identification key |
| `company_name` | Text | Dimension | Full corporate legal name |
| `ticker` | Text | Dimension | Stock ticker / shorthand identifier |
| `sector` / `sub_sector` | Text | Hierarchy | Macro sector and granular industry vertical |
| `annual_revenue_m` | Decimal | Metric | Trailing Twelve Months (TTM) Revenue in $M |
| `revenue_velocity_yoy`| Decimal (%) | Metric | Year-over-Year revenue growth rate |
| `altman_z_score` | Decimal | Metric | Credit solvency proxy (>2.6 Safe, <1.1 Distress) |
| `debt_to_equity` | Decimal | Metric | Balance sheet leverage ratio |
| `lead_probability_score` | Decimal (0-100)| Core Target | Module A ML calibrated conversion likelihood |
| `priority_tier` | Text | Filter/Slicer | Strategic Alpha, High Conviction, Medium, Monitor |
| `prospecting_summary` | Text | GenAI Output | Module B LangChain synthesized executive dossier |
| `talking_points` | Text | GenAI Output | Module B 3-point C-suite conversation openers |
| `pitch_angle` | Text | GenAI Output | Module B strategic commercial angle |

---

## 4. Recommended Dashboard Layout & Visual Components

### Page 1: Institutional Prospecting Command Center

```
+----------------------------------------------------------------------------------------------------+
|  TOP NAVIGATION & GLOBAL SLICERS: [Sector: All v]  [Priority Tier: All v]  [Min Score Slider: 0-100] |
+----------------------------------------------------------------------------------------------------+
| [ KPI 1: Total Leads ] | [ KPI 2: High-Conviction ] | [ KPI 3: Avg Score ] | [ KPI 4: Weighted Pipe ]|
|        50 Accounts     |        24 Targets          |       71.4 / 100     |         $18.4M ARR   |
+---------------------------------------+------------------------------------------------------------+
| VISUAL A: Credit Health vs Growth     | VISUAL B: Account Leaderboard Table                        |
| Scatter Chart:                        | Columns:                                                   |
| - X-Axis: Altman Z-Score (Solvency)   | • Company Name | Ticker                                    |
| - Y-Axis: YoY Revenue Velocity %      | • Sector | Annual Revenue ($M)                             |
| - Bubble Size: Market Cap             | • Altman Z-Score | Debt-to-Equity                          |
| - Legend: Priority Tier               | • Lead Probability Score (Data Bars / Conditional Color)   |
| - Quadrant Line: Z=2.6 & Growth=15%   | • Qualification Status                                     |
+---------------------------------------+------------------------------------------------------------+
| VISUAL C: AI Strategic Prospecting Dossier (Dynamic Card updating on Account Row Selection)        |
|----------------------------------------------------------------------------------------------------|
| Account: NovaScale AI Systems (NVSC) | Tier 1: Strategic Alpha | Lead Score: 94.2 / 100            |
| Executive Summary:                                                                                 |
| NovaScale exhibits robust top-line momentum (48.2% YoY growth) supported by an elite Altman        |
| Z-Score of 4.12. Catalyst 'Capital Raise' signals immediate budget for AI infrastructure.          |
|----------------------------------------------------------------------------------------------------|
| Strategic Pitch Angle: Scalable Enterprise Infrastructure & Rapid Time-to-Value Acceleration       |
|----------------------------------------------------------------------------------------------------|
| C-Suite Outreach Talking Points:                                                                   |
| • Congratulate leadership on recent $140M growth equity raise and 48% YoY scale velocity.         |
| • Explore how their platform architecture handles surging transaction throughput without margin dip.|
| • Position our accelerated deployment blueprint to compress procurement lead-time by 40%.         |
+----------------------------------------------------------------------------------------------------+
```

### Visual Specifications:
1. **Header KPI Cards**:
   - `[Total Institutional Leads]`
   - `[High Conviction Targets]`
   - `[Average Lead Probability Score]`
   - `[Probability-Weighted Pipeline ($M)]`

2. **Visual A: Strategic Quadrant Scatter Matrix**:
   - **X-Axis**: `Prospects[altman_z_score]` (Reference line at `2.6` for Safe Zone boundary).
   - **Y-Axis**: `Prospects[revenue_velocity_yoy]` (Reference line at `15%`).
   - **Values**: `Prospects[company_name]`.
   - **Size**: `Prospects[market_cap_m]`.
   - **Legend**: `Prospects[priority_tier]`.
   - **Strategic Value**: Instantly isolates upper-right quadrant companies: *Accelerating Growth + Fortress Balance Sheet*.

3. **Visual B: Dynamic Target Leaderboard Table**:
   - Include conditional formatting on `lead_probability_score` using a continuous gradient from Orange (`#D83B01`) to Deep Green (`#107C41`).
   - Add a Bookmark / Drill-Through to detailed company financial statements.

4. **Visual C: AI-Generated Account Intelligence Dossier**:
   - Use custom Card / HTML Content / Multi-row card visual bound to measures:
     - `[Selected Account Name]`
     - `[Selected Account Score]`
     - `[Selected Account AI Summary]`
     - `[Selected Account Talking Points]`
     - `[Selected Account Pitch Angle]`
     - `[Selected Account Objection Handler]`
   - Dynamically updates as sales reps click on any row in the Leaderboard.
