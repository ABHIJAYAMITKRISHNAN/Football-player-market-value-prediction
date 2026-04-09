# ⚽ Football Player Transfer Market Intelligence Platform

> A full-stack Machine Learning system for football player valuation, scouting analytics, and transfer market insights — built on real Transfermarkt data.

---

## 📁 Project Structure

```
football-ml-platform/
│
├── data/
│   └── football.csv                  # Raw dataset (10,754 players, 22 features)
│
├── notebooks/                        # Exploratory notebooks (optional)
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data cleaning, feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── price_predictor.py        # Module 1: Market value prediction
│   │   ├── depreciation.py           # Module 2: Value depreciation/appreciation
│   │   ├── segmentation.py           # Module 3: Player archetypes clustering
│   │   ├── similarity.py             # Module 4: Similar player finder
│   │   ├── valuation_detector.py     # Module 5: Overvalued/undervalued detection
│   │   ├── performance_ranking.py    # Module 6: Position-wise ranking
│   │   └── injury_classifier.py      # Module 7: Injury risk classifier
│   └── utils.py                      # Shared helpers, plotting utils
│
├── saved_models/                     # Joblib-serialized trained models
│   ├── price_predictor.pkl
│   ├── depreciation_model.pkl
│   ├── kmeans_clusters.pkl
│   ├── scaler.pkl
│   ├── injury_classifier.pkl
│   └── label_encoders.pkl
│
├── frontend/
│   ├── app.py                        # Streamlit frontend entry point
│   ├── pages/
│   │   ├── 01_price_prediction.py
│   │   ├── 02_depreciation.py
│   │   ├── 03_archetypes.py
│   │   ├── 04_similar_players.py
│   │   ├── 05_valuation.py
│   │   ├── 06_performance_ranking.py
│   │   └── 07_injury_risk.py
│   └── components/
│       ├── charts.py                 # Reusable Plotly chart builders
│       └── styles.py                 # Custom CSS and theming
│
├── train_all.py                      # Master script: trains all models
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Reference

**File:** `data/football.csv`  
**Rows:** 10,754 players | **Columns:** 22

| Column | Type | Description |
|---|---|---|
| `player` | str | Transfermarkt profile URL (unique ID) |
| `team` | str | Current club |
| `name` | str | Player full name |
| `position` | str | Playing position (15 unique values) |
| `height` | int | Height in cm |
| `age` | int | Age in years |
| `appearance` | int | Number of appearances |
| `goals` | float | Goals per game (already normalized) |
| `assists` | float | Assists per game (already normalized) |
| `yellow cards` | float | Yellow cards per game (already normalized) |
| `second yellow cards` | int | Second yellow cards (raw count) |
| `red cards` | int | Red cards (raw count) |
| `goals conceded` | float | Goals conceded per game (GK/DEF) |
| `clean sheets` | float | Clean sheets per game |
| `minutes played` | int | Total minutes played |
| `days_injured` | int | Total days injured |
| `games_injured` | int | Games missed due to injury |
| `award` | int | Number of awards won |
| `current_value` | int | Current market value (EUR) |
| `highest_value` | int | Career-peak market value (EUR) |
| `position_encoded` | int | Numeric encoding of position |
| `winger` | int | Binary: 1 if winger |

**All 15 positions in the dataset:**
```
Goalkeeper, Defender Centre-Back, Defender Left-Back, Defender Right-Back,
midfield-DefensiveMidfield, midfield-CentralMidfield, midfield-AttackingMidfield,
midfield-RightMidfield, midfield-LeftMidfield, Attack-LeftWinger,
Attack-RightWinger, Attack Centre-Forward, Attack-SecondStriker, midfield, Attack
```

---

## 🧰 Tech Stack & Libraries

Install all dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
xgboost>=2.0.0
statsmodels>=0.14.0
shap>=0.43.0
joblib>=1.3.0
streamlit>=1.28.0
streamlit-extras>=0.3.0
streamlit-option-menu>=0.3.6
scipy>=1.11.0
imbalanced-learn>=0.11.0
```

---

## ⚙️ Preprocessing (`src/preprocessing.py`)

This module is shared across all other modules. Build it first.

### Tasks:
1. **Load data** from `data/football.csv`

2. **Data cleaning:**
   - Drop duplicate rows (if any) based on the `player` URL column
   - **Missing value handling:**
     - Numerical columns (`height`, `appearance`, `goals`, `assists`, `yellow cards`, `second yellow cards`, `red cards`, `goals conceded`, `clean sheets`, `minutes played`, `days_injured`, `games_injured`, `award`, `current_value`, `highest_value`): fill NaNs with `0`
     - Categorical columns (`position`, `team`): fill NaNs with the string `'Unknown'`
     - Drop any row where both `current_value` and `highest_value` are 0 (unidentifiable records)
   - **Type enforcement** — cast columns to correct dtypes after loading to prevent silent errors:
     ```python
     INT_COLS = [
         'height', 'age', 'appearance', 'second yellow cards', 'red cards',
         'minutes played', 'days_injured', 'games_injured', 'award',
         'current_value', 'highest_value', 'position_encoded', 'winger'
     ]
     FLOAT_COLS = ['goals', 'assists', 'yellow cards', 'goals conceded', 'clean sheets']
     df[INT_COLS] = df[INT_COLS].fillna(0).astype(int)
     df[FLOAT_COLS] = df[FLOAT_COLS].fillna(0.0).astype(float)
     ```
   - **Outlier clipping** — clip the two heavily right-skewed value columns to the 99th percentile to prevent extreme values from distorting models (note: log-transform is still applied at training time per Critical Note #1):
     ```python
     for col in ['current_value', 'highest_value']:
         cap = df[col].quantile(0.99)
         df[col] = df[col].clip(upper=cap)
     ```

3. **Feature engineering:**
   - `value_drop_ratio = (highest_value - current_value) / (highest_value + 1)` — how much value has been lost from peak
   - `injury_burden = days_injured / (appearance + 1)` — injury days per appearance
   - `productivity_score = goals + assists` — combined attacking output per game
   - `age_group` — bin age into: `Youth (<=21)`, `Prime (22-28)`, `Senior (29-32)`, `Veteran (33+)`
   - `position_group` — simplify the 15 positions into 4 broad groups: `GK`, `DEF`, `MID`, `FWD`
     - GK: `Goalkeeper`
     - DEF: any position containing `Defender`
     - MID: any position containing `midfield`
     - FWD: any position containing `Attack`
   - `injury_risk_label` — engineer here (not just in Module 7) so it is available to all modules via the shared preprocessed DataFrame:
     ```python
     def label_injury_risk(row):
         score = 0
         if row['days_injured'] > 180: score += 2
         elif row['days_injured'] > 60: score += 1
         if row['games_injured'] > 20: score += 2
         elif row['games_injured'] > 7: score += 1
         if row['age'] > 30: score += 1
         if score >= 4: return 'High'
         elif score >= 2: return 'Medium'
         else: return 'Low'
     df['injury_risk_label'] = df.apply(label_injury_risk, axis=1)
     ```

4. **Encoding:** Label-encode the following columns and save all encoders in a single dict to `saved_models/label_encoders.pkl`:
   - `position` → `position_encoded` (overwrite the existing column; it is already present in the CSV but re-encode for consistency)
   - `team` → `team_encoded`
   - `position_group` → `position_group_encoded`
   - `age_group` → `age_group_encoded` ← **required by Module 2**
   - `injury_risk_label` → `injury_risk_encoded` ← **required by Module 7**
   ```python
   from sklearn.preprocessing import LabelEncoder
   import joblib

   ENCODE_COLS = ['position', 'team', 'position_group', 'age_group', 'injury_risk_label']
   label_encoders = {}
   for col in ENCODE_COLS:
       le = LabelEncoder()
       df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
       label_encoders[col] = le
   joblib.dump(label_encoders, 'saved_models/label_encoders.pkl')
   ```

5. **Scaling:** Fit `StandardScaler` on `BASE_FEATURES` columns of the **training split only** (when called from a model module). In `load_and_preprocess`, return the full cleaned/engineered DataFrame unscaled — each model module handles its own train/test split before fitting the scaler. Save the scaler once (fitted on the price predictor's training split in Module 1) to `saved_models/scaler.pkl` for reuse across all modules.

6. **Return contract** — `load_and_preprocess(path: str) -> pd.DataFrame` must return the full cleaned DataFrame with all engineered columns and encoded columns added. It must **not** drop the original string columns (`position`, `team`, `name`, `player`, `age_group`, `position_group`, `injury_risk_label`) — these are needed for display and filtering in the frontend.

### Feature Sets (define as constants, reused across modules):

```python
BASE_FEATURES = [
    'height', 'age', 'appearance', 'goals', 'assists',
    'yellow cards', 'red cards', 'minutes played',
    'days_injured', 'games_injured', 'award',
    'position_encoded', 'winger',
    'value_drop_ratio', 'injury_burden', 'productivity_score'
]

GK_EXTRA_FEATURES = ['goals conceded', 'clean sheets']

TARGET_VALUE = 'current_value'
TARGET_DEPRECIATION = 'value_drop_ratio'
TARGET_INJURY = 'injury_risk_label'  # already engineered in load_and_preprocess
```

---

## 🧠 Module 1 — Market Value Prediction (`src/models/price_predictor.py`)

### Goal
Predict `current_value` (EUR) for any player given their stats.

### Approach
- **Algorithm:** XGBoost Regressor (primary), Random Forest Regressor (baseline comparison)
- **Target:** `current_value` — apply `np.log1p` before training; inverse with `np.expm1` at prediction time
- **Features:** `BASE_FEATURES`
- **Train/Test Split:** 80/20 stratified by `position_group`
- **Validation:** 5-Fold Cross-Validation, report mean RMSE and R²

### Steps:
1. Log-transform target: `y = np.log1p(df['current_value'])`
2. Train XGBoost with hyperparameter tuning via `RandomizedSearchCV`:
   ```
   n_estimators: [200, 300, 500]
   max_depth: [4, 6, 8]
   learning_rate: [0.05, 0.1, 0.15]
   subsample: [0.8, 1.0]
   colsample_bytree: [0.8, 1.0]
   ```
3. Evaluate: RMSE, MAE, R² on test set
4. **SHAP Analysis:**
   - Use `shap.TreeExplainer(model)`
   - Generate summary plot (bar) + waterfall for single prediction
   - Save summary plot to `saved_models/shap_price_summary.png`
5. Save model: `joblib.dump(model, 'saved_models/price_predictor.pkl')`

### Output for frontend:
- Predicted market value (formatted: `€12,500,000`)
- SHAP waterfall chart for the individual prediction
- Actual vs Predicted scatter plot (log scale)
- Model performance metrics displayed as KPI cards

---

## 📉 Module 2 — Value Depreciation / Appreciation Forecasting (`src/models/depreciation.py`)

### Goal
Predict whether a player's value will **increase, stay stable, or depreciate** based on their current profile.

### Target Engineering:
```python
def label_trajectory(row):
    ratio = row['value_drop_ratio']
    if ratio < 0.10:
        return 'Appreciating'       # < 10% drop from peak = still growing
    elif ratio < 0.40:
        return 'Stable'             # 10-40% drop from peak
    else:
        return 'Depreciating'       # > 40% drop from peak
```

### Approach
- **Algorithm:** XGBoost Classifier (primary); Logistic Regression (for interpretability)
- **Key Features:** `age`, `age_group_encoded`, `appearance`, `injury_burden`, `days_injured`, `productivity_score`, `award`, `position_encoded`
- Handle class imbalance with `class_weight='balanced'`

### Steps:
1. Engineer `trajectory_label` column
2. Train classifier; report classification report + confusion matrix
3. Optionally use Statsmodels Logistic Regression for coefficient table
4. Save: `joblib.dump(model, 'saved_models/depreciation_model.pkl')`

### Output for frontend:
- Trajectory class badge: `Appreciating / Stable / Depreciating`
- Probability confidence bars for all 3 classes
- Feature importance chart
- Age vs Average Value line chart grouped by position (annotate the queried player)

---

## 👥 Module 3 — Player Archetypes / Segmentation (`src/models/segmentation.py`)

### Goal
Cluster players into meaningful archetypes using unsupervised learning.

### Features for clustering:
```python
CLUSTER_FEATURES = [
    'age', 'goals', 'assists', 'appearance', 'minutes played',
    'days_injured', 'yellow cards', 'red cards', 'award',
    'productivity_score', 'injury_burden', 'position_encoded'
]
```

### Algorithm: K-Means Clustering
- Scale features with StandardScaler before clustering
- Determine optimal k using Elbow Method (k=2 to 12) and Silhouette Score
- Fit final K-Means with optimal k (likely 6-9 clusters)

### Archetype Labeling:
After fitting, inspect cluster centroids and assign human-readable labels. Likely candidates:
- `Elite Attacker` — high goals/assists, high value, low injury
- `Workhorse Midfielder` — high appearances/minutes, avg goals
- `Injury-Prone Veteran` — high age, high injury days, declining value
- `Rising Talent` — low age, low injury, growing value
- `Defensive Wall` — high clean sheets, low goals, solid value
- `Impact Substitute` — low appearance but high per-game stats
- `Declining Veteran` — high age, low value, was once highly valued
- *(Adjust labels based on actual cluster centroids observed)*

### Steps:
1. Scale features
2. Elbow Method: plot inertia for k=2 to k=12
3. Silhouette Analysis: find k with highest score
4. Fit K-Means with optimal k
5. PCA (2D) for visualization: `from sklearn.decomposition import PCA`
6. Save: `joblib.dump(kmeans, 'saved_models/kmeans_clusters.pkl')`

### Output for frontend:
- Interactive Plotly scatter (PCA 2D), color-coded by archetype, player name on hover
- Radar/spider chart per archetype (centroid values on 6 key stats)
- Any player's archetype label shown on their profile page

---

## 🔍 Module 4 — Similar Player Finder (`src/models/similarity.py`)

### Goal
Given a player's name, return the top-N most statistically similar players.

### Approach
- **Algorithm:** Cosine Similarity on scaled feature vectors
- **Features:** `CLUSTER_FEATURES` from Module 3, scaled with the same saved scaler
- No training needed — compute similarity at query time

### Implementation:
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_players(player_name, df_scaled, df_original, n=10, same_position=False):
    idx = df_original[df_original['name'] == player_name].index[0]
    query_vec = df_scaled[idx].reshape(1, -1)
    scores = cosine_similarity(query_vec, df_scaled)[0]
    scores[idx] = -1  # exclude self
    if same_position:
        pos = df_original.loc[idx, 'position_group']
        mask = df_original['position_group'] != pos
        scores[mask] = -1
    top_n = np.argsort(scores)[::-1][:n]
    result = df_original.iloc[top_n].copy()
    result['similarity_score'] = (scores[top_n] * 100).round(1)
    return result
```

### Output for frontend:
- Table: Name, Team, Position, Age, Current Value, Similarity %
- Side-by-side radar chart: query player vs best match (6 key stats)
- "Transfer Replacement" framing: position this as finding alternatives to a player

---

## 📊 Module 5 — Overvalued / Undervalued Player Detection (`src/models/valuation_detector.py`)

### Goal
Compare predicted vs actual `current_value` to flag market inefficiencies.

### Steps:
1. Load trained price predictor from `saved_models/price_predictor.pkl`
2. Run predictions on entire dataset
3. Compute valuation gap:
   ```python
   df['predicted_value'] = np.expm1(model.predict(X))  # inverse log transform
   df['value_gap'] = df['current_value'] - df['predicted_value']
   df['value_gap_pct'] = df['value_gap'] / (df['predicted_value'] + 1) * 100

   def classify_valuation(pct):
       if pct > 30:
           return 'Overvalued'
       elif pct < -30:
           return 'Undervalued'
       else:
           return 'Fairly Valued'

   df['valuation_label'] = df['value_gap_pct'].apply(classify_valuation)
   ```
4. Save results as `saved_models/valuation_results.parquet` for fast frontend loading

### Output for frontend:
- Top 20 most undervalued players (best bargains — sorted by largest negative gap)
- Top 20 most overvalued players (overpaid — sorted by largest positive gap)
- Scatter: Actual vs Predicted value (color by valuation label)
- Filter controls: position, team, age range

---

## 🏆 Module 6 — Position-wise Performance Ranking (`src/models/performance_ranking.py`)

### Goal
Rank players within each position using a composite weighted performance score.

### Scoring Formulas
Normalize all component features to [0, 1] within position group before applying weights.

```python
# Forwards (FWD)
score = (goals * 0.40) + (assists * 0.25) + (appearance_norm * 0.10) + 
        (minutes_played_norm * 0.10) + (award_norm * 0.15)

# Midfielders (MID)
score = (goals * 0.20) + (assists * 0.30) + (appearance_norm * 0.20) + 
        (minutes_played_norm * 0.15) + (award_norm * 0.15)

# Defenders (DEF)
score = (clean_sheets * 0.30) + (appearance_norm * 0.20) + 
        (minutes_played_norm * 0.15) + (red_cards_inv * 0.15) + 
        (yellow_cards_inv * 0.05) + (award_norm * 0.15)
# Note: red_cards_inv = 1 - normalized(red_cards)

# Goalkeepers (GK)
score = (clean_sheets * 0.40) + (goals_conceded_inv * 0.25) + 
        (appearance_norm * 0.15) + (minutes_played_norm * 0.10) + 
        (award_norm * 0.10)
# Note: goals_conceded_inv = 1 - normalized(goals_conceded)
```

### Steps:
1. Group by `position_group`
2. Normalize all feature components within group
3. Compute composite score per group
4. Rank within each group
5. Save: `saved_models/performance_rankings.parquet`

### Output for frontend:
- Top 10 leaderboard per position (interactive animated bar chart)
- Full sortable/filterable table per position
- Radar chart for top player in each position
- Club-level leaderboard: average composite score per team

---

## 🏥 Module 7 — Injury Risk Classifier (`src/models/injury_classifier.py`)

### Goal
Classify each player into injury risk tier: `Low`, `Medium`, or `High`.

### Target Engineering:
```python
def label_injury_risk(row):
    score = 0
    if row['days_injured'] > 180:
        score += 2
    elif row['days_injured'] > 60:
        score += 1
    if row['games_injured'] > 20:
        score += 2
    elif row['games_injured'] > 7:
        score += 1
    if row['age'] > 30:
        score += 1
    if score >= 4:
        return 'High'
    elif score >= 2:
        return 'Medium'
    else:
        return 'Low'

df['injury_risk_label'] = df.apply(label_injury_risk, axis=1)
```

### Approach
- **Algorithm:** Random Forest Classifier (primary), compare with XGBoost
- **Features:**
  ```python
  INJURY_FEATURES = [
      'age', 'height', 'appearance', 'minutes played',
      'days_injured', 'games_injured', 'position_encoded',
      'yellow cards', 'red cards', 'injury_burden'
  ]
  ```
- Handle class imbalance: `class_weight='balanced'` (or SMOTE from imbalanced-learn)
- Evaluate: Classification report, confusion matrix, ROC-AUC (one-vs-rest)

### Steps:
1. Engineer `injury_risk_label`
2. Train classifier with 5-fold cross-validation
3. SHAP analysis: `shap.TreeExplainer` for feature importance
4. Save: `joblib.dump(model, 'saved_models/injury_classifier.pkl')`

### Output for frontend:
- Risk tier badge: Low / Medium / High with color coding
- Probability bars for all 3 risk tiers
- SHAP waterfall for individual player prediction
- Heatmap: injury risk distribution across top clubs
- Scatter: age vs days_injured, colored by risk tier

---

## 🖥️ Frontend — Streamlit App

### Theming (`.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#00ff87"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#161b22"
textColor = "#e6edf3"
font = "sans serif"
```

### Custom CSS (`frontend/components/styles.py`)
Inject via `st.markdown(css, unsafe_allow_html=True)` on every page:

```css
/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,255,135,0.08);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value { font-size: 2rem; font-weight: 700; color: #00ff87; }
.kpi-delta { font-size: 0.85rem; color: #8b949e; }
.kpi-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Section Header */
.section-header {
    font-size: 1.4rem; font-weight: 600;
    border-left: 4px solid #00ff87;
    padding-left: 12px; margin: 24px 0 16px 0;
    color: #e6edf3;
}

/* Risk Badges */
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-low    { background: rgba(35,134,54,0.25);  color: #3fb950; border: 1px solid #238636; }
.badge-medium { background: rgba(158,106,3,0.25);  color: #d29922; border: 1px solid #9e6a03; }
.badge-high   { background: rgba(218,54,51,0.25);  color: #f85149; border: 1px solid #da3633; }

/* Valuation Badges */
.badge-under  { background: rgba(0,255,135,0.15);  color: #00ff87; border: 1px solid #00ff87; }
.badge-over   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }
.badge-fair   { background: rgba(139,148,158,0.15); color: #8b949e; border: 1px solid #8b949e; }

/* Player Card */
.player-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px; margin: 8px 0;
}
.player-name { font-size: 1.1rem; font-weight: 600; color: #e6edf3; }
.player-meta { font-size: 0.8rem; color: #8b949e; }

/* Progress Bar (Similarity Score) */
.progress-bar-bg { background: #21262d; border-radius: 4px; height: 8px; }
.progress-bar-fill { background: linear-gradient(90deg, #00ff87, #00d4ff); border-radius: 4px; height: 8px; }
```

### Plotly Chart Theme (`frontend/components/charts.py`)
Apply this layout to every Plotly figure for visual consistency:

```python
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e6edf3', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    colorway=['#00ff87', '#00d4ff', '#ffd700', '#ff6b6b', '#c084fc', '#fb923c'],
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor='#161b22', bordercolor='#30363d', font_color='#e6edf3'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#30363d')
)
```

---

### Page Specifications

#### `frontend/app.py` — Home Dashboard
- Header with logo emoji and platform title
- Navigation via `streamlit-option-menu` in sidebar (icons for each module)
- 4 KPI cards (2x2 grid using `st.columns`):
  - Total Players in Dataset
  - Average Market Value (formatted EUR)
  - Highest Value Player (name + value)
  - Most Injury-Prone Position
- Player count bar chart by position group (Plotly)
- Market value distribution histogram (log scale, Plotly)
- Quick search: type a player name → jump to their full profile

#### `frontend/pages/01_price_prediction.py`
- Two modes via toggle: "Search Existing Player" | "Manual Input"
  - Search mode: autocomplete dropdown → auto-fill all stats → predict
  - Manual mode: sliders + dropdowns for each feature
- "Predict Value" button → large KPI card showing predicted value in EUR
- Confidence range: show ±15% band
- SHAP waterfall chart explaining the prediction
- Below: similar players table (from Module 4) as comparison

#### `frontend/pages/02_depreciation.py`
- Player search (autocomplete)
- Trajectory badge prominently displayed
- Probability confidence bars (3 classes, styled with colors)
- Line chart: average value by age bracket for same position group
  - Mark the queried player's position on the chart
- Feature importance bar chart

#### `frontend/pages/03_archetypes.py`
- Full-width PCA scatter plot: all 10,754 players, colored by archetype
  - Hover shows: player name, team, archetype, current value
- Sidebar: filter by position group, age group → highlights matching points
- Archetype selector: choose a cluster → show centroid radar chart + top 10 players table
- Any player search → highlight their dot on the scatter, show their archetype info panel

#### `frontend/pages/04_similar_players.py`
- Player search → top 10 results table
  - Columns: Rank, Name, Team, Position, Age, Value, Similarity %
  - Similarity % shown as inline progress bar
- Side-by-side radar chart: query player (green) vs #1 match (blue) — 6 stats
- Toggle: "Same Position Only" checkbox
- Use case framing: "Who can replace [Player]?"

#### `frontend/pages/05_valuation.py`
- Main scatter: Actual Value vs Predicted Value
  - Color by valuation label (green=undervalued, red=overvalued, grey=fair)
  - Hover: player name, team, gap %
- Two tabs: "Top Undervalued" | "Top Overvalued" (20 players each, sortable table)
- Sidebar filters: position group, team, age range
- Individual player lookup: name → valuation badge + gap % + gap EUR amount

#### `frontend/pages/06_performance_ranking.py`
- Position group tabs: GK / DEF / MID / FWD (use `st.tabs`)
- Per tab:
  - Animated horizontal bar chart (top 20 players by composite score)
  - Full sortable table below
- Club Leaderboard section: average composite score per club (bar chart, top 20 clubs)
- Player Comparison: select 2 players → radar chart overlay

#### `frontend/pages/07_injury_risk.py`
- Player search → risk badge (Low / Medium / High) displayed prominently
- Probability bars for all 3 tiers
- SHAP waterfall chart for the prediction
- Bottom section:
  - Scatter: age vs days_injured, colored by risk tier (all players, filter by position)
  - Club heatmap: injury risk composition by club (stacked bar: % Low / Medium / High)

---

## 🚀 Training Pipeline (`train_all.py`)

Run this once before launching the frontend:

```python
"""
Master training script — trains and saves all models.
Run: python train_all.py
"""
import os
os.makedirs("saved_models", exist_ok=True)

from src.preprocessing import load_and_preprocess
from src.models.price_predictor import train_price_model
from src.models.depreciation import train_depreciation_model
from src.models.segmentation import train_segmentation_model
from src.models.injury_classifier import train_injury_model
from src.models.valuation_detector import compute_valuation_gaps
from src.models.performance_ranking import compute_rankings

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess("data/football.csv")

    print("[1/6] Training price predictor...")
    train_price_model(df)

    print("[2/6] Training depreciation forecaster...")
    train_depreciation_model(df)

    print("[3/6] Training segmentation model...")
    train_segmentation_model(df)

    print("[4/6] Training injury risk classifier...")
    train_injury_model(df)

    print("[5/6] Computing valuation gaps...")
    compute_valuation_gaps(df)

    print("[6/6] Computing performance rankings...")
    compute_rankings(df)

    print("\nAll models trained and saved to saved_models/")
    print("Run: streamlit run frontend/app.py")
```

---

## 📏 Evaluation Summary

| Module | Algorithm | Primary Metric | Target |
|---|---|---|---|
| Price Prediction | XGBoost Regressor | R², RMSE | R² > 0.80 |
| Depreciation | XGBoost Classifier | F1 (weighted) | F1 > 0.75 |
| Segmentation | K-Means | Silhouette Score | > 0.35 |
| Similar Player | Cosine Similarity | Manual QA | — |
| Valuation | Derived from Module 1 | Value gap MAE | Minimize |
| Performance Rank | Weighted formula | Rank correlation | — |
| Injury Risk | Random Forest | ROC-AUC | AUC > 0.80 |

---

## 🏃 How to Run

```bash
# 1. Set up the project directory
mkdir football-ml-platform && cd football-ml-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
mkdir data
cp /path/to/football.csv data/football.csv

# 4. Train all models (runs in ~2-5 minutes)
python train_all.py

# 5. Launch the frontend
streamlit run frontend/app.py
```

---

## ⚠️ Critical Implementation Notes

1. **Log-transform market values.** `current_value` and `highest_value` are heavily right-skewed (max = €180M). Always apply `np.log1p()` before training regressors and `np.expm1()` at inference time.

2. **Stats are already per-game normalized.** Columns `goals`, `assists`, `yellow cards`, `goals conceded`, and `clean sheets` are already per-game rates in the CSV. Do NOT divide by `appearance` again.

3. **`player` column is a URL** (Transfermarkt path). Use it as a unique identifier only. Use `name` for display.

4. **Position-specific logic is mandatory** for Module 6 (ranking). GKs must never be ranked on goals; attackers must never be ranked on clean sheets. Always branch on `position_group`.

5. **Save and reuse the StandardScaler.** Fit it once on training data, save to `saved_models/scaler.pkl`, and load it for all inference. Never refit on test or inference data.

6. **Streamlit performance.** Use `@st.cache_data` for all data loading functions and `@st.cache_resource` for model loading to prevent reloading on every interaction.

7. **SHAP rendering in Streamlit.** Use `fig, ax = plt.subplots()` then `shap.plots.waterfall(shap_values[i], show=False)` then `st.pyplot(fig)`. Clear with `plt.clf()` after each render.

8. **Similarity module.** No model file needed. Save the scaled feature matrix as a numpy `.npy` or parquet file for fast load, alongside the original player metadata.

9. **Valuation module** depends on Module 1. Always run price predictor training first in `train_all.py`.

10. **Player name autocomplete.** Build a sorted list of all `name` values from the dataset and use `st.selectbox` with this list for all player search inputs.

---

## 🎨 Design Principles

The frontend should feel like a professional football analytics platform — think **Sofascore + FBref in dark mode**:

- **Data-first layout:** Charts dominate, text is minimal and purposeful
- **Consistent color language:**
  - Green (`#00ff87`) = positive performance, growth, undervalued, low risk
  - Red (`#f85149`) = overvalued, high risk, depreciation
  - Gold (`#ffd700`) = award winners, top-ranked players
  - Blue (`#00d4ff`) = comparison/secondary data
- **Interactivity everywhere:** Every Plotly chart has hover tooltips with player context
- **Progressive disclosure:** Overview first, drill into detail on click/search
- **Mobile-aware:** Use `st.columns` ratios that collapse gracefully on smaller screens

---

*Dataset: Transfermarkt player data | Framework: Streamlit + Scikit-learn + XGBoost | Visualization: Plotly*
