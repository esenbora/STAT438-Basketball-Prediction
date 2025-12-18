# STAT438 Project 2 - Basketball Prediction
## Group 9

---

# Project Overview

In this project, we used **first half data** from Turkish Basketball League (ING BSL) matches to perform three different prediction tasks:

1. **Game Winner Prediction** - Which team wins the game?
2. **Two-Point Trials Leader** - Which team will have more 2PT attempts?
3. **Turnover Leader** - Which team will have more turnovers?

**Algorithms Used:** Decision Tree and XGBoost

---

# Dataset

## Data Source
- **Source:** Turkish Basketball Federation (TBF)
- **League:** ING Basketball Super League
- **Season:** 2018-2019
- **Total Actions:** 462,035 rows
- **Matches Analyzed:** 233 games

## Data Structure
The dataset is in play-by-play (action-based) format:
- Shot attempts (2PT, 3PT)
- Turnovers
- Rebounds
- Assists
- Blocks
- Fouls
- Free throws

---

# Methodology

## 1. Data Preprocessing
- First half data filtered (Period 1 and 2)
- Team-based statistics calculated for each match
- 76,491 first half actions processed

## 2. Feature Engineering

### First Half Features (67 Total Features)

#### Raw Features (20 features)
| Category | Features |
|----------|----------|
| Shot Attempts | 2PT attempts, 3PT attempts |
| Ball Control | Turnovers, Steals |
| Rebounds | Offensive, Defensive rebounds |
| Playmaking | Assists |
| Defense | Blocks, Fouls |
| Score | First half score |
| Other | Free throws |

#### Derived Features (7 features)
- Score difference (score_diff_1h)
- 2PT attempt difference (2pt_diff_1h)
- Turnover difference (turnover_diff_1h)
- Rebound difference (rebound_diff_1h)
- Assist difference (assist_diff_1h)
- Total shots per team

#### **Custom Metrics (3 metrics - 9 features)**

**These custom metrics were specifically created for this project to capture advanced basketball analytics:**

##### 1. Paint Dominance (Spatial Scoring Efficiency)
- **Formula:** `paint_points / total_first_half_score`
- **Rationale:** Measures the percentage of scoring from high-percentage shots near the basket
- **Features:** team1_paint_dominance, team2_paint_dominance, paint_dominance_diff

##### 2. Defensive Pressure (Defensive Effectiveness)
- **Formula:** `(steals + blocks + opponent_turnovers) / opponent_possessions`
- **Rationale:** Quantifies how effectively a team disrupts opponent's offense
- **Features:** team1_def_pressure, team2_def_pressure, def_pressure_diff

##### 3. Lead Share & Max Scoring Run (Momentum)
- **Formula:** `actions_with_lead / total_actions` and longest scoring streak
- **Rationale:** Captures game control and psychological momentum
- **Features:** team1_lead_share, team2_lead_share, lead_share_diff, team1_max_run, team2_max_run, max_run_diff

#### Historical Form Features (~31 features)
- Recent wins/losses (last 3, 5, 10 games)
- Average scoring differential
- Streak indicators

## 3. Target Variables

| Target | Description | Distribution |
|--------|-------------|--------------|
| Winner | Team that wins the game | Team1: 62.2% |
| 2PT Leader | Team with more 2PT attempts | Team1: 55.8% |
| TO Leader | Team with more turnovers | Team1: 54.5% |

## 4. Model Training

### Cross-Validation Strategy
- **5-Fold Stratified Cross-Validation** for robust performance estimation
- **Pipeline-based RFE** (Recursive Feature Elimination) to prevent data leakage
  - RFE applied INSIDE each CV fold (not on full dataset)
  - Top 25 features selected per task
  - Ensures unbiased performance estimates

### Hyperparameter Tuning
- **GridSearchCV** with nested cross-validation
- Decision Tree: 160 parameter combinations tested
- XGBoost: 216 parameter combinations tested

### Decision Tree Parameter Grid
```
max_depth: [3, 5, 7, 10, None]
min_samples_split: [2, 5, 10, 20]
min_samples_leaf: [1, 2, 5, 10]
criterion: ['gini', 'entropy']
```

### XGBoost Parameter Grid
```
n_estimators: [50, 100, 200]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.1, 0.2]
subsample: [0.8, 1.0]
colsample_bytree: [0.8, 1.0]
```

### Evaluation Metrics
- **Accuracy** - Overall correctness
- **F1 Score** - Balance between precision and recall
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **ROC-AUC** - Discrimination ability

---

# Results

## Model Performance

### Game Winner Prediction

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Decision Tree | 70.21% | 0.7586 |
| XGBoost | 70.21% | 0.7742 |

**Result:** Both models have equal accuracy, XGBoost has better F1 score

### Two-Point Trials Leader (2PT Leader)

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Decision Tree | **74.47%** | 0.7600 |
| XGBoost | 70.21% | 0.7407 |

**Result:** Decision Tree performs better (74.47%)

### Turnover Leader

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Decision Tree | 61.70% | 0.6087 |
| XGBoost | **65.96%** | 0.6522 |

**Result:** XGBoost performs better (65.96%)

---

# Best Models Summary

| Task | Best Model | Accuracy |
|------|------------|----------|
| Game Winner | Decision Tree | **70.2%** |
| 2PT Leader | Decision Tree | **74.5%** |
| Turnover Leader | XGBoost | **66.0%** |

---

# Key Findings

## 1. Most Important Features

### For Game Winner:
- **score_diff_1h** (First half score difference) - Strongest predictor
- team1_score_1h, team2_score_1h
- Total shot attempts

### For 2PT Leader:
- **2pt_diff_1h** (First half 2PT difference)
- team1_2pt_1h, team2_2pt_1h
- Total shot attempts

### For Turnover Leader:
- **turnover_diff_1h** (First half turnover difference)
- Steal counts
- Foul counts

## 2. Model Comparison

- **Decision Tree:** More interpretable, better in 2 tasks
- **XGBoost:** Better at turnover prediction
- Both models significantly outperform baseline (50%)

## 3. Prediction Difficulty

| Task | Difficulty | Explanation |
|------|------------|-------------|
| 2PT Leader | Easy | First half trend continues |
| Game Winner | Medium | Score difference is strong signal |
| Turnover Leader | Hard | Game dynamics are variable |

---

# Confusion Matrix Analysis

## Decision Tree - Game Winner
- True Positive (Team1 correct): High
- False Negative: Low
- Model predicts Team1 wins well

## XGBoost - Turnover Leader
- Balanced prediction performance
- Similar success rate for both classes

---

# Conclusions and Recommendations

## Conclusions

1. **First half data** is effective in predicting game outcomes (70%+ accuracy)
2. **2PT attempt prediction** is the most successful task (74.5%)
3. **Turnover prediction** is the hardest task - high in-game variability
4. **Decision Tree** is sufficient and interpretable for most tasks

## Recommendations

1. **More features:** Shooting percentages, player-based statistics
2. **Cross-validation:** More reliable performance estimation
3. **Hyperparameter tuning:** Optimization with grid search
4. **Ensemble methods:** Improvement through model combination

---

# Technical Details

## Technologies Used
- **Python 3.11**
- **pandas** - Data processing
- **scikit-learn** - Decision Tree
- **XGBoost** - Gradient Boosting
- **matplotlib/seaborn** - Visualization

## Files
- `Basketball_Prediction_Project.ipynb` - Main notebook
- `model_comparison.png` - Model comparison chart
- `confusion_matrices.png` - Confusion matrices
- `dt_winner_tree.png` - Decision Tree visualization

---

# Thank You

**STAT438 Project 2**
**Group 9**

Questions are welcome!
