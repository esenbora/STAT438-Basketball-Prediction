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

### First Half Features (27 Features)

| Category | Features |
|----------|----------|
| Shot Attempts | 2PT attempts, 3PT attempts |
| Ball Control | Turnovers, Steals |
| Rebounds | Offensive, Defensive rebounds |
| Playmaking | Assists |
| Defense | Blocks, Fouls |
| Score | First half score |

### Derived Features
- Score difference (score_diff)
- 2PT attempt difference (2pt_diff)
- Turnover difference (turnover_diff)
- Rebound difference (rebound_diff)
- Assist difference (assist_diff)

## 3. Target Variables

| Target | Description | Distribution |
|--------|-------------|--------------|
| Winner | Team that wins the game | Team1: 62.2% |
| 2PT Leader | Team with more 2PT attempts | Team1: 55.8% |
| TO Leader | Team with more turnovers | Team1: 54.5% |

## 4. Model Training

### Train/Test Split
- **Training:** 186 matches (80%)
- **Test:** 47 matches (20%)
- **Stratified sampling** for balanced distribution

### Decision Tree Parameters
```
max_depth = 5
min_samples_split = 10
min_samples_leaf = 5
```

### XGBoost Parameters
```
n_estimators = 100
max_depth = 5
learning_rate = 0.1
```

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
