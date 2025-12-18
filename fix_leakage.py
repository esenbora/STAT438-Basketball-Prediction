#!/usr/bin/env python3
"""
Script to fix data leakage in b1.ipynb by:
1. Adding Pipeline import
2. Refactoring tune_and_evaluate() to use Pipeline with RFE
3. Deleting get_selected_X() function
4. Updating Task 1-3 calls to use new tune_and_evaluate()
"""

import json
import sys

def main():
    # Load notebook
    with open('b1.ipynb', 'r') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # ===  STEP 1: Find and modify the cell with get_selected_X ===
    for idx, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))

        # Cell 34: Has all_results, best_models, RFE import, get_selected_X
        if 'def get_selected_X' in source:
            print(f"\n[STEP 1] Found get_selected_X in cell {idx}")

            # New source: Keep all_results and best_models, add Pipeline import, add NEW tune_and_evaluate
            new_source = """
# Store all results for comparison
all_results = {
    'Task': [],
    'Model': [],
    'Accuracy': [],
    'Accuracy_Std': [],
    'F1': [],
    'F1_Std': [],
    'Precision': [],
    'Precision_Std': [],
    'Recall': [],
    'Recall_Std': [],
    'ROC_AUC': [],
    'ROC_AUC_Std': [],
    'Best_Params': []
}

# Store best models
best_models = {}


# === PIPELINE SETUP FOR RFE (FIX DATA LEAKAGE) ===
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def tune_and_evaluate(X, y, task_name, use_rfe=True, n_features=25):
    \"\"\"
    Perform hyperparameter tuning with optional RFE in a Pipeline.
    RFE is applied INSIDE CV folds to prevent data leakage.

    Parameters:
    - X: Full feature matrix (all 67 features)
    - y: Target variable
    - task_name: Name of the prediction task
    - use_rfe: Whether to use RFE feature selection
    - n_features: Number of features to select (if use_rfe=True)

    Returns:
    - Dictionary with 'dt' and 'xgb' results
    \"\"\"
    print("="*70)
    print(f"TASK: {task_name}")
    print("="*70)

    task_results = {}

    # === DECISION TREE with RFE Pipeline ===
    print("\\n>>> Decision Tree - Hyperparameter Tuning <<<")

    if use_rfe:
        # Create pipeline: RFE → Classifier
        dt_pipeline = Pipeline([
            ('rfe', RFE(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                n_features_to_select=n_features,
                step=5
            )),
            ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
        ])

        # Parameter grid with 'classifier__' prefix
        dt_param_grid_pipeline = {
            'classifier__max_depth': [3, 5, 7, 10, None],
            'classifier__min_samples_split': [2, 5, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        }
    else:
        # No RFE - use all features
        dt_pipeline = DecisionTreeClassifier(random_state=RANDOM_STATE)
        dt_param_grid_pipeline = dt_param_grid

    dt_grid = GridSearchCV(
        dt_pipeline,
        dt_param_grid_pipeline,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        refit=True
    )
    dt_grid.fit(X, y)  # X is FULL feature set, RFE happens inside CV

    print(f"Best Parameters: {dt_grid.best_params_}")
    print(f"Best CV Accuracy: {dt_grid.best_score_:.4f}")

    # Evaluate with all metrics
    dt_cv_results = evaluate_with_cv(
        dt_grid.best_estimator_, X, y,
        "DECISION TREE (Tuned + RFE)" if use_rfe else "DECISION TREE (Tuned)",
        task_name
    )

    # Store results
    all_results['Task'].append(task_name)
    all_results['Model'].append('Decision Tree')
    all_results['Accuracy'].append(dt_cv_results['accuracy']['mean'])
    all_results['Accuracy_Std'].append(dt_cv_results['accuracy']['std'])
    all_results['F1'].append(dt_cv_results['f1']['mean'])
    all_results['F1_Std'].append(dt_cv_results['f1']['std'])
    all_results['Precision'].append(dt_cv_results['precision']['mean'])
    all_results['Precision_Std'].append(dt_cv_results['precision']['std'])
    all_results['Recall'].append(dt_cv_results['recall']['mean'])
    all_results['Recall_Std'].append(dt_cv_results['recall']['std'])
    all_results['ROC_AUC'].append(dt_cv_results['roc_auc']['mean'])
    all_results['ROC_AUC_Std'].append(dt_cv_results['roc_auc']['std'])
    all_results['Best_Params'].append(str(dt_grid.best_params_))

    task_results['dt'] = {
        'model': dt_grid.best_estimator_,
        'params': dt_grid.best_params_,
        'cv_results': dt_cv_results
    }

    # === XGBOOST with RFE Pipeline ===
    print("\\n>>> XGBoost - Hyperparameter Tuning <<<")

    if use_rfe:
        xgb_pipeline = Pipeline([
            ('rfe', RFE(
                estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False),
                n_features_to_select=n_features,
                step=5
            )),
            ('classifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False))
        ])

        xgb_param_grid_pipeline = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0]
        }
    else:
        xgb_pipeline = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
        xgb_param_grid_pipeline = xgb_param_grid

    xgb_grid = GridSearchCV(
        xgb_pipeline,
        xgb_param_grid_pipeline,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        refit=True
    )
    xgb_grid.fit(X, y)

    print(f"Best Parameters: {xgb_grid.best_params_}")
    print(f"Best CV Accuracy: {xgb_grid.best_score_:.4f}")

    xgb_cv_results = evaluate_with_cv(
        xgb_grid.best_estimator_, X, y,
        "XGBOOST (Tuned + RFE)" if use_rfe else "XGBOOST (Tuned)",
        task_name
    )

    # Store results
    all_results['Task'].append(task_name)
    all_results['Model'].append('XGBoost')
    all_results['Accuracy'].append(xgb_cv_results['accuracy']['mean'])
    all_results['Accuracy_Std'].append(xgb_cv_results['accuracy']['std'])
    all_results['F1'].append(xgb_cv_results['f1']['mean'])
    all_results['F1_Std'].append(xgb_cv_results['f1']['std'])
    all_results['Precision'].append(xgb_cv_results['precision']['mean'])
    all_results['Precision_Std'].append(xgb_cv_results['precision']['std'])
    all_results['Recall'].append(xgb_cv_results['recall']['mean'])
    all_results['Recall_Std'].append(xgb_cv_results['recall']['std'])
    all_results['ROC_AUC'].append(xgb_cv_results['roc_auc']['mean'])
    all_results['ROC_AUC_Std'].append(xgb_cv_results['roc_auc']['std'])
    all_results['Best_Params'].append(str(xgb_grid.best_params_))

    task_results['xgb'] = {
        'model': xgb_grid.best_estimator_,
        'params': xgb_grid.best_params_,
        'cv_results': xgb_cv_results
    }

    return task_results

print("Tuning and evaluation function ready (with RFE Pipeline)!")
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            print(f"  ✓ Replaced cell {idx} with Pipeline-based tune_and_evaluate()")

    # === STEP 2: Find and update Task 1 call ===
    for idx, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))

        if 'X_winner, cols_winner = get_selected_X' in source and 'Game Winner' in source:
            print(f"\n[STEP 2] Found Task 1 (Game Winner) in cell {idx}")

            new_source = """# Task 1: Game Winner Prediction
y_winner = final_df['winner']
best_models['winner'] = tune_and_evaluate(
    X,  # FULL feature set (67 features)
    y_winner,
    "Game Winner",
    use_rfe=True,  # Enable RFE in Pipeline
    n_features=25
)
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            print(f"  ✓ Updated Task 1 cell {idx}")

    # === STEP 3: Find and update Task 2 call ===
    for idx, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))

        if 'X_2pt, cols_2pt = get_selected_X' in source and 'Two-Point Leader' in source:
            print(f"\n[STEP 3] Found Task 2 (Two-Point Leader) in cell {idx}")

            new_source = """# Task 2: Two-Point Leader Prediction
y_2pt = final_df['twopoint_leader']
best_models['twopoint'] = tune_and_evaluate(
    X,
    y_2pt,
    "Two-Point Leader",
    use_rfe=True,
    n_features=25
)
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            print(f"  ✓ Updated Task 2 cell {idx}")

    # === STEP 4: Find and update Task 3 call ===
    for idx, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))

        if 'X_to, cols_to = get_selected_X' in source and 'Turnover Leader' in source:
            print(f"\n[STEP 4] Found Task 3 (Turnover Leader) in cell {idx}")

            new_source = """# Task 3: Turnover Leader Prediction
y_to = final_df['turnover_leader']
best_models['turnover'] = tune_and_evaluate(
    X,
    y_to,
    "Turnover Leader",
    use_rfe=True,
    n_features=25
)
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            print(f"  ✓ Updated Task 3 cell {idx}")

    # Save modified notebook
    with open('b1.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print("\n" + "="*70)
    print("✓ Successfully fixed data leakage in b1.ipynb!")
    print("="*70)
    print("\nChanges made:")
    print("1. Added Pipeline import")
    print("2. Replaced tune_and_evaluate() with RFE Pipeline version")
    print("3. Deleted get_selected_X() function")
    print("4. Updated Task 1-3 calls to use full feature set")
    print("\nNext steps:")
    print("- Run the notebook to verify no errors")
    print("- Expect ~2-5% accuracy drop (normal for fixing leakage)")
    print("- Add custom metrics documentation")

if __name__ == '__main__':
    main()
