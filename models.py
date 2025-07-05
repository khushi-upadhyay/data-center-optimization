# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from preprocessing import load_and_prepare

# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# try:
#     from xgboost import XGBRegressor
#     xgb_available = True
# except ImportError:
#     print("XGBoost not installed, skipping this model.")
#     xgb_available = False


# def save_feature_importance(model, feature_names, model_name):
#     plt.figure(figsize=(8, 6))
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#     elif hasattr(model, 'coef_'):
#         importances = model.coef_
#     else:
#         print(f"No feature importance or coefficients available for {model_name}")
#         return

#     sns.barplot(x=importances, y=feature_names)
#     plt.title(f"{model_name} Feature Importance / Coefficients")
#     plt.tight_layout()
#     plt.savefig(f'results/{model_name}_feature_importance.png')
#     plt.close()


# def main():
#     os.makedirs('results', exist_ok=True)

#     df_min, X, y, _ = load_and_prepare()

#     models = {
#         'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
#         'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#         'LinearRegression': LinearRegression(),
#         'SVR': SVR(),
#     }

#     if xgb_available:
#         models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

#     results = []

#     for name, model in models.items():
#         print(f"Training {name}...")
#         model.fit(X, y)
#         y_pred = model.predict(X)

#         r2 = r2_score(y, y_pred)
#         rmse = mean_squared_error(y, y_pred) ** 0.5
#         mae = mean_absolute_error(y, y_pred)

#         results.append({
#             'Model': name,
#             'R2': r2,
#             'RMSE': rmse,
#             'MAE': mae
#         })

#         save_feature_importance(model, X.columns, name)

#     metrics_df = pd.DataFrame(results)
#     metrics_df.to_csv('results/model_comparison_metrics.csv', index=False)

#     print("Done! Metrics saved to 'results/model_comparison_metrics.csv'.")


# if __name__ == "__main__":
#     main()



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessing import load_and_prepare

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    print("XGBoost not installed, skipping this model.")
    xgb_available = False


def save_feature_importance(model, feature_names, model_name):
    plt.figure(figsize=(8, 6))
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_
    else:
        print(f"No feature importance or coefficients available for {model_name}")
        return

    sns.barplot(x=importances, y=feature_names)
    plt.title(f"{model_name} Feature Importance / Coefficients")
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_feature_importance.png')
    plt.close()


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df_min, X, y, _ = load_and_prepare()

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
    }

    if xgb_available:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    results = []
    best_r2 = -float("inf")
    best_model = None

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
       
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        mae = mean_absolute_error(y, y_pred)

        print(f"✅ {name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        results.append({
            'Model': name,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })

        save_feature_importance(model, X.columns, name)

        # Save best XGBoost model
        if name == 'XGBoost' and r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Save model metrics
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv('results/model_comparison_metrics.csv', index=False)
    print("\n📊 Metrics saved to 'results/model_comparison_metrics.csv'")

    # Save best XGBoost model
    if best_model:
        best_model.save_model('models/best_xgboost_model.json')
        print("💾 Best XGBoost model saved to 'models/best_xgboost_model.json'")


if __name__ == "__main__":
    main()
