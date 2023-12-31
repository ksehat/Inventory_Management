import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from functions import api_token_handler

def inventory_management_predictor():
    token = api_token_handler()
    df_info = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8083/api/Inventory/GetInventoryManagementInformation',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)["getInventoryManagementInformationItemsResponseViewModels"])

    df_info['yearMonth'] = pd.to_datetime(df_info['yearMonth'])
    df_info.sort_values('yearMonth', inplace=True)
    df_info['year'] = df_info['yearMonth'].dt.year
    df_info['month'] = df_info['yearMonth'].dt.month
    df_info.drop(columns=['yearMonth'], inplace=True)

    df_info = df_info[df_info['fkPartNumber'].map(df_info['fkPartNumber'].value_counts()) > 1]

    x = df_info[['fkPartNumber', 'registerCount', 'year', 'month']]
    y = df_info['partNumberCount']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=df_info['fkPartNumber'])

    model = GradientBoostingRegressor(max_depth=4, n_estimators=200, n_iter_no_change=10000, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r2}")

    results_list = []
    fkPartNumbers = x_test['fkPartNumber'].unique()
    for fkPartNumber in fkPartNumbers:
        x_test_fkPartNumber = x_test[x_test['fkPartNumber'] == fkPartNumber]
        y_test_fkPartNumber = y_test[x_test['fkPartNumber'] == fkPartNumber]
        y_pred_fkPartNumber = model.predict(x_test_fkPartNumber)
        mae_fkPartNumber = mean_absolute_error(y_test_fkPartNumber, y_pred_fkPartNumber)
        r2_fkPartNumber = r2_score(y_test_fkPartNumber, y_pred_fkPartNumber)
        results_list.append({'fkPartNumber': fkPartNumber, 'MAE': mae_fkPartNumber, 'R2': r2_fkPartNumber})

    results = pd.DataFrame(results_list)
    return results


def show_results(results):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    sns.barplot(x='fkPartNumber', y='MAE', data=results, ax=axs[0])
    axs[0].set_title('Mean Absolute Error for each fkPartNumber')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    sns.barplot(x='fkPartNumber', y='R2', data=results, ax=axs[1])
    axs[1].set_title('R2 Score for each fkPartNumber')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.show()


results = inventory_management_predictor()
print(results)



