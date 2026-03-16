import papermill as pm

notebooks = [
    "01_eda",
    "02_preprocess_feature",
    "03_mining_clustering",
    "04_modeling",
    "05_evaluation_report"
]

for nb in notebooks:
    pm.execute_notebook(
        f"notebooks/{nb}.ipynb",
        f"outputs/{nb}_output.ipynb"
    )