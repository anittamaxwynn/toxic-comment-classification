from data.download import download_competition_dataset

competition_identifier = input(
    "Enter the Kaggle competition identifier (e.g., 'titanic'): "
)
try:
    download_competition_dataset(competition_identifier)
except Exception as e:
    print(f"An error occurred: {e}")
