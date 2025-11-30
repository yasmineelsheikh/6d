import pandas as pd
from ares.databases.structured_database import RolloutSQLModel, setup_database, db_to_df

engine = setup_database(RolloutSQLModel, path="sqlite:///data/robot_data.db")
df = db_to_df(engine)
print("Total rollouts:", len(df))
print("Datasets:", df.dataset_formalname.unique())
print("Sample rollout:")
print(df.iloc[-1][['dataset_formalname', 'description_estimate', 'length']])
