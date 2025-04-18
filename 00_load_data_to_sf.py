# %%
# 1. Extract data from the package
# from pycaret.datasets import get_data
# # all_datasets = get_data('index')
# df_insurance = get_data('insurance', verbose=True)

import pandas as pd
url = 'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv'
df_insurance = pd.read_csv(url)


# %%
from dotenv import load_dotenv
import os
from snowflake.snowpark import Session
load_dotenv()

# %%
# 2. Load data to snowflake
connection_parameters = {
        "account"   : os.environ.get("SF_ACCOUNT"),
        "user"      : os.environ.get("USERNAME"),
        "role"      : os.environ.get("SF_ROLE"),
        "database"  : os.environ.get("SF_DATABASE"),
        "schema"    : os.environ.get("SF_SCHEMA"),
        "warehouse" : os.environ.get("SF_WAREHOUSE"),
        "authenticator" : "externalbrowser",
    }

session = Session.builder.configs(connection_parameters).create()

# %%
df_sf = session.create_dataframe(df_insurance) # 1338
df_sf.write.mode("overwrite").save_as_table("INSURANCE_DATA_ALL")

data_train = df_insurance.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = df_insurance.drop(data_train.index).reset_index(drop=True)

data_train_sf = session.create_dataframe(data_train) # 1204
data_train_sf.write.mode("overwrite").save_as_table("INSURANCE_DATA_HISTORICAL")

data_unseen_sf = session.create_dataframe(data_unseen) # 134
data_unseen_sf.write.mode("overwrite").save_as_table("INSURANCE_DATA_NEW")
