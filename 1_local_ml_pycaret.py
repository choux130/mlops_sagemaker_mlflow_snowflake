# %%
# 1. Load the dataset
from pycaret.datasets import get_data
# all_datasets = get_data('index')
df_insurance = get_data('insurance')

# %%
# 2. Check the dataset
print('Data field types:')
print(df_insurance.dtypes)

# %%
# 3. Data split
data_train = df_insurance.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = df_insurance.drop(data_train.index).reset_index(drop=True)

print('Total Data: ' + str(df_insurance.shape))
print('Data for Modeling: ' + str(data_train.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# %%
# 3.  import pycaret regression and init setup
from pycaret.regression import RegressionExperiment
reg_exp = RegressionExperiment()
reg_exp.setup(
    data_train, target = 'charges', 
    session_id = 123, train_size = 0.7, data_split_shuffle=True
)

reg_exp.get_config()

# %%
# 4. Compare multiple models (using default hyperparameters) with k-fold CV and then select the best one
reg_exp.models() # all available models

best_model_from_compare = reg_exp.compare_models(fold=3, sort = 'R2')
print('Best Model:')
print(best_model_from_compare)

best_models_from_compare = reg_exp.compare_models(
    fold=3, sort = 'R2', include=['gbr', 'lr', 'dt'],
    n_select=2
)
print('Best Models:')
print(best_models_from_compare)

reg_exp.pull()

# %%
# 5. Train a model with default hyperparameters
model_gbr = reg_exp.create_model('gbr', fold=10, return_train_score=True)
print('Model with default hyperparameters:')
print(model_gbr)

reg_exp.pull()

# 6. Hyperparameter Tuning the model (random grid search)
tuned_model_gbr, tuner = reg_exp.tune_model(model_gbr, return_tuner=True)
print('Model with tuned hyperparameters:')
print(tuned_model_gbr)
print(tuner)


# %%
# 5. Analyze the model
reg_exp.plot_model(best_model_from_compare, plot = 'residuals')
reg_exp.plot_model(best_model_from_compare, plot = 'error')
reg_exp.plot_model(best_model_from_compare, plot = 'feature')
reg_exp.evaluate_model(best_model_from_compare)

# %%
# 6. Predict on test dataset and unseen data
pred_test_data = reg_exp.predict_model(best_model_from_compare) # (created during the setup function)
pred_test_data

pred_data_unseen = reg_exp.predict_model(best_model_from_compare, data=data_unseen)
pred_data_unseen


# %%
# 6. Save model
best_model_pipeline = reg_exp.finalize_model(best_model_from_compare)
type(best_model_pipeline) # pycaret.internal.pipeline.Pipeline

# save both pipeline and the model
type(best_model_from_compare) # sklearn.ensemble._gb.GradientBoostingRegressor
reg_exp.save_model(best_model_from_compare, "model_insurance_best_model")

# 7. Load model
from pycaret.regression import RegressionExperiment
reg_exp_load = RegressionExperiment()
loaded_best_pipeline = reg_exp_load.load_model('model_insurance_best_model')
loaded_best_pipeline # pycaret.internal.pipeline.Pipeline