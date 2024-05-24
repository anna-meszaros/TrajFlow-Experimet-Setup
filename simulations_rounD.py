from experiment import Experiment

# Draw latex figure
Experiment_name = 'rounD'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# Select the datasets
Data_sets = [{'scenario': 'RounD_round_about', 'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.2, 'num_timesteps_in': (15, 15), 'num_timesteps_out': (25, 25)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Cross_split', 'repetition': [0,1,2,3,4], 'test_part': 0.2}]

# Select the models to be trained
Models = []

Models.append({'model': 'trajflow_meszaros',
			'kwargs': {'fut_enc_sz': 20, 
			'scene_encoding_size': 64,
			'obs_encoding_size': 64,
			'beta_noise': 0.0,
			'gamma_noise': 0.0,
			'alpha': 10,
			'decoder_type': 'none',
			'pos_loss': True,
			'seed': 42,
			'fut_ae_lr_decay': 1.0,
			'flow_lr': 1e-3,
			'flow_lr_decay': 0.98}})


Models.append({'model': 'flomo_schoeller',
					'kwargs': {'obs_encoding_size': 16,
						'scene_encoding_size': 64,
						'beta_noise': 0.2,
						'gamma_noise': 0.02,
						'alpha': 10,
						's_min': 0.3,
						's_max': 1.7,
						'sigma': 0.5,
						'seed': 42}})
						
Models.append({'model': 'flomo_schoeller',
					'kwargs': {'obs_encoding_size': 64,
						'scene_encoding_size': 64,
						'beta_noise': 0.0,
						'gamma_noise': 0.0,
						'alpha': 10,
                        			'scale_NF': False,
						'lr_decay': 0.98,
						'seed': 42}})

Models.append({'model': 'trajectron_salzmann_old',
		'kwargs': {'seed': 42}})
Models.append({'model': 'mid_gu'})             
Models.append({'model': 'pecnet_mangalam'})

# Select the metrics to be used
Metrics = ['minADE20_indep', 'minADE20_indepExtrap', 'minFDE20_indep', 'KDE_NLL_indep', 'KDE_NLL_joint']


new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 100

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_time = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = True

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = True

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = True

# Use all available agents for predictions
agents_to_predict = 'predefined'

# Determine if allready existing results should be overwritten, or if not, be used instead
overwrite_results = 'metric'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = False

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_time, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, evaluate_on_train_set)


#%% Run experiment
# new_experiment.run() 

# new_experiment.plot_paths(load_all=True, only_show_pred_agents=True)

new_experiment.plot_paths(load_all = True, plot_similar_futures = True, plot_train = False,
                          only_show_pred_agents = False, likelihood_visualization = True, plot_only_lines = False)


# Load results
Results = new_experiment.load_results(plot_if_possible = False)
import numpy as np
import pandas as pd
import scipy as sp
np.set_printoptions(precision=3, suppress=True, linewidth=300)
R = Results.squeeze()
print(R)

Take mean over seeds
M = pd.DataFrame([pd.Series(np.array([m['model']] + list(m['kwargs'].values())), index =(['model'] + list(m['kwargs'].keys()))) for m in Models])
M = M[['model', 'decoder_type', 'beta_noise']]

# Calculate statistic significance (using paired t-test due to correlation between splits)
T, P = sp.stats.ttest_1samp(R[:,np.newaxis] - R[:,:,np.newaxis], 0, axis=0, 
                            nan_policy='omit', alternative='greater')
print('Statistic significance')
print(T.shape)

 
# Take mean over splits
RA = np.nanmean(R, axis=0)
Std = np.nanstd(R, axis=0)

useless = np.isnan(RA).all(1)
RA = RA[~useless]
M  = M.iloc[~useless]
Std = Std[~useless]

print('Correlation between metrics')
corr = np.corrcoef(RA, rowvar=False)
print(corr)
M_std = M.copy()
M['minADE20']       = RA[:, 0]
M['minADE20Extrap'] = RA[:, 1]
M['minFDE20']       = RA[:, 2]
M['KDE_NLL_indep']  = RA[:, 3]
M['KDE_NLL_joint']  = RA[:, 4]

M.to_excel("RounD.xlsx")

M_std['minADE20'] = Std[:, 0]
M_std['minADE20Extrap'] = Std[:, 1]
M_std['minFDE20'] = Std[:, 2]
M_std['KDE_NLL_indep'] = Std[:, 3]
M_std['KDE_NLL_joint'] = Std[:, 4]
 
M_std.to_excel("RounD_std.xlsx")
