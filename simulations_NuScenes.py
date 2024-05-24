from experiment import Experiment

# Draw latex figure
Experiment_name = 'NuScenes'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# Select the datasets

Data_sets = [{'scenario': 'NuScenes_interactive', 'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.5, 'num_timesteps_in': (4, 4), 'num_timesteps_out': (12, 12)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Predefined_split'}]

# Select the models to be trained
Models = []
for seed in range(5):
	for decoder_type in ['none']:
				Models.append({'model': 'trajflow_meszaros',
				'kwargs': {'fut_enc_sz': 20, 
				'obs_encoding_size': 64,
				'scene_encoding_size': 64,
				'beta_noise': 0.0,
				'gamma_noise': 0.0,
				'alpha': 10,
				'decoder_type': decoder_type,
				'pos_loss': True,
				'seed': seed,
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
							'seed': seed}})
							
	Models.append({'model': 'flomo_schoeller',
						'kwargs': {'obs_encoding_size': 64,
							'scene_encoding_size': 64,
							'beta_noise': 0.0,
							'gamma_noise': 0.0,
							'alpha': 10,
							's_min': 0.8,
							's_max': 1.2,
							'sigma': 0.5, 
                        				'scale_NF': False,
							'lr_decay': 0.98,
							'seed': seed}})

							

	Models.append({'model': 'trajectron_salzmann_old',
			'kwargs': {'seed': seed}})
	Models.append({'model': 'mid_gu',
			'kwargs': {'seed': seed}})             
	Models.append({'model': 'pecnet_mangalam',
			'kwargs': {'seed': seed}})

# Select the metrics to be used
Metrics = ['minADE20_indep', 'minFDE20_indep', 'KDE_NLL_indep', 'KDE_NLL_joint']

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

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = False

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_time, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, evaluate_on_train_set)

new_experiment.plot_paths(load_all = True, plot_similar_futures = True, plot_train = False,
                          only_show_pred_agents = False, likelihood_visualization = True, plot_only_lines = False)




#%% Run experiment
# new_experiment.run() 

# # Load results
# Results = new_experiment.load_results(plot_if_possible = False)
# import numpy as np
# import pandas as pd
# import scipy as sp
# np.set_printoptions(precision=3, suppress=True, linewidth=300)
# R = Results.squeeze()

# # Take mean over seeds
# M = pd.DataFrame([pd.Series(np.array([m['model']] + list(m['kwargs'].values())), index =(['model'] + list(m['kwargs'].keys()))) for m in Models])
# M = M.iloc[(M.seed == '0').to_numpy()]
# M = M[['model', 'decoder_type', 'beta_noise']]


# # Separate by seeds
# R = R.reshape(5, -1, len(Metrics))

# # Calculate statistic significance (using paired t-test due to correlation between splits)
# T, P = sp.stats.ttest_1samp(R[:,np.newaxis] - R[:,:,np.newaxis], 0, axis=0, 
#                             nan_policy='omit', alternative='greater')
# print('Statistic significance')
# print(T.shape)


# # Take mean over seeds
# Sa = np.nanstd(R, axis=0)
# Ra = np.nanmean(R, axis=0)
 
# useless = np.isnan(Ra).all(1)
# Ra = Ra[~useless]
# M = M.iloc[~useless]
# Sa = Sa[~useless]
 

# print('Correlation between metrics')
# corr = np.corrcoef(Ra, rowvar=False)
# print(corr)
# M['minADE20']           = Ra[:, 0]
# M['std(minADE20)']      = Sa[:,0]
# M['minFDE20']           = Ra[:, 1]
# M['std(minFDE20)']      = Sa[:,1]
# M['KDE_NLL_indep']      = Ra[:, 2]
# M['std(KDE_NLL_indep)'] = Sa[:,2]
# M['KDE_NLL_joint']      = Ra[:, 3]
# M['std(KDE_NLL_joint)'] = Sa[:,3]

# num_samples = np.isfinite(R).sum(0).max(-1).T
# M['num_samples'] = num_samples[~useless].tolist()

# M.to_excel("NuScenes.xlsx")
