from experiment import Experiment

# Draw latex figure
Experiment_name = 'FP_aug'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# Select the datasets
Data_sets = [{'scenario': 'Forking_Paths_augmented', 'max_num_agents': None, 't0_type': 'start', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.4, 'num_timesteps_in': (8, 8), 'num_timesteps_out': (12, 12)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Cross_split', 'repetition': [0,1,2,3,4], 'test_part': 0.2}]

# Select the models to be trained
seed = 0
Models = []
for seed in [0, 1, 2, 3, 4]:
     # for decoder_type in ['none', 'fac_20']:
     for decoder_type in ['none']:
          Models.append({'model': 'trajflow_meszaros',
                                   'kwargs': {'fut_enc_sz': 20, 
                                             'scene_encoding_size': 4,
                                             'obs_encoding_size': 64,
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
                                        'beta_noise': 0.2,
                                        'gamma_noise': 0.02,
                                        'alpha': 10,
                                        's_min': 0.3,
                                        's_max': 1.7,
                                        'sigma': 0.5,
                                        'seed': seed}})
                              
     Models.append({'model': 'flomo_schoeller',
                              'kwargs': {'obs_encoding_size': 64,
                                         'scene_encoding_size': 4,
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
Metrics = ['minADE20_joint', 'minFDE20_joint', 'KDE_NLL_joint', 'JSD_traj_joint']

new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 50

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


#%% Run experiment
# new_experiment.run() 

# Load results
Results = new_experiment.load_results(plot_if_possible = False)

import numpy as np
import pandas as pd
import scipy as sp
np.set_printoptions(precision=3, suppress=True, linewidth=300)
R = Results.squeeze()
# R.shape = 5 splits x (5 seeds * 41 models (36 TF, 4 FM, 1 T++)) x 8 metrics

# Take mean over seeds
M = pd.DataFrame([pd.Series(np.array([m['model']] + list(m['kwargs'].values())), index =(['model'] + list(m['kwargs'].keys()))) for m in Models])
M = M.iloc[(M.seed == '0').to_numpy()]
M = M[['model', 'decoder_type', 'beta_noise']]

# Take mean over seeds
R  = R.reshape(len(new_experiment.Splitters), 5, 6, len(Metrics))
RA = R.reshape(-1, 6, len(Metrics))

# Calculate statistic significance (using paired t-test due to correlation between splits)
T, P = sp.stats.ttest_1samp(RA[:,np.newaxis] - RA[:,:,np.newaxis], 0, axis=0, 
                            nan_policy='omit', alternative='greater')
print('Statistic significance')
print(T.shape)


# Take mean over seeds
SA = np.nanstd(RA, axis=0)
RA = np.nanmean(RA, axis=0)

useless = np.isnan(RA).all(1)
RA = RA[~useless]
SA = SA[~useless]
M  = M.iloc[~useless]


print('Correlation between metrics')
corr = np.corrcoef(RA, rowvar=False)
print(corr)
M['minADE20']      = RA[:, 0]
M['std(minADE20)'] = SA[:, 0]
M['minFDE20']      = RA[:, 1]
M['std(minFDE20)'] = SA[:, 1]
M['KDE_NLL']       = RA[:, 2]
M['std(KDE_NLL)']  = SA[:, 2]
M['JSD_traj']      = RA[:, 3]
M['std(JSD_traj)'] = SA[:, 3]

# Get number of samples underlying the results
num_samples = np.isfinite(R).sum(1).max(-1).T
M['num_samples'] = num_samples[~useless].tolist()

M.to_excel("Forking_Paths.xlsx")
