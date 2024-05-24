#%%
from experiment import Experiment
import scipy as sp

# Draw latex figure
Experiment_name = 'Toy'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# Select the datasets
Data_sets = [{'scenario': 'Toy_bi_modal', 'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.25, 'num_timesteps_in': (10, 10), 'num_timesteps_out': (14, 14)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'no_split'}]

# Select the models to be trained
Models = []
for seed in range(10):
    
    Models.append({'model': 'trajflow_meszaros',
                'kwargs': {'fut_enc_sz': 20, 
                'obs_encoding_size': 16,
                'beta_noise': 0.0,
                'gamma_noise': 0.0,
                'alpha': 10,
                'decoder_type': 'none',
                'pos_loss': True,
                'seed': seed,
                'fut_ae_lr_decay': 1.0,
                'flow_lr': 1e-3,
                'flow_lr_decay': 0.98}})

    Models.append({'model': 'flomo_schoeller',
                            'kwargs': {'obs_encoding_size': 16,
                                        'beta_noise': 0.2,
                                        'gamma_noise': 0.1 * 0.2,
                                        'alpha': 10,
                                        's_min': 0.3,
                                        's_max': 1.7,
                                        'sigma': 0.5,
                                        'seed': seed}})
                            
    Models.append({'model': 'flomo_schoeller',
                            'kwargs': {'obs_encoding_size': 16,
                                        'beta_noise': 0,
                                        'gamma_noise': 0.0,
                                        'alpha': 10,
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
Metrics = [
            'minADE20_indep', 
           'minFDE20_indep',
           'KDE_NLL_indep', 
           'JSD_traj_indep'
           ]

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

# Use only predefined agents for predictions
agents_to_predict = 'all'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no' #'prediction' #'no' #'metric'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = True

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_time, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, evaluate_on_train_set)


#%% Run experiment
# new_experiment.run() 

# Load results
Results = new_experiment.load_results(plot_if_possible = True)

import numpy as np
import pandas as pd
np.set_printoptions(precision=3, suppress=True, linewidth=300)
R = Results.squeeze()
 
# Take mean over seeds
M = pd.DataFrame([pd.Series(np.array([m['model']] + list(m['kwargs'].values())), index =(['model'] + list(m['kwargs'].keys()))) for m in Models])
M = M.iloc[(M.seed == '0').to_numpy()]
M = M.drop(columns=['seed', 'obs_encoding_size', 'alpha', 's_min', 's_max', 'sigma', 'flow_epochs', 'fut_ae_lr', 'fut_ae_lr_decay', 'batch_size', 'gamma_noise', 'hs_rnn', 'n_layers_rnn', 'scene_encoding_size', 'fut_ae_epochs', 'flow_wd', 'flow_lr', 'fut_ae_wd', 'vary_input_length', 'scale_AE', 'scale_NF'])
M.flow_lr_decay.iloc[np.isnan(M.flow_lr_decay.to_numpy().astype(float))] = M.iloc[np.isnan(M.flow_lr_decay.to_numpy().astype(float))].lr_decay
M = M.drop(columns=['lr_decay'])
M = M.rename(columns={'flow_lr_decay': 'lr_decay'})
 
 
# Get correlation
useful  = np.isfinite(R).all(1)
m = M.to_numpy()[:, [0, 1, 4, 5]]
m[:,2] = m[:,2] == 'True'
m[:,1] = np.log2(m[:,1].astype(float))
 
m = np.tile(m, (10, 1))
useful &= m[:,0] == 'trajflow_meszaros'
m = m[:,1:]
m = m.astype(float)
 
r = R[useful]
m = m[useful]
 
# Correlation between parameters and metrics
np.set_printoptions(precision=3, suppress=True, linewidth=300)
corr = np.corrcoef(np.concatenate((m,r), axis = 1), rowvar=False)[:3, 3:]
print('Correlation between parameters and metrics')
print(corr)
print('')
 
 
 
# Take mean over seeds
R = R.reshape(10, -1, len(Metrics))
 
# Calculate statistic significance
T, P = sp.stats.ttest_ind(R[:,np.newaxis], R[:,:,np.newaxis], axis=0,
                          equal_var=False, nan_policy='omit', alternative='greater')
print('Statistic significance')
print(T.shape)
 
 
# Take mean over seeds
Std = np.nanstd(R, axis=0)
R = np.nanmean(R, axis=0)
 
useless = np.isnan(R).all(1)
R = R[~useless]
M = M.iloc[~useless]
Std = Std[~useless]
 
print('Correlation between metrics')
corr = np.corrcoef(R, rowvar=False)
print(corr)
M_std = M.copy()
M['minADE20'] = R[:, 0]
M['minFDE20'] = R[:, 1]
M['KDE_NLL_indep'] = R[:, 2]
M['JSD_traj_indep'] = R[:, 3]
 
M.to_excel("Toy_results_LR.xlsx")
 
 
M_std['minADE20'] = Std[:, 0]
M_std['minFDE20'] = Std[:, 1]
M_std['KDE_NLL_indep'] = Std[:, 2]
M_std['JSD_traj_indep'] = Std[:, 3]
 
M_std.to_excel("Toy_results_LR_std.xlsx")
