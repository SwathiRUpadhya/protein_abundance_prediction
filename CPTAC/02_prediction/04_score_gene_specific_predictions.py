#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

pandarallel.initialize()


# In[2]:

get_results_path = lambda fname: os.path.abspath("/home/people/18200264/local_data/output/GuanLab/CPTAC2/"+ fname)
get_input_path = lambda fname: os.path.abspath("/home/people/18200264/local_data/input_data2/"+ fname)


# In[3]:


def get_file_path(filename_parts):
    return(get_results_path('_'.join(filename_parts) + parquet_ext))


# In[4]:


file_cptac_transcriptomics = get_input_path('CPTAC_Transcriptomics_processed.parquet')
file_cptac_sample_info = get_input_path('CPTAC_sample_info.parquet')
file_cptac_proteomics = get_input_path('CPTAC_Proteomics_processed.parquet')

indiv_pred = 'im_results'
trans_pred = 'tm_results'
baseline_pred = 'baseline'
im_fallback_pred = "im_results_fallback"
tm_fallback_pred = "tm_results_fallback"
all_trans = 'all_transcriptomes'
vae100 = 'vae100'
vae500 = 'vae500'
vae1000 = 'vae1000'
corum = 'corum'
pi400 = 'pi400'
pi800 = 'pi800'
ppi400 = 'ppi400'
ppi800 = 'ppi800'
brca20 = 'BrCa2020'
pdac = 'Pdac'
ccrcc = 'ccRCC'
hnscc = 'HNSCC'
lscc = 'LSCC'
luad = 'LUAD'
parquet_ext = '.parquet'


# In[5]:


cptac_sample = pd.read_parquet(file_cptac_sample_info)[['Study']]
cptac_sample.replace(' ', '', regex=True, inplace=True)
print("Dimensions: ", cptac_sample.shape)
cptac_sample[:2]


# In[7]:


cptac_transcriptomics = pd.read_parquet(file_cptac_transcriptomics)
print(cptac_transcriptomics.shape)
cptac_transcriptomics[:2]


# In[8]:


cptac_proteomics = pd.read_parquet(file_cptac_proteomics)
print(cptac_proteomics.shape)
cptac_proteomics[:2]


# In[12]:


common_proteins = np.intersect1d(cptac_transcriptomics.index, cptac_proteomics.index)
common_samples = np.intersect1d(cptac_transcriptomics.columns, cptac_proteomics.columns)
print("Common proteins = " + str(len(common_proteins)) + " and common samples = " + str(len(common_samples)))
cptac_transcriptomics_subset = cptac_transcriptomics.reindex(common_samples, axis=1).T
cptac_proteomics_subset = cptac_proteomics.reindex(common_samples, axis=1).reindex(common_proteins).T


# In[13]:


cptac_transcriptomics_study = cptac_transcriptomics_subset.merge(cptac_sample, right_index=True,
                                                               left_index=True).reset_index().sort_values(['Study', 'index']).set_index('index')
cptac_transcriptomics_study[:2]


# In[15]:


def complete_results(results, fallback_results):
    return(results.parallel_apply(lambda row: fallback_results.loc[row.name] if row.isnull().all() else row,  axis=1))


# In[16]:


def nrmse(ground_truth, predictions):
    if(len(ground_truth) > 4):
        rmse = mean_squared_error(ground_truth, predictions, squared=False)
        if(np.isnan(rmse)):
            return np.nan
        return rmse/(np.max(ground_truth) - np.min(ground_truth))
    else:
        return(np.nan)

def pearson(ground_truth, predictions):
    if(len(predictions)>5):
        return pearsonr(ground_truth, predictions)[0]
    else: 
        return(np.nan)

def spearman(ground_truth, predictions):
    if(len(predictions)>5):
        return spearmanr(ground_truth, predictions)[0]
    else: 
        return(np.nan)

def compute_scores(fold, protein, y_observed, y_predictions):
    y_obs = y_observed.loc[protein].dropna()
    y_pred = y_predictions.loc[protein].dropna()
    
    common_samples= np.intersect1d(y_obs.index, y_pred.index)
    if(len(common_samples) > 0):
        y_obs = y_obs.reindex(common_samples)
        y_pred = y_pred.reindex(common_samples)
        
        return(pd.Series([fold+1, pearson(y_obs, y_pred), spearman(y_obs, y_pred), 
                          r2_score(y_obs, y_pred), mean_squared_error(y_obs, y_pred, squared=False), 
                          nrmse(y_obs, y_pred)], 
                         index=['fold', 'test_pearson', 'test_spearman', 'test_r2', 'test_rmse', 'test_nrmse'], name=protein))
    
    else:
        return(pd.Series([fold+1, np.nan, np.nan, np.nan, np.nan, np.nan], 
                         index=['fold', 'test_pearson', 'test_spearman', 'test_r2', 'test_rmse', 'test_nrmse'], 
                         name=protein, dtype=np.float64))

def score_results(results, study, featureSet):
    scores = []
    desired_study = cptac_transcriptomics_study[cptac_transcriptomics_study['Study'] == study]
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for fold, (train, test) in enumerate(kf.split(desired_study)):
        fold_indices = desired_study.iloc[test].index
        scores.append(results.index.to_series().parallel_apply(lambda protein: compute_scores(fold, protein, 
                                                                                     cptac_proteomics[fold_indices],
                                                                                     results[fold_indices])))
    scores_across_folds = pd.concat(scores)
    mean_score_across_folds = scores_across_folds.groupby(scores_across_folds.index).mean().drop(columns=['fold'])
    mean_score_across_folds['Study'] = study
    mean_score_across_folds['FeatureSet'] = featureSet
    return(mean_score_across_folds)


# In[17]:


brca20_baseline_results = pd.read_parquet(get_file_path([baseline_pred, brca20]))
pdac_baseline_results = pd.read_parquet(get_file_path([baseline_pred, pdac]))
ccrcc_baseline_results = pd.read_parquet(get_file_path([baseline_pred, ccrcc]))
hnscc_baseline_results = pd.read_parquet(get_file_path([baseline_pred, hnscc]))
lscc_baseline_results = pd.read_parquet(get_file_path([baseline_pred, lscc]))
luad_baseline_results = pd.read_parquet(get_file_path([baseline_pred, luad]))


# In[19]:


brca20_fallback_indiv_results = pd.read_parquet(get_file_path([im_fallback_pred, brca20]))
pdac_fallback_indiv_results = pd.read_parquet(get_file_path([im_fallback_pred, pdac]))
ccrcc_fallback_indiv_results = pd.read_parquet(get_file_path([im_fallback_pred, ccrcc]))
hnscc_fallback_indiv_results = pd.read_parquet(get_file_path([im_fallback_pred, hnscc]))
lscc_fallback_indiv_results = pd.read_parquet(get_file_path([im_fallback_pred, lscc]))
luad_fallback_indiv_results = pd.read_parquet(get_file_path([im_fallback_pred, luad]))


# In[20]:


brca20_vae100_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae100, brca20]))
pdac_vae100_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae100, pdac]))
ccrcc_vae100_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae100, ccrcc]))
hnscc_vae100_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae100, hnscc]))
lscc_vae100_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae100, lscc]))
luad_vae100_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae100, luad]))

brca20_vae500_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae500, brca20]))
pdac_vae500_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae500, pdac]))
ccrcc_vae500_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae500, ccrcc]))
hnscc_vae500_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae500, hnscc]))
lscc_vae500_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae500, lscc]))
luad_vae500_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae500, luad]))

brca20_vae1000_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae1000, brca20]))
pdac_vae1000_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae1000, pdac]))
ccrcc_vae1000_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae1000, ccrcc]))
hnscc_vae1000_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae1000, hnscc]))
lscc_vae1000_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae1000, lscc]))
luad_vae1000_indiv_results = pd.read_parquet(get_file_path([indiv_pred, vae1000, luad]))


# In[ ]:


brca20_indiv_all_trans_results = pd.read_parquet(get_file_path([indiv_pred, all_trans, brca20]))
pdac_indiv_all_trans_results = pd.read_parquet(get_file_path([indiv_pred, all_trans, pdac]))
ccrcc_indiv_all_trans_results = pd.read_parquet(get_file_path([indiv_pred, all_trans, ccrcc]))
hnscc_indiv_all_trans_results = pd.read_parquet(get_file_path([indiv_pred, all_trans, hnscc]))
lscc_indiv_all_trans_results = pd.read_parquet(get_file_path([indiv_pred, all_trans, lscc]))
luad_indiv_all_trans_results = pd.read_parquet(get_file_path([indiv_pred, all_trans, luad]))


# In[ ]:


brca20_vae100_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae100, pi400, brca20])), 
                                              brca20_vae100_indiv_results)
ccrcc_vae100_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae100, pi400, ccrcc])), 
                                             ccrcc_vae100_indiv_results)
hnscc_vae100_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae100, pi400, hnscc])), 
                                              hnscc_vae100_indiv_results)
lscc_vae100_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae100, pi400, lscc])), 
                                             lscc_vae100_indiv_results)
luad_vae100_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae100, pi400, luad])), 
                                             luad_vae100_indiv_results)
pdac_vae100_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae100, pi400, pdac])), 
                                             pdac_vae100_indiv_results)

brca20_vae500_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae500, pi400,  brca20])), 
                                               brca20_vae500_indiv_results)
ccrcc_vae500_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae500, pi400,  ccrcc])), 
                                              ccrcc_vae500_indiv_results)
hnscc_vae500_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae500, pi400,  hnscc])), 
                                              hnscc_vae500_indiv_results)
lscc_vae500_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae500, pi400,  lscc])), 
                                             lscc_vae500_indiv_results)
luad_vae500_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae500, pi400,  luad])), 
                                             luad_vae500_indiv_results)
pdac_vae500_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae500, pi400,  pdac])), 
                                             pdac_vae500_indiv_results)

brca20_vae1000_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae1000, pi400, brca20])), 
                                              brca20_vae1000_indiv_results)
ccrcc_vae1000_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae1000, pi400, ccrcc])), 
                                             ccrcc_vae1000_indiv_results)
hnscc_vae1000_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae1000, pi400, hnscc])), 
                                              hnscc_vae1000_indiv_results)
lscc_vae1000_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae1000, pi400, lscc])), 
                                             lscc_vae1000_indiv_results)
luad_vae1000_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae1000, pi400, luad])), 
                                             luad_vae1000_indiv_results)
pdac_vae1000_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, vae1000, pi400, pdac])), 
                                             pdac_vae1000_indiv_results)


# In[ ]:


brca20_cor_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, corum, brca20])), 
                                     brca20_fallback_indiv_results)
ccrcc_cor_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, corum, ccrcc])), 
                                     ccrcc_fallback_indiv_results)
hnscc_cor_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, corum, hnscc])), 
                                     hnscc_fallback_indiv_results)
lscc_cor_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, corum, lscc])), 
                                    lscc_fallback_indiv_results)
luad_cor_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, corum, luad])), 
                                    luad_fallback_indiv_results)
pdac_cor_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, corum, pdac])), 
                                    pdac_fallback_indiv_results)

brca20_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi400, brca20])), 
                                       brca20_fallback_indiv_results)
ccrcc_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi400, ccrcc])), 
                                       ccrcc_fallback_indiv_results)
hnscc_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi400, hnscc])), 
                                       hnscc_fallback_indiv_results)
lscc_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi400, lscc])), 
                                      lscc_fallback_indiv_results)
luad_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi400, luad])), 
                                      luad_fallback_indiv_results)
pdac_pi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi400, pdac])), 
                                      pdac_fallback_indiv_results)

brca20_pi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi800, brca20])), 
                                       brca20_fallback_indiv_results)
ccrcc_pi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi800, ccrcc])), 
                                       ccrcc_fallback_indiv_results)
hnscc_pi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi800, hnscc])), 
                                       hnscc_fallback_indiv_results)
lscc_pi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi800, lscc])), 
                                      lscc_fallback_indiv_results)
luad_pi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi800, luad])), 
                                      luad_fallback_indiv_results)
pdac_pi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, pi800, pdac])), 
                                      pdac_fallback_indiv_results)

brca20_ppi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi400, brca20])), 
                                        brca20_fallback_indiv_results)
ccrcc_ppi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi400, ccrcc])), 
                                        ccrcc_fallback_indiv_results)
hnscc_ppi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi400, hnscc])), 
                                        hnscc_fallback_indiv_results)
lscc_ppi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi400, lscc])), 
                                      lscc_fallback_indiv_results)
luad_ppi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi400, luad])), 
                                       luad_fallback_indiv_results)
pdac_ppi400_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi400, pdac])), 
                                       pdac_fallback_indiv_results)

brca20_ppi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi800, brca20])), 
                                         brca20_fallback_indiv_results)
ccrcc_ppi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi800, ccrcc])), 
                                        ccrcc_fallback_indiv_results)
hnscc_ppi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi800, hnscc])), 
                                        hnscc_fallback_indiv_results)
lscc_ppi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi800, lscc])), 
                                       lscc_fallback_indiv_results)
luad_ppi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi800, luad])), 
                                       luad_fallback_indiv_results)
pdac_ppi800_indiv_results = complete_results(pd.read_parquet(get_file_path([indiv_pred, ppi800, pdac])), 
                                       pdac_fallback_indiv_results)


# In[ ]:


BASELINE = 'Baseline'
CORUM = 'CORUM' 
PI400 = 'PI400'
PI800 = 'PI800'
PPI400 = 'PPI400'
PPI800 = 'PPI800'
VAE100 = 'VAE100'
VAE500 = 'VAE500'
VAE1000 = 'VAE1000'
ALL = 'Transcriptome'
VAE100PI400 = 'VAE100 + PI400'
VAE500PI400 = 'VAE500 + PI400'
VAE1000PI400 = 'VAE1000 + PI400'

stacked_results = pd.concat([score_results(brca20_baseline_results, brca20, BASELINE),
                             score_results(pdac_baseline_results, pdac, BASELINE),
                             score_results(ccrcc_baseline_results, ccrcc, BASELINE),
                             score_results(hnscc_baseline_results, hnscc, BASELINE),
                             score_results(lscc_baseline_results, lscc, BASELINE),
                             score_results(luad_baseline_results, luad, BASELINE),
                             score_results(brca20_cor_indiv_results, brca20, CORUM),
                             score_results(pdac_cor_indiv_results, pdac, CORUM),
                             score_results(ccrcc_cor_indiv_results, ccrcc, CORUM),
                             score_results(hnscc_cor_indiv_results, hnscc, CORUM),
                             score_results(lscc_cor_indiv_results, lscc, CORUM),
                             score_results(luad_cor_indiv_results, luad, CORUM),
                             score_results(brca20_pi400_indiv_results, brca20, PI400),
                             score_results(pdac_pi400_indiv_results, pdac, PI400),
                             score_results(ccrcc_pi400_indiv_results, ccrcc, PI400),
                             score_results(hnscc_pi400_indiv_results, hnscc, PI400),
                             score_results(lscc_pi400_indiv_results, lscc, PI400),
                             score_results(luad_pi400_indiv_results, luad, PI400),
                             score_results(brca20_pi800_indiv_results, brca20, PI800),
                             score_results(pdac_pi800_indiv_results, pdac, PI800),
                             score_results(ccrcc_pi800_indiv_results, ccrcc, PI800),
                             score_results(hnscc_pi800_indiv_results, hnscc, PI800),
                             score_results(lscc_pi800_indiv_results, lscc, PI800),
                             score_results(luad_pi800_indiv_results, luad, PI800),
                             score_results(brca20_ppi400_indiv_results, brca20, PPI400),
                             score_results(pdac_ppi400_indiv_results, pdac, PPI400),
                             score_results(ccrcc_ppi400_indiv_results, ccrcc, PPI400),
                             score_results(hnscc_ppi400_indiv_results, hnscc, PPI400),
                             score_results(lscc_ppi400_indiv_results, lscc, PPI400),
                             score_results(luad_ppi400_indiv_results, luad, PPI400),
                             score_results(brca20_ppi800_indiv_results, brca20, PPI800),
                             score_results(pdac_ppi800_indiv_results, pdac, PPI800),
                             score_results(ccrcc_ppi800_indiv_results, ccrcc, PPI800),
                             score_results(hnscc_ppi800_indiv_results, hnscc, PPI800),
                             score_results(lscc_ppi800_indiv_results, lscc, PPI800),
                             score_results(luad_ppi800_indiv_results, luad, PPI800),                              
                             score_results(brca20_vae100_indiv_results, brca20, VAE100),
                             score_results(pdac_vae100_indiv_results, pdac, VAE100),
                             score_results(ccrcc_vae100_indiv_results, ccrcc, VAE100),
                             score_results(hnscc_vae100_indiv_results, hnscc, VAE100),
                             score_results(lscc_vae100_indiv_results, lscc, VAE100),
                             score_results(luad_vae100_indiv_results, luad, VAE100),
                             score_results(brca20_vae500_indiv_results, brca20, VAE500),
                             score_results(pdac_vae500_indiv_results, pdac, VAE500),
                             score_results(ccrcc_vae500_indiv_results, ccrcc, VAE500),
                             score_results(hnscc_vae500_indiv_results, hnscc, VAE500),
                             score_results(lscc_vae500_indiv_results, lscc, VAE500),
                             score_results(luad_vae500_indiv_results, luad, VAE500), 
                             score_results(brca20_vae1000_indiv_results, brca20, VAE1000),
                             score_results(pdac_vae1000_indiv_results, pdac, VAE1000),
                             score_results(ccrcc_vae1000_indiv_results, ccrcc, VAE1000),
                             score_results(hnscc_vae1000_indiv_results, hnscc, VAE1000),
                             score_results(lscc_vae1000_indiv_results, lscc, VAE1000),
                             score_results(luad_vae1000_indiv_results, luad, VAE1000),
                             score_results(brca20_indiv_all_trans_results, brca20, ALL),
                             score_results(pdac_indiv_all_trans_results, pdac, ALL),
                             score_results(ccrcc_indiv_all_trans_results, ccrcc, ALL), 
                             score_results(hnscc_indiv_all_trans_results, hnscc, ALL),
                             score_results(lscc_indiv_all_trans_results, lscc, ALL),
                             score_results(luad_indiv_all_trans_results, luad, ALL), 
                             score_results(brca20_vae100_pi400_indiv_results, brca20, VAE100PI400),
                             score_results(pdac_vae100_pi400_indiv_results, pdac, VAE100PI400),
                             score_results(ccrcc_vae100_pi400_indiv_results, ccrcc, VAE100PI400),
                             score_results(hnscc_vae100_pi400_indiv_results, hnscc, VAE100PI400),
                             score_results(lscc_vae100_pi400_indiv_results, lscc, VAE100PI400),
                             score_results(luad_vae100_pi400_indiv_results, luad, VAE100PI400),
                             score_results(brca20_vae500_pi400_indiv_results, brca20, VAE500PI400),
                             score_results(pdac_vae500_pi400_indiv_results, pdac, VAE500PI400),
                             score_results(ccrcc_vae500_pi400_indiv_results, ccrcc, VAE500PI400), 
                             score_results(hnscc_vae500_pi400_indiv_results, hnscc, VAE500PI400),
                             score_results(lscc_vae500_pi400_indiv_results, lscc, VAE500PI400),
                             score_results(luad_vae500_pi400_indiv_results, luad, VAE500PI400), 
                             score_results(brca20_vae1000_pi400_indiv_results, brca20, VAE1000PI400),
                             score_results(pdac_vae1000_pi400_indiv_results, pdac, VAE1000PI400),
                             score_results(ccrcc_vae1000_pi400_indiv_results, ccrcc, VAE1000PI400),
                             score_results(hnscc_vae1000_pi400_indiv_results, hnscc, VAE1000PI400),
                             score_results(lscc_vae1000_pi400_indiv_results, lscc, VAE1000PI400),
                             score_results(luad_vae1000_pi400_indiv_results, luad, VAE1000PI400)])])


# In[ ]:


stacked_results.rename(index=lambda x: str(x), inplace=True)
stacked_results.to_parquet(get_results_path('Collated_IM_Results_CPTAC.parquet'))

