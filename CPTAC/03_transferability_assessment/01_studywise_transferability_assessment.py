#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random
import pandas as pd 
import numpy as np
import itertools as it
from pandarallel import pandarallel
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

pandarallel.initialize()


# In[ ]:


get_input_path = lambda fname: os.path.abspath("/home/people/18200264/local_data/input_data2/"+ fname)
get_output_path = lambda fname: os.path.abspath("/home/people/18200264/local_data/output/GuanLab/CPTAC2/"+ fname)


# In[ ]:


file_cptac_transcriptomics = get_input_path('CPTAC_Transcriptomics_processed.parquet')
file_cptac_proteomics = get_input_path('CPTAC_Proteomics_processed.parquet')
file_cptac_sample_info = get_input_path('CPTAC_sample_info.parquet')
file_protein_complexes = get_input_path('allComplexes.txt')
file_pi_400 = get_input_path('protein_interactions_400.csv')
file_pi_800 = get_input_path('protein_interactions_800.csv')
file_ppi_400 = get_input_path('protein_physical_interactions_400.csv')
file_ppi_800 = get_input_path('protein_physical_interactions_800.csv')
file_vae_100_features = get_input_path('encoded_rnaseq_vae100_cptac.parquet')
file_vae_500_features = get_input_path('encoded_rnaseq_vae500_cptac.parquet')
file_vae_1000_features = get_input_path('encoded_rnaseq_vae1000_cptac.parquet')


# In[ ]:


cptac_sample = pd.read_parquet(file_cptac_sample_info)[['Study']]
print("Dimensions: ", cptac_sample.shape)
cptac_sample[:2]


# In[ ]:


cptac_proteomics = pd.read_parquet(file_cptac_proteomics)
print(cptac_proteomics.shape)
cptac_proteomics[:2]


# In[ ]:


cptac_transcriptomics = pd.read_parquet(file_cptac_transcriptomics)
print(cptac_transcriptomics.shape)
cptac_transcriptomics[:2]


# #### Process transcripts and proteins
# 
# 1. Handling Missing values:
#    - Proteins: Exclude proteins with >40% missing values 
#    - Transcripts: Fill missing values with 0s
# 
# 2. Protein isoforms / Repeated samples: 
#    - Aggregating by computing the mean                         
#                                                                                
# 3. Consider only the transcripts with greater than 0 variance   
# 4. Regress the cancer type from proteomics data 

# In[ ]:


common_proteins = np.intersect1d(cptac_transcriptomics.index, cptac_proteomics.index)
common_samples = np.intersect1d(cptac_transcriptomics.columns, cptac_proteomics.columns)
print("Common proteins = " + str(len(common_proteins)) + " and common samples = " + str(len(common_samples)))
cptac_transcriptomics_subset = cptac_transcriptomics.reindex(common_samples, axis=1).T
cptac_proteomics_subset = cptac_proteomics.reindex(common_samples, axis=1).reindex(common_proteins).T


# In[ ]:


cptac_transcriptomics_study = cptac_transcriptomics_subset.merge(cptac_sample, right_index=True,
                                                               left_index=True).reset_index().sort_values(['Study', 'index']).set_index('index')
cptac_transcriptomics_study[:2]


# In[ ]:


cptac_proteomics_study = cptac_proteomics_subset.merge(cptac_sample, right_index=True,
                                                       left_index=True).reset_index().sort_values(['Study', 
                                                                                                   'index']).set_index('index')
cptac_proteomics_study[:2]


# In[ ]:


sample_count_per_study = cptac_proteomics_subset.merge(cptac_sample, right_index=True, left_index=True)['Study'].value_counts()
sample_count_per_study.sort_values(ascending=False, inplace=True)
sample_count_per_study


# #### Protein complex members

# In[ ]:


protein_complexes = pd.read_csv(file_protein_complexes, sep="\t").dropna(subset=['subunits(Gene name)'])
complexes = {c: {x.replace('(', '').replace(')', '') for p in protein_complexes.loc[protein_complexes['ComplexID'] == c, 'subunits(Gene name)'] for x in p.split(';')} for c in protein_complexes['ComplexID']}
complexes = pd.DataFrame({(p1, p2) for c in complexes for (p1, p2) in it.combinations(complexes[c], 2)}, 
                               columns = ['A1', 'A2'])
complexes = complexes[(complexes['A1'].isin(cptac_transcriptomics_subset.columns)) & 
                      (complexes['A2'].isin(cptac_transcriptomics_subset.columns))]

all_protein_pairs = pd.concat([complexes, complexes.rename(columns={'A1': 'A2', 'A2': 'A1'})])
all_protein_pairs = all_protein_pairs.groupby('A1')['A2'].apply(list)
print(all_protein_pairs.shape)
# all_protein_pairs[:2]


# #### StringDB Protein Interaction partners 

# In[ ]:


pi_400 = pd.read_csv(file_pi_400, index_col=0, 
                     converters={"protein2": lambda x: x.strip("[]").replace("'","").split(", ")})
print(len(np.intersect1d(pi_400.index, cptac_proteomics_subset.columns)))
# pi_400[:2]


# In[ ]:


pi_800 = pd.read_csv(file_pi_800, index_col=0, converters={"protein2": lambda x: x.strip("[]").replace("'","").split(", ")})
print(len(np.intersect1d(pi_800.index, cptac_proteomics_subset.columns)))
# pi_800[:2]


# In[ ]:


ppi_400 = pd.read_csv(file_ppi_400, index_col=0, converters={"protein2": lambda x: x.strip("[]").replace("'","").split(", ")})
print(len(np.intersect1d(ppi_400.index, cptac_proteomics_subset.columns)))
# ppi_400[:2]


# In[ ]:


ppi_800 = pd.read_csv(file_ppi_800, index_col=0, converters={"protein2": lambda x: x.strip("[]").replace("'","").split(", ")})
print(len(np.intersect1d(ppi_800.index, cptac_proteomics_subset.columns)))
# ppi_800[:2]


# #### VAE encoded data

# In[ ]:


vae100_features = pd.read_parquet(file_vae_100_features)
# vae100_features[:2]


# In[ ]:


vae500_features = pd.read_parquet(file_vae_500_features)
# vae500_features[:2]


# In[ ]:


vae1000_features = pd.read_parquet(file_vae_1000_features)
# vae1000_features[:2]


# In[ ]:


common_samples = np.intersect1d(vae100_features.index, cptac_proteomics_subset.index)
vae100_features = vae100_features.reindex(common_samples)
vae500_features = vae500_features.reindex(common_samples)
vae1000_features = vae1000_features.reindex(common_samples)


# In[ ]:


FALLBACK_METHOD = "selfmRNA"
CORUM = "selfmRNA + CORUM"
PI400 = "selfmRNA + PI400"
PI800 = "selfmRNA + PI800"
PPI400 = "selfmRNA + PPI400"
PPI800 = "selfmRNA + PPI800"
VAE100 = "selfmRNA + VAE(100)"
VAE100_PI400 = "selfmRNA + VAE(100) + STRINGPI400"
VAE500 = "selfmRNA + VAE(500)"
VAE500_PI400 = "selfmRNA + VAE(500) + STRINGPI400"
VAE1000 = "selfmRNA + VAE(1000)"
VAE1000_PI400 = "selfmRNA + VAE(1000) + STRINGPI400"
ALL = "All transcriptomes"

def add_features(self_mRNA, scenario, other_genes=None):
    if(scenario == VAE100):
        return pd.concat([self_mRNA, vae100_features], axis=1, join='inner')
    elif(scenario == VAE500):
        return pd.concat([self_mRNA, vae500_features], axis=1, join='inner')
    elif(scenario == VAE1000):
        return pd.concat([self_mRNA, vae1000_features], axis=1, join='inner')
    elif(scenario in [CORUM, PI400, PI800, PPI400, PPI800]):
        # check if complex members / interaction partners are null - np.nan is an instance of float
        if(isinstance(other_genes, float)):
            return np.nan
        else: 
            common_genes = np.intersect1d(cptac_transcriptomics_subset.columns, other_genes)
            return pd.concat([self_mRNA, cptac_transcriptomics_subset[common_genes]], axis=1, join='inner')
    elif(scenario in [VAE100_PI400]):
        # check if complex members / interaction partners are null - np.nan is an instance of float
        if(isinstance(other_genes, float)):
            return np.nan
        else: 
            common_genes = np.intersect1d(cptac_transcriptomics_subset.columns, other_genes)
            return pd.concat([self_mRNA, vae100_features, cptac_transcriptomics_subset[common_genes]], axis=1, join='inner')
    elif(scenario in [VAE500_PI400]):
        # check if members / interaction partners are null - np.nan is an instance of float
        if(isinstance(other_genes, float)):
            return np.nan
        else: 
            common_genes = np.intersect1d(cptac_transcriptomics_subset.columns, other_genes)
            return pd.concat([self_mRNA, vae500_features, cptac_transcriptomics_subset[common_genes]], axis=1, join='inner')
    elif(scenario in [VAE1000_PI400]):
        # check if complex members / interaction partners are null - np.nan is an instance of float
        if(isinstance(other_genes, float)):
            return np.nan
        else: 
            common_genes = np.intersect1d(cptac_transcriptomics_subset.columns, other_genes)
            return pd.concat([self_mRNA, vae1000_features, cptac_transcriptomics_subset[common_genes]], axis=1, join='inner')
    else:
        return (cptac_transcriptomics_subset)


# In[ ]:


def perform_prediction(protein, x_train, y_train, x_test, y_test):    
    # check if predictor df is null - np.nan is an instance of float
    if(isinstance(x_train, float) | isinstance(x_test, float)):
        return(pd.Series(index=y_test.index, name=protein, dtype=np.float64))
    
    common_samples = np.intersect1d(x_train.dropna().index, y_train.dropna().index)
    x_train_subset = x_train.reindex(common_samples)
    y_train_subset = y_train.reindex(common_samples)
    
    common_samples = np.intersect1d(x_test.dropna().index, y_test.dropna().index)
    x_test_subset = x_test.reindex(common_samples)
    y_test_subset = y_test.reindex(common_samples)
        
    if(len(x_train_subset) < 2):
        return(pd.Series(index=y_test_subset.index, name=protein, dtype=np.float64))
    
    if(isinstance(x_train_subset, pd.Series)):
        x_train_subset = x_train_subset.values.reshape(-1, 1)
        x_test_subset = x_test_subset.values.reshape(-1, 1)

    predictor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    predictor.fit(x_train_subset, y_train_subset)
    
    y_test_pred = pd.Series(predictor.predict(x_test_subset), index=y_test_subset.index, name=protein)
                        
    return(y_test_pred)


# ### Scenarios: 
# 
# 1. self-mRNA + VAE
# 2. self-mRNA + VAE + CORUM 
# 3. self-mRNA + VAE + STRING 400
# 4. self-mRNA + VAE + STRING 800
# 5. self-mRNA + VAE + STRING physical interactions 400
# 6. self-mRNA + VAE + STRING physical interactions 800

# In[ ]:


genes_with_max_mat = []
genes_with_max_trans = []
for study in cptac_proteomics_study['Study'].value_counts().index:
    study_proteomics = cptac_proteomics_study[cptac_proteomics_study['Study'] == study].drop(columns=['Study'])
    study_transcriptomics = cptac_transcriptomics_study[cptac_transcriptomics_study['Study'] == study].drop(columns=['Study'])
    
    study_proteomics.dropna(thresh=0.6*len(study_proteomics), axis=1, inplace=True)
    study_transcriptomics = study_transcriptomics.loc[:, study_transcriptomics.isin([0]).mean() < .6]
    genes_with_max_trans.append(study_transcriptomics.columns)
    genes_with_max_mat.append(np.intersect1d(study_proteomics.columns, study_transcriptomics.columns))
genes_to_predict = list(set(genes_with_max_mat[0]).intersection(*genes_with_max_mat))
genes_as_features = list(set(genes_with_max_trans[0]).intersection(*genes_with_max_trans))
print("Proteins to predict (commonly present across all studies): ", len(genes_to_predict))
print("Proteins to be used as features (commonly present across all transcriptomic studies): ", len(genes_as_features))


# In[ ]:


def get_studywise_train_test_split(study):
    # ensuring the proteomics and transcriptomics data is available for at least 60% of the samples
    test_proteomics = cptac_proteomics_study[cptac_proteomics_study['Study'] == study].drop(columns=['Study'])
    test_transcriptomics = cptac_transcriptomics_study[cptac_transcriptomics_study['Study'] == study].drop(columns=['Study']) 
    
    train_transcriptomics = cptac_transcriptomics_study[cptac_transcriptomics_study['Study'] != study].drop(columns=['Study'])
    train_proteomics = cptac_proteomics_study[cptac_proteomics_study['Study'] != study].drop(columns=['Study'])
    
    # Select only those proteins with both proteomic and transcriptomic data available for at least 60% 
    # samples in each study
    test_proteomics = test_proteomics.reindex(genes_to_predict, axis=1)
    test_transcriptomics = test_transcriptomics.reindex(genes_as_features, axis=1)
    train_transcriptomics = train_transcriptomics.reindex(genes_as_features, axis=1)
    train_proteomics = train_proteomics.reindex(genes_to_predict, axis=1)
    
    return(train_transcriptomics, train_proteomics, test_transcriptomics, test_proteomics)


def run_parallelized_baseline_pipeline(study, feature_set, desired_filename): 
    all_folds_results = []
    print("Running " + study)
    X_train, y_train, X_test, y_test = get_studywise_train_test_split(study)
    
    assert y_test.shape[1] == 4046, "Number of proteins to predictis not as expected for the test study"
    assert y_train.shape[1] == 4046, "Number of proteins to predictis not as expected for the train studies"
    assert X_train.shape[1] == 12580, "Number of input features to be used is not as expected for the test study"
    assert X_test.shape[1] == 12580, "Number of input features to be used is not as expected for other studies"
          
    all_folds_results.append(y_train.columns.to_series().parallel_apply(lambda p: \
                                      perform_prediction(p, X_train[p], y_train[p], 
                                                         X_test[p], y_test[p])))
    print("\tCompleted")
    
    pd.concat(all_folds_results, axis=1).to_parquet(get_output_path(desired_filename))

def run_parallelized_prediction_pipeline(study, feature_set, desired_filename): 
    all_folds_results = []
    print("Running " + study)
    X_train, y_train, X_test, y_test = get_studywise_train_test_split(study)
    
    assert y_test.shape[1] == 4046, "Number of proteins to predictis not as expected for the test study"
    assert y_train.shape[1] == 4046, "Number of proteins to predictis not as expected for the train studies"
    assert X_train.shape[1] == 12580, "Number of input features to be used is not as expected for the test study"
    assert X_test.shape[1] == 12580, "Number of input features to be used is not as expected for other studies"
    
    all_folds_results.append(y_train.columns.to_series().parallel_apply(lambda p: \
                                      perform_prediction(p, add_features(X_train[p], feature_set), 
                                                         y_train[p], 
                                                         add_features(X_test[p], feature_set), 
                                                         y_test[p])))
    print("\tCompleted!")
    
    pd.concat(all_folds_results, axis=1).to_parquet(get_output_path(desired_filename))
    
def run_parallelized_prediction_pipeline2(study, feature_set, desired_filename, additional_features): 
    all_folds_results = []
    print("Running " + study)
    X_train, y_train, X_test, y_test = get_studywise_train_test_split(study)
    
    assert y_test.shape[1] == 4046, "Number of proteins to predictis not as expected for the test study"
    assert y_train.shape[1] == 4046, "Number of proteins to predictis not as expected for the train studies"
    assert X_train.shape[1] == 12580, "Number of input features to be used is not as expected for the test study"
    assert X_test.shape[1] == 12580, "Number of input features to be used is not as expected for other studies"
    
    all_folds_results.append(y_train.columns.to_series().parallel_apply(lambda p: \
                                      perform_prediction(p, add_features(X_train[p], feature_set, 
                                                                         additional_features.reindex(index=[p]).squeeze()), 
                                                         y_train[p], add_features(X_test[p], feature_set, 
                                                                     additional_features.reindex(index=[p]).squeeze()), 
                                                         y_test[p])))
    print("\tCompleted")
    
    pd.concat(all_folds_results, axis=1).to_parquet(get_output_path(desired_filename))


# In[ ]:


parquet_extension = ".parquet"
fallback_method_prefix = "loso_tm_results_fallback_"
corum_file_prefix = 'loso_tm_results_corum_'
pi400_file_prefix = "loso_tm_results_pi400_"
pi800_file_prefix = "loso_tm_results_pi800_"
ppi400_file_prefix = "loso_tm_results_ppi400_"
ppi800_file_prefix = "loso_tm_results_ppi800_"
vae100_file_prefix = "loso_tm_results_vae100_"
vae500_file_prefix = "loso_tm_results_vae500_"
vae100_pi400_file_prefix = "loso_tm_results_vae100_pi400_"
vae500_pi400_file_prefix = "loso_tm_results_vae500_pi400_"
vae1000_file_prefix = "loso_tm_results_vae1000_"
vae1000_pi400_file_prefix = "loso_tm_results_vae1000_pi400_"
all_transcriptomes_file_prefix = "loso_tm_results_all_transcriptomes_"


# In[ ]:


print(FALLBACK_METHOD)
for index, value in sample_count_per_study.head(6).items():
    desired_filename = fallback_method_prefix + index + parquet_extension
    run_parallelized_baseline_pipeline(index, FALLBACK_METHOD, desired_filename)

print(PI400)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = pi400_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline2(index, PI400, desired_filename, pi_400)

print(VAE100)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = vae100_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline(index, VAE100, desired_filename)

print(VAE500)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = vae500_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline(index, VAE500, desired_filename)

print(VAE100_PI400)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = vae100_pi400_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline2(index, VAE100_PI400, desired_filename, pi_400)

print(VAE500_PI400)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = vae500_pi400_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline2(index, VAE500_PI400, desired_filename, pi_400)

print(VAE1000)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = vae1000_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline(index, VAE1000, desired_filename)
    
print(VAE1000_PI400)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = vae1000_pi400_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline2(index, VAE1000_PI400, desired_filename, pi_400)

print(ALL)
for index,value in sample_count_per_study.head(6).items():
    desired_filename = all_transcriptomes_file_prefix + index + parquet_extension
    run_parallelized_prediction_pipeline(index, ALL, desired_filename)

