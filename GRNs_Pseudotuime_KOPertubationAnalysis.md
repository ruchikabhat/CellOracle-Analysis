# Start celloracle analysis from scratch to find Gene Regulatory Networks and Pseudotime analysis and KO pertubations of important genes obtained from analysis.

```import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = [6, 4.5]

#Assuming you have run the preanalysis step of converting your final rds to h5ad and saved that data in mouse.h5ad

adata=sc.read_h5ad("/data-store/iplant/home/ruchikabhat/data/CellOracle/mouse.h5ad")

adata
        #O/P: AnnData object with n_obs × n_vars = 24453 × 32285
         obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'percent.ribo', 'Log10_nCount', 'Log10_nFeature', 'Sample', 'Condition', 'RNA_snn_res.0.8', 'seurat_clusters', 'Source', 'nCount_Protein', 'nFeature_Protein', 'Sub_Celltypes', 'Sub_Celltypes_2', 'Ref_celltypes', 'LBF_U', 'final_subtypes', 'updated_subtypes', 'Clusters', 'barcode', 'UMAP_1', 'UMAP_2'
         uns: 'Clusters_colors'
         obsm: 'X_pca', 'X_umap'#

adata_ori=adata

adata.obs['ClusCond'] = ''
adata.obs['ClusCond'] = adata.obs[['Clusters', 'Condition']].agg(' '.join, axis=1)

adata_ao= adata[adata.obs['Condition'].isin(['Adult','Old'])]

adata=adata_ao

adata_ao= adata[adata.obs['LBF_U'].isin(['FRCs'])]

adata=adata_ao

adata

     #O/P:View of AnnData object with n_obs × n_vars = 9559 × 32285
    obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'percent.ribo', 'Log10_nCount', 'Log10_nFeature', 'Sample', 'Condition', 'RNA_snn_res.0.8', 'seurat_clusters', 'Source', 'nCount_Protein', 'nFeature_Protein', 'Sub_Celltypes', 'Sub_Celltypes_2', 'Ref_celltypes', 'LBF_U', 'final_subtypes', 'updated_subtypes', 'Clusters', 'barcode', 'UMAP_1', 'UMAP_2', 'ClusCond'
    uns: 'Clusters_colors'
    obsm: 'X_pca', 'X_umap'##

adata.obs.ClusCond.unique().tolist()
adata.obs


#just clearing things I dont need

adata.uns = {}
adata.obsp = {}
adata.obsm = {}


filter_result = sc.pp.filter_genes_dispersion(adata.X,
                                              n_top_genes=3000,
                                              log=False)

# Subset the genes
adata = adata[:, filter_result.gene_subset]

adata

      #O/P:View of AnnData object with n_obs × n_vars = 9559 × 3000
       #obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'percent.ribo', 'Log10_nCount', 'Log10_nFeature', 'Sample', 'Condition', 'RNA_snn_res.0.8', 'seurat_clusters', 'Source', 'nCount_Protein', 'nFeature_Protein', 'Sub_Celltypes', 'Sub_Celltypes_2', 'Ref_celltypes', 'LBF_U', 'final_subtypes', 'updated_subtypes', 'Clusters', 'barcode', 'UMAP_1', 'UMAP_2', 'ClusCond'

```
# Now Do PSEUDOTIME Analysis

```
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scvelo as scv #just using for dataset

import celloracle as co
from celloracle.applications import Pseudotime_calculator
from celloracle.applications import Gradient_calculator
from celloracle.applications import Oracle_development_module

%matplotlib inline

adata.raw = adata

FRC= adata

np.random.seed(42)

adata.layers['counts'] = adata.X.copy()

#Only consider genes with more than 1 count
sc.pp.filter_genes(adata, min_cells = 1)

#sc.pp.highly_variable_genes(adata, subset = True, inplace = True, flavor='cell_ranger')

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, subset = True, inplace = True)

#conda install -c conda-forge leidenalg

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.3)


sc.tl.umap(adata)

sc.pl.umap(adata, color='leiden', legend_loc='on data')

sc.pl.umap(adata, color='Clusters', legend_loc='on data')

pt = Pseudotime_calculator(adata=adata,
                           obsm_key="X_umap", # Dimensional reduction data name
                           cluster_column_name="ClusCond" # Clustering data name
                           )

lineage_dict = {'the_one':adata.obs.ClusCond.unique().tolist()} #can have multiple lineages

lineage_dict

pt.set_lineage(lineage_dictionary=lineage_dict)
pt.plot_lineages()

coord_sum = adata.obsm['X_umap'].sum(axis = 1)
max_sum = pd.Series(coord_sum).idxmax()
root_cells = {"the_one": adata.obs_names[max_sum]}

root_cells

pt.set_root_cells(root_cells=root_cells)

pt.plot_root_cells()

sc.tl.diffmap(pt.adata)

pt.get_pseudotime_per_each_lineage()

pt.plot_pseudotime(cmap="rainbow")

plt.savefig('pt.png', dpi = 300)

pt.adata.obs

base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()

oracle = co.Oracle()
oracle.import_anndata_as_raw_count(adata = adata, cluster_column_name = 'ClusCond', embedding_name = 'X_umap')

oracle.import_TF_data(TF_info_matrix=base_GRN)

oracle.perform_PCA()

# Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
plt.show()
print(n_comps)
n_comps = min(n_comps, 50)

n_cell = oracle.adata.shape[0]
print(f"cell number is :{n_cell}")

k = int(0.025*n_cell)
print(f"Auto-selected k is :{k}")

oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)

%%time

links = oracle.get_links(cluster_name_for_GRN_unit="ClusCond", alpha=10,
                         verbose_level=10)

links.filter_links(p=0.05, weight="coef_abs", threshold_number=10000) #10k is default top num to keep

links.plot_degree_distributions(plot_model=True)

links.get_network_score()

links.merged_score

## OR can do links for condition (optional)
%%time
link=oracle.get_links(cluster_name_for_GRN_unit="Condition")

#save results as csv
link.links_dict['Adult'].to_csv(f"raw_GRN_for_Adult.csv")
#or
links.links_dict["Nr4a1+ Old"].to_csv(f"raw_GRN_for_Nr4a1+Old.csv")

plt.rcParams["figure.figsize"] = [9, 7.5]
links.plot_scores_as_rank(cluster="Nr4a1+ Adult", n_gene=30)

fig, ax = plt.subplots(figsize=[10, 11])
plt.rcParams['xtick.major.pad'] = 40
plt.rcParams['ytick.major.pad'] = 40
plt.ticklabel_format(style='sci', axis='y',scilimits=(0,0),)
links.plot_score_comparison_2D(value="degree_centrality_all",
                               cluster1="Nr4a1+ Adult", cluster2="Nr4a1+ Old",
                               percentile=91)

fig, ax = plt.subplots(figsize=[10, 11])
plt.rcParams['xtick.major.pad'] = 40
plt.rcParams['ytick.major.pad'] = 40
plt.ticklabel_format(style='sci', axis='y',scilimits=(0,0),)
links.plot_score_comparison_2D(value="eigenvector_centrality",
                               cluster1="Nr4a1+ Adult", cluster2="Nr4a1+ Old",
                               percentile=91)

fig, ax = plt.subplots(1, 2,  figsize=[6, 6])
links.plot_score_per_cluster(goi="Osr2")

oracle.get_cluster_specific_TFdict_from_Links(links_object=links)

oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

sc.pl.umap(oracle.adata, color = ['Klf6'], layer="imputed_count", size=38)

sc.get.obs_df(oracle.adata, keys=["Osr2"],layer="imputed_count").max()
    #O/P:Osr2    0.071794
         dtype: float64

#### DOuble the expression of genes of Interest

oracle.simulate_shift(perturb_condition={'Osr2': 0.14}, n_propagation=3)

# Double KOs of Bhlhe40 and Nfkb1
oracle.simulate_shift(perturb_condition={'Bhlhe40': 0.0,'Nfkb1': 0.0}, n_propagation=3)

oracle.simulate_shift(perturb_condition={'Osr2': 0.0}, n_propagation=3)












