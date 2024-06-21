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

## Add the column with Cluster and condition information together
adata.obs['ClusCond'] = ''
adata.obs['ClusCond'] = adata.obs[['Clusters', 'Condition']].agg(' '.join, axis=1)

## Choose only Adult and Old conditions not the other conditions
adata_ao= adata[adata.obs['Condition'].isin(['Adult','Old'])]

adata=adata_ao

## Now choose only FRCs from the metadata column LBF_U with entries as FRCs.
adata_ao= adata[adata.obs['LBF_U'].isin(['FRCs'])]

adata=adata_ao

## See what adata contains (9559 cells and 32285 genes)
adata

     #O/P:View of AnnData object with n_obs × n_vars = 9559 × 32285
    obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'percent.ribo', 'Log10_nCount', 'Log10_nFeature', 'Sample', 'Condition', 'RNA_snn_res.0.8', 'seurat_clusters', 'Source', 'nCount_Protein', 'nFeature_Protein', 'Sub_Celltypes', 'Sub_Celltypes_2', 'Ref_celltypes', 'LBF_U', 'final_subtypes', 'updated_subtypes', 'Clusters', 'barcode', 'UMAP_1', 'UMAP_2', 'ClusCond'
    uns: 'Clusters_colors'
    obsm: 'X_pca', 'X_umap'##

# to check what is there in ClusCond column as unique entries

adata.obs.ClusCond.unique().tolist()
adata.obs


#just clearing things I dont need

adata.uns = {}
adata.obsp = {}
adata.obsm = {}

# to keep only top 3000 genes for downstream processing
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

#Keep one raw copy safe
adata.raw = adata

FRC= adata

np.random.seed(42)

adata.layers['counts'] = adata.X.copy()

#Only consider genes with more than 1 count
sc.pp.filter_genes(adata, min_cells = 1)

#sc.pp.highly_variable_genes(adata, subset = True, inplace = True, flavor='cell_ranger')

# Log normalise the data
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Find highly variable genes
sc.pp.highly_variable_genes(adata, subset = True, inplace = True)

# If you haven't installed leidenalg please uncomment the below command and install it before going further
#conda install -c conda-forge leidenalg

# Run PCA and neighbour finding
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.3)

# Run UMAP
sc.tl.umap(adata)

# this will show cluster numbers on the UMAP
sc.pl.umap(adata, color='leiden', legend_loc='on data')

# this will show cluster names stored under the metadata column 'Clusters' of your annotated data on the UMAP
sc.pl.umap(adata, color='Clusters', legend_loc='on data')

# Run pseudotime on ClusCond Column
pt = Pseudotime_calculator(adata=adata,
                           obsm_key="X_umap", # Dimensional reduction data name
                           cluster_column_name="ClusCond" # Clustering data name
                           )

lineage_dict = {'the_one':adata.obs.ClusCond.unique().tolist()} #can have multiple lineages

# this will list the sequence of the lineage predicted
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
```
# Let's do GRN calculations
```
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
```
# Save results as csv

```

# to save the full Base GRN dictionary
link.links_dict['Adult'].to_csv(f"raw_GRN_for_Adult.csv")
#or
links.links_dict["Nr4a1+ Old"].to_csv(f"raw_GRN_for_Nr4a1+Old.csv")
# to save only the filtered GRNS
links.filtered_links["Inmt+ Adult"].to_csv(f"Inmt+AdultFiltered.csv")
```
# Find Genes of Interest for Knock Out Pertubations

```
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
```
#### Double the expression of genes of Interest

```oracle.simulate_shift(perturb_condition={'Osr2': 0.14}, n_propagation=3)```

### Double KOs of Bhlhe40 and Nfkb1

```oracle.simulate_shift(perturb_condition={'Bhlhe40': 0.0,'Nfkb1': 0.0}, n_propagation=3)```

### Single KO of goi

```oracle.simulate_shift(perturb_condition={'Osr2': 0.0}, n_propagation=3)```

# Analysis on the pseudotime trajectory
```
oracle.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

oracle.calculate_embedding_shift(sigma_corr=0.05)

fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 40 #bigger = smaller vector
# Show quiver plot
oracle.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: Double exp Osr2")

# Show quiver plot that was calculated with randomized graph.
oracle.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()

#plt.figure(None(10,10))
oracle.plot_grid_arrows(quiver_scale=2.0,
                       scatter_kwargs_dict={"alpha":0.35,"lw":0.35,
                                              "edgecolor":0.4,"s":38,"rasterized":True},
                                              min_mass=0.015,angles='xy',scale_units='xy',
                                             headaxislength=2.75,headlength=5,headwidth=4.8,minlength=1.5,
                                             plot_random=False,scaletype="relative")

n_grid = 40
oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

oracle.suggest_mass_thresholds(n_suggestion=12)

min_mass = 6.2
oracle.calculate_mass_filter(min_mass=min_mass, plot=True)

fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale_simulation = 20
# Show quiver plot
oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector:Double Expression of Osr2 (0.14)")

# Show quiver plot that was calculated with randomized graph.
oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()

oracle.adata.obs['Pseudotime'] = pt.adata.obs.Pseudotime

gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key="Pseudotime")

gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)

gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly":3}, plot=True)

gradient.calculate_gradient()

scale_dev = 40
gradient.visualize_results(scale=scale_dev, s=5)

# pseudotime gradient
fig, ax = plt.subplots(figsize=[6, 6])
gradient.plot_dev_flow_on_grid(scale=scale_dev, ax=ax)

dev = Oracle_development_module()

dev.load_differentiation_reference_data(gradient_object=gradient)

dev.load_perturb_simulation_data(oracle_object=oracle)

# Calculate inner produc scores
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)

# Show perturbation scores
vm = 1 #adjust based on data

fig, ax = plt.subplots(1, 2, figsize=[12, 6])
dev.plot_inner_product_on_grid(vm=vm, s=50, ax=ax[0])
ax[0].set_title(f"PS")

dev.plot_inner_product_random_on_grid(vm=vm, s=50, ax=ax[1])
ax[1].set_title(f"PS calculated with Randomized simulation vector")
plt.show()

# Show perturbation scores with perturbation simulation vector field
fig, ax = plt.subplots(figsize=[6, 6])
dev.plot_inner_product_on_grid(vm=vm, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=scale_simulation, show_background=False, ax=ax)

plt.savefig('pertubation_score_Double_Exp0.14Osr2.png', dpi = 300)

dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=scale_simulation,
                                          s_grid=50,
                                          scale_for_pseudotime=scale_dev,
                                          vm=vm)

```

Hopefully you might have enjoyed using this script to analyse your scRNA data :)










