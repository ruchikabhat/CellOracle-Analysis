# Convert Seurat rds to .h5ad

#R Script

#Date: 10 April 2024

#Time: 9:20 am

#Author: Dr. Ruchika Bhat

#Script helps with generating files needed for .rds to h5ad conversion

# Assuming you have your final .rds object loaded in the seurat object named 'mouse'

#Generate these files

```mouse<-readRDS("yourcomplete.rdsfile")

```counts_matrix <-GetAssayData(mouse, assay='RNA',slot='counts')```

#save matrix file

```writeMM(counts_matrix, file=paste0(file='matrix.mtx')) ```

#save pca embeddings

```write.csv(mouse@reductions$pca@cell.embeddings,file='pca.csv', quote=F, row.names=F) ```
```write.table(data.frame('gene'=rownames(counts_matrix)),file='gene_names.csv',quote=F,row.names=F,col.names=F)```

#Save cell barcode information

```mouse$barcode<-colnames(mouse)```

#Save UMAP reductions

```mouse$UMAP_1<-mouse@reductions$umap@cell.embeddings[,1]```

```mouse$UMAP_2<-mouse@reductions$umap@cell.embeddings[,2]```

#Finally save the metadata

```write.csv(mouse@meta.data,file='metadata.csv', quote=F,row.names=F)```



#######################################################################################
# Now go to Jupyter Notebook to convert it into h5ad with same embeddings information

```import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scanpy as sc
import igraph
import scvelo as scv
import loompy as lmp
import anndata
from scipy import io
from scipy.sparse import coo_matrix, csr_matrix
import os


X= io.mmread("/data-store/iplant/home/ruchikabhat/data/CellOracle/matrix.mtx")

adata =anndata.AnnData(X=X.transpose().tocsr())

metadata = pd.read_csv("/data-store/iplant/home/ruchikabhat/data/CellOracle/metadata.csv")

with open("/data-store/iplant/home/ruchikabhat/data/CellOracle/gene_names.csv",'r') as f:
      gene_names = f.read().splitlines()

adata.obs = metadata
adata.obs.index =adata.obs['barcode']
adata.var.index = gene_names

pca =pd.read_csv("/data-store/iplant/home/ruchikabhat/data/CellOracle/pca.csv")
pca.index =adata.obs.index

adata.obsm['X_pca'] = pca.to_numpy()
adata.obsm['X_umap'] = np.vstack((adata.obs['UMAP_1'].to_numpy(), adata.obs['UMAP_2'].to_numpy())).T

sc.pl.umap(adata, color =['Clusters'],frameon=False, save=True)

# Save everything as a .h5ad file

adata.write("/data-store/iplant/home/ruchikabhat/data/CellOracle/mouse.h5ad")

# Read this data back

adata=sc.read_h5ad("/data-store/iplant/home/ruchikabhat/data/CellOracle/mouse.h5ad")

