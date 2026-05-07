library(devtools)
#devtools::install_github("JSB-UCLA/scDEED")

library(MuDataSeurat)
library(Seurat)
library(gridExtra)
library(dplyr)
library(patchwork)
library(VGAM)
library(gplots)
library(ggplot2)
library(pracma)
library(resample)
library(foreach)
library(distances)
library(utils)
library(doParallel)
library(scDEED)
library(reticulate)
library(stringr)
library(stringi)
library(viridis)
library(writexl)


use_condaenv('<ENV_ROOT>/BrainAtlas')
##You can change the number of cores here
#registerDoParallel(cores=6)

grid_search_res = '<PROJECT_ROOT>/grid_search'
date = '260131'

#model.dir = '<PROJECT_ROOT>/results_with_gridsearch'
model.dir = '<PROJECT_ROOT>/results/Telen-WholeHuman'
model.list = list.files(model.dir, pattern='h5ad')
model.list

#model.names = c('Telen-C2S') #'BrainCell-WholeHuman', 'Telen-C2S', 'Telen-CellFM', 'Telen-Mouse'


for (model in model.list){
  print(sprintf("scDEED for %s", model))
  #model.dir = sprintf("<PROJECT_ROOT>/results/%s", model_name)
  #model = paste0(model_name, '.h5ad')
  data_path = sprintf('%s/%s', model.dir, model)
  
  data = ReadH5AD(data_path)
  print(data)
  
  # model_name = str_split(model, '_')[[1]][1] 
  model_name = str_remove(model, '.h5ad')
  
  data@assays$RNA@counts = data@assays$RNA@data
  data = ScaleData(data)
  data[["RNA"]] <- as(object = data[["RNA"]], Class = "Assay5")
  
  data = FindVariableFeatures(data)
  #data
  
  data = RunPCA(data) # default npcs=50
  Seurat::ElbowPlot(data, ndims=50)
  
  K = 40 # <- user setting
  
  # Before optimization
  data = RunUMAP(data, dims=1:K, seed.use=98, umap.method='umap-learn')
  FeaturePlot(data, reduction = 'umap', 'pLI', pt.size=1) +
    scale_colour_viridis(option = "C") +
    scale_fill_viridis(option = "C")
  
  
  # Run scDEED -> hold K, and fine-tuned n_neighbors and min.dist
  result = scDEED(data, K = K, n_neighbors = c(5, 10, 20, 30, 40, 50), 
                  min.dist = c(0.1, 0.5), reduction.method = 'umap', rerun = F)
  result$num_dubious
  
  
  min(result$num_dubious$number_dubious_cells)
  opt = which(result$num_dubious$number_dubious_cells==min(result$num_dubious$number_dubious_cells))
  m = result$num_dubious$min.dist[opt]
  n = result$num_dubious$n_neighbors[opt]
  print(m)
  print(n)
  
  
  dubious_cells = result$full_results$dubious_cells[opt]
  dubious_cells = as.numeric(strsplit(dubious_cells, ',')[[1]])
  trustworthy_cells =  result$full_results$trustworthy_cells[opt]
  trustworthy_cells = as.numeric(strsplit(trustworthy_cells, ',')[[1]])
  data_opt = RunUMAP(data, dims = 1:K, min.dist = m[1], n.neighbors = n[1], seed.use = 98)
  
  
  DimPlot(data_opt, reduction = 'umap', 
          cells.highlight = list('dubious' = dubious_cells, 'trustworthy' = trustworthy_cells)) + 
    scale_color_manual(values = c('gray', 'blue', 'red'))
  
  
  # After optimization
  FeaturePlot(data_opt, reduction = 'umap', 'pLI', pt.size=1) +
    scale_colour_viridis(option = "C") +
    scale_fill_viridis(option = "C")
  
  
  saveRDS(result, file=sprintf("<PROJECT_ROOT>/grid_search/%s_scDEED_results_objects.rds", model_name))
}
  

##
res.dir = '<PROJECT_ROOT>/grid_search'
res.files = list.files(res.dir, pattern='rds')
length(res.files)

all_res_list = list()

for (res_file in res.files){
  res_path = sprintf('%s/%s', res.dir, res_file)
  res_path
  
  res = readRDS(res_path)
  
  model_name = str_remove(res_file, "_scDEED_results_objects.rds")
  model_name
  
  opt = which(res$num_dubious$number_dubious_cells==min(res$num_dubious$number_dubious_cells))
  m = res$num_dubious$min.dist[opt] 
  n = res$num_dubious$n_neighbors[opt]
  
  print(model_name)
  print(m)
  print(n)
  
  if (length(n) > 2){
    if (length(n)==12){
      m = 0.5
      n = 15
    }else{
      m = m[[1]] %>% as.numeric()
      n = n[[1]] %>% as.numeric()
    }
  }else{
    m = m[[1]] %>% as.numeric()
    n = n[[1]] %>% as.numeric()
  }
  
  model_res = c(
    'Model'=model_name,
    'min_dist'=m,
    'n_neighs'=n
  )
  
  all_res_list[[model_name]] = model_res
}
  
all_res = bind_rows(all_res_list)
all_res


write_xlsx(all_res, sprintf('%s/All_model_prediction_embedding_optimal_paramters.%s.xlsx', grid_search_res, date))
  

  
  