
# 2022.0287

### ![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)

# Controlling Homophily in Social Network Regression Analysis by Machine Learning

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

This repository contains supporting material for the paper "Controlling Homophily in Social Network Regression Analysis by Machine Learning" by Xuanqi Liu, and Ke-Wei Huang.

# Cite
To cite the contents of this repository, please cite both the paper and this repo, using the following DOIs.

[https://doi.org/10.1287/ijoc.2022.0287](https://doi.org/10.1287/ijoc.2022.0287)

[https://doi.org/10.1287/ijoc.2022.0287.cd](https://doi.org/10.1287/ijoc.2022.0287.cd)

Below is the BibTex for citing this version of the code.
```latex
@article{Liu2024IJOC,
  author =        {Liu, Xuanqi and Huang, Ke-Wei},
  publisher =     {INFORMS Journal on Computing},
  title =         {Controlling Homophily in Social Network Regression Analysis by Machine Learning},
  year =          {2024},
  doi =           {10.1287/ijoc.2022.0287.cd},
  url =           {https://github.com/INFORMSJoC/2022.0287},
  note = {Available for download at https://github.com/INFORMSJoC/2022.0287)},
}  
```

## Description
Empirical studies related to social networks has been one of the most popular
research subjects in recent years. A frequently examined topic within these studies is the quantification of
peer influence while controlling for homophily effects. However, there exists scarce literature on controlling the latent homophily effects in observational settings. This study proposes two ML-based methods that leverage network structure information to better control for latent homophily. The first method incorpoares node embedding into double machine learning estimator to partially out the latent homophily effect. The second method involves estimating peer influence effect by a novel neural network model.

This project contains four folders: `data`, `results`,`codes`.
- `data`: include simulated undirected networks, specifications for peer influence effects and outcomes, as well as the corresponding node embedding vectors. 
- `results`: include all experimental results.
- `codes`: include both source codes to simulate network data and codes to directly replicate the experimental results in the paper.

## Results
`results\Table2`: reports the overall estimation results for pure homophily case. Detailed estimation results for each model across 100 simulated datasets can be found in separate worksheets within this Excel file. Additionally, beta distribution (refer to Figure 2(a) in the paper) is provided in a worksheet in the same file.

`results\Table3`: reports the overall estimation results for positive peer effect case. Detailed estimation results for each model on each simulated dataset are available in separate worksheets in this Excel file. Additionally, beta distribution (refer to Figure 2(b) in the paper) is saved in a worksheet in the same file.

`results\TableA2`: reports the pairwise relationship between latent homophily features and individual embedding dimension. 

`results\TableA3`: reports the regression of latent homophily features on entire embedding vector. 

`results\FigureA2(a,b,c,d)`: illustrates four relationship patterns between latent homophily features and individual embedding dimension. 

`results\FigureA3`: reports the robustness check results for pure homophily case in Appendix H.

`results\FigureA4`: reports the robustness check results for positive peer efffect case in Appendix H.

## Replicating
To get main estimation results of Table 2 and Table 3 in the paper, please run the following 5 codes in the `codes` folder. 

- To get the baseline estimation results in Panel (a), (b) and (c) of Tables, do
  - ``` 
    python codes/estimation/03.Get_baseline_results.py  
    ```
  - This python script generates the estimation results of True model, Basic model, IV model, Centrality model, Chen et al. (2022) and its variants, PSM, and PSW.

- To get the estimation results of GPSW, do
  - ``` 
    Rscript codes/estimation/03.Get_GPSW_results.R
    ```
  - This R script has been slightly modified from Zhu et al. (2015). 
  - Access their code: https://www.degruyter.com/document/doi/10.1515/jci-2014-0022/html#APP0001_w2aab3b7e2173b1b6b1ab2aaAa

- To get the estimation results of stepwise regression model, do
  - ``` 
    stata -b do codes/estimation/03.Get_stepwise regression_results.do
    ```
- To get the estimation results of the proposed estimators, do
  - ``` 
    python codes/estimation/03.Get_DML_resultss.py
    python codes/estimation/03.Get_OML_resultss.py
    ```


See the README.md file inside the `codes` folder for detailed instructions on replicating our findings from scratch.
