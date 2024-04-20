
## Replicating results from scratch

Please follow the instructions to replicate results from scratch:

1. To simulate newtworks, specify peer influence effects/outcomes, and calculate instrument, please run the following script in the `codes/data_preparation` file
- ``` 
    Rscript codes/01.Simulate network structure and peer effect cases.R
    ```
- This R script has been adapted from Shalizi and Thomas (2011). (Access their R code: http://www.stat.cmu.edu/cshalizi/homophilyconfounding/).
- The script simulates the following:
    - **edge_list_undirected.txt**: Undirected networks comprising 1000 nodes. 
    - **features.csv**: Each node has two latent homophily features following a normal distribution. Outcomes are simulated in two senarios: 
    **(1) pure homophily effect case**: peer coefficient = 0, and **(2) positive peer influence case**: peer coefficient = 0.2. Finally, we calculate the **intrument variable** as the average outcome of friends with a two-degree connection (but not one-degree connection).

2. To calculate traditional centrality measures and generate node embeddings from network structures, please run the following script in the `codes/data_preparation` file
- ``` 
    python codes/02.Calculate_centrality_and_generate_embedding.py
    ```
- This python script generates the following:
    - **centrality_result.csv**: 10 centrality measures are calculated for each node within a network.
    - **Deepwalk/Node2vec_embedding_result.csv**: Embeddings are generated for each node using Deepwalk/Node2vec techniques.
    - **neighbor_embedding_result.csv**: It calculates the average embedding vector for each node's immediate neighbors.

3. Once the features are ready, please run the following 5 codes in the `codes/estimation` folder to get peer estimation results:
- ``` 
    python codes/estimation/03.Get_baseline_results.py  
    python codes/estimation/03.Get_DML_resultss.py
    python codes/estimation/03.Get_OML_resultss.py
    ```
- ``` 
    Rscript codes/estimation/03.Get_GPSW_results.R
    ```
- ``` 
    stata -b do codes/estimation/03.Get_stepwise regression_results.do
    ```

## Replicating results in Appendix
To get Table A2, Table A3, and Figure A2 in **Appendix F**, please run the following python script
- ``` 
    python codes/robust_analysis/04.Analysis_homophily_features_and_embedding.py
    ```

To read robustness check results of Figure A3 and Figure A4 in **Appendix H**, please run the following stata script
- ``` 
    stata -b do codes/robust_analysis/05.Get_robustness_check_results.do
    ```
