import numpy as np
from pandas import ExcelWriter
import networkx as nx
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from deepwalk import graph
from deepwalk.skipgram import Skipgram
from gensim.models import Word2Vec
import node2vec
from node2vec import Node2Vec

def calculate_centrality(G):
    degree_centrality=nx.degree_centrality(G)
    eigenvector_centrality=nx.eigenvector_centrality_numpy(G)
    katz_centrality=nx.katz_centrality_numpy(G)
    closeness_centrality=nx.closeness_centrality(G)
    betweenness_centrality=nx.betweenness_centrality(G)
    harmonic_centrality=nx.harmonic_centrality(G)
    pagerank=nx.pagerank(G)
    constraint=nx.constraint(G) 
    effective_size=nx.effective_size(G)

    efficencty={n: v/G.degree(n) for n,v in effective_size.items()}

    centrality_result = pd.DataFrame(np.nan, index=list(G.nodes), columns=['degree_centrality','eigenvector_centrality',
                                                              'katz_centrality','closeness_centrality',
                                                              'betweenness_centrality','harmonic_centrality',
                                                              'pagerank','constraint','effective_size','efficency'])
    
    for i, row in centrality_result.iterrows():
        row['degree_centrality']=degree_centrality[i]
        row['eigenvector_centrality']=eigenvector_centrality[i]
        row['katz_centrality']=katz_centrality[i]
        row['closeness_centrality']=closeness_centrality[i]
        row['betweenness_centrality']=betweenness_centrality[i]
        row['harmonic_centrality']=harmonic_centrality[i]
        row['pagerank']=pagerank[i]
        row['constraint']=constraint[i]
        row['effective_size']=effective_size[i]
        row['efficency']=efficencty[i]
    
    
    centrality_result1 = preprocessing.scale(centrality_result.values)
    # C0-C9 represent: 'degree','eigenvector','katz','closeness','betweenness','harmonic','pagerank','constraint','effective_size','efficency'
    index_list = [int(i) for i in centrality_result.index]

    centrality_result2 = pd.DataFrame(centrality_result1,index = index_list, columns=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    centrality_result2.insert(0,column='node_id',value=index_list)
    centrality_result2 = centrality_result2.sort_values(by='node_id',ascending=True)

    print('Complete centrality measures calculation!')

    return centrality_result2

def get_deepwalk_embedding(input, dimensions, walk_length,num_walks,window_size,n_iter,workers,undirect_indicator):
    G = graph.load_edgelist(input, undirected=undirect_indicator)
    print("Number of nodes: {}".format(len(G.nodes())))
    print("Number of walks: {}".format(num_walks))
    data_size = num_walks * walk_length
    print("Data size (walks*length): {}".format(data_size))
    
    walks = graph.build_deepwalk_corpus(G, num_paths=num_walks,path_length=walk_length)
    
    print("Deepwalk Training...")
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=1, sg=1, hs=1, workers=workers,iter=n_iter)
    # model.wv.save_word2vec_format(output)
    embeddings = [model.wv[str(node)] for node in model.wv.index2entity]
    node_ids = [int(node) for node in model.wv.index2entity]

    # Create a DataFrame with embeddings and node IDs
    emd = pd.DataFrame(embeddings, index=node_ids)
    emd.insert(0,column='node_id',value=node_ids)
    emd.columns = ['node_id']+['E'+str(i) for i in range(0,64)] 
    emd = emd.sort_values(by='node_id',ascending=True)
    print("******* Deepwalk Complete! *******")

    return emd

def get_node2vec_embedding(input, dimensions, walk_length,num_walks,window_size,n_iter,workers,undirect_indicator,p,q):

    nx_G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1
    if undirect_indicator:
        nx_G = nx_G.to_undirected()

    node2vec = Node2Vec(nx_G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=workers)
    print("Node2vec Training...")
    model = node2vec.fit(window = window_size, min_count=1, batch_words=4,iter=n_iter)

    # model.wv.save_word2vec_format(output)
    embeddings = [model.wv[str(node)] for node in model.wv.index2entity]
    node_ids = [int(node) for node in model.wv.index2entity]

    # Create a DataFrame with embeddings and node IDs
    emd = pd.DataFrame(embeddings, index=node_ids)
    emd.insert(0,column='node_id',value=node_ids)
    emd.columns = ['node_id']+['E'+str(i) for i in range(0,64)] 
    emd = emd.sort_values(by='node_id',ascending=True)
    print("******* Node2vec Complete! *******")

    return emd

def get_avg_neighbor_embedding(G, emb):
    results = list()
    for i in list(G.nodes):
        neighbor_list = [int(n) for n in G.neighbors(i)]
        neighbor_embedding = emb[emb['node_id'].isin(neighbor_list)]
        avg_neighbor = neighbor_embedding.mean()
        results.append(avg_neighbor.tolist()[1:])
    
    node_index = list(G.nodes)

    results = pd.DataFrame(results,index=node_index)
    results.insert(0,column='node_id',value=node_index)
    results.columns = ['node_id']+['N'+str(i) for i in range(0,64)] 
    results = results.sort_values(by='node_id',ascending=True)
    return results



# Deepwalk parameter settings #
num_walks = 80 # number of walks generated starting with a specific node
walk_length =  40 # the length of each walk
window_size = 1 # the context nodes used to predict a target node, if windowsize = 41 then we use nodes with distance [-1,1] to the target node.
dimensions = 64 # the embedding for each node embedding vector
workers = 2
undirected = True
n_iter = 2

# Node2vec has two additional parameters #
p = 2
q = 0.5 

#----------------------------------------------#
#    1. Get measures for pure homophily case   #
#----------------------------------------------#

edge_folder = 'data/Final_simulated_data_pure_homophily/Simulated_networks/'
output_folder = 'data/Final_simulated_data_pure_homophily/'

for i in range(1,101):
    edge_file = edge_folder + str(i)+'_pure_homophily_beta0_edge_list_undirected.txt'
    G = nx.read_edgelist(edge_file)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    centrality_result = calculate_centrality(G)
    centrality_result.to_csv(output_folder+str(i)+'centrality_result.csv',header=True)
    # print('finish datasets', i) 

    deepwalk_emb = get_deepwalk_embedding(edge_file, dimensions, walk_length,num_walks,window_size, n_iter, workers, undirected)
    deepwalk_emb.to_csv(output_folder+str(i)+'deepwalk_embedding_result.csv', header=True)
    # get average of neighbors' embedding vectors
    deepwalk_neighbor_emb = get_avg_neighbor_embedding(G, deepwalk_emb)
    deepwalk_neighbor_emb.to_csv(output_folder+str(i)+'deepwalk_neighbor_embedding_result.csv', header=True)


    node2vec_emb = get_node2vec_embedding(edge_file, dimensions, walk_length,num_walks,window_size,n_iter,workers,undirected,p,q)
    node2vec_emb.to_csv(output_folder+str(i)+'node2vec_embedding_result.csv', header=True)
    # get average of neighbors' embedding vectors
    node2vec_neighbor_emb = get_avg_neighbor_embedding(G, node2vec_emb)
    node2vec_neighbor_emb.to_csv(output_folder+str(i)+'node2vec_neighbor_embedding_result.csv', header=True)

#----------------------------------------------------#
#    2. Get measures for positive peer effect case   #
#----------------------------------------------------#

edge_folder = 'data/Final_simulated_data_positive_peer_effect/Simulated_networks/'
output_folder = 'data/Final_simulated_data_positive_peer_effect/'

for i in range(1,101):
    edge_file = edge_folder + str(i)+'_positive_peer_beta0.2_edge_list_undirected.txt'
    G = nx.read_edgelist(edge_file)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    centrality_result = calculate_centrality(G)
    centrality_result.to_csv(output_folder+str(i)+'centrality_result.csv',header=True)
    # print('finish datasets', i) 

    deepwalk_emb = get_deepwalk_embedding(edge_file, dimensions, walk_length,num_walks,window_size, n_iter, workers, undirected)
    deepwalk_emb.to_csv(output_folder+str(i)+'deepwalk_embedding_result.csv', header=True)
    # get average of neighbors' embedding vectors
    deepwalk_neighbor_emb = get_avg_neighbor_embedding(G, deepwalk_emb)
    deepwalk_neighbor_emb.to_csv(output_folder+str(i)+'deepwalk_neighbor_embedding_result.csv', header=True)


    node2vec_emb = get_node2vec_embedding(edge_file, dimensions, walk_length,num_walks,window_size,n_iter,workers,undirected,p,q)
    node2vec_emb.to_csv(output_folder+str(i)+'node2vec_embedding_result.csv', header=True)
    # get average of neighbors' embedding vectors
    node2vec_neighbor_emb = get_avg_neighbor_embedding(G, node2vec_emb)
    node2vec_neighbor_emb.to_csv(output_folder+str(i)+'node2vec_neighbor_embedding_result.csv', header=True)
