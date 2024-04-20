#-----------------------------------------Specification--------------------------------------------# 
#- **Network Size**: 1000 nodes                                                                  
#- **Node Features**: Each node with two latent features.
#- **Pure Homophily Effect**: Nodes connect with others who share similar latent features.
#- **Positive Peer Effect**: A positive peer effect is introduced with a beta coefficient of 0.2.
#--------------------------------------------------------------------------------------------------# 

#--------------------------------------------Declare-----------------------------------------------# 
# The network.sim function is slightly modified on the basis of Shalizi and Thomas (2011) 
# Reference: Shalizi, C. R., & Thomas, A. C. (2011). Homophily and contagion are generically confounded in observational social network studies. 
#           Sociological methods & research, 40(2), 211-239.
#--------------------------------------------------------------------------------------------------# 

network.sim <- function(num.nodes=1000, scale=4, offset=-1, auto_correlation=0.1, peer_beta=0,time.trend=2,
                           y.noise=0.5, response="linear", nominate.by="distance") {
  invlogit <- function(cc) exp(cc)/(1+exp(cc))
  # Generate latent features by normal distribution
  x1 <- rnorm(num.nodes,0,1)
  x2 <- rnorm(num.nodes,0,1)
  x_all <- cbind(x1,x2)
  distances <- as.matrix(dist(x_all,method = "euclidean", diag=TRUE,upper=TRUE))
  
  # Simulate undirected networks and save edge list
  network_mat <- array(0, dim(distances))
  for (ii in 1:num.nodes){
    for(jj in ii:num.nodes){
      prob <- invlogit(offset-scale*distances[ii,jj])
      network_mat[ii,jj]<-rbinom(1,1,prob)
    }
  } 
  diag(network_mat) <- 0
  network_mat[lower.tri(network_mat)] <- t(network_mat)[lower.tri(network_mat)]
  total_neighbor<-rowSums(network_mat)
  
  edges <- data.frame(col1 = numeric(), col2 = numeric())
  for (i in 1:999) {
    for (j in (i+1):1000) {
      if (network_mat[i, j] == 1) {
        edge  <- c(i-1, j-1)
        edges <- rbind(edges, edge)
      }
    }
  }

  
  
  # Simulate outcome at period t = 0 
  y0 = (x1)^3+(x2)^3+rnorm(num.nodes,0,y.noise)

  # Simulate peer influence variable, which is the average of friends’ outcomes at preceding period
  influence <-(network_mat%*%y0)/total_neighbor
  
  # Simulate outcome at period t = 1
  y1 = auto_correlation*y0 + time.trend*cos(x1)+time.trend*sin(x2) + peer_beta*influence +rnorm(num.nodes,0,y.noise)
  
  # Calculate instrument: average y0 of two-degree friends but not one-degree friends
  net_squared <- network_mat %*% network_mat
  diag_net_squared <- diag(net_squared)
  net_squared <- net_squared - diag(diag(net_squared)) - network_mat
  # Get nodes with two-degree friends
  friends_2d <- which(diag_net_squared == 2)
  # For each node, get two-degree friends that are not one-degree friends
  friends_2d_not_1d <- lapply(1:nrow(network_mat), function(node) {
    friends_2d_for_node <- which(net_squared[node, ] == 1)
    friends_1d_for_node <- which(network_mat[node, ] == 1)
    friends_2d_not_1d_for_node <- setdiff(friends_2d_for_node, friends_1d_for_node)
    return(friends_2d_not_1d_for_node)
  })
  iv_avg_nn <- sapply(friends_2d_not_1d, function(friends) {
    if (length(friends) == 0) {
      return(NaN)
    } else {
      return(mean(y0[friends]))
    }
  })
  
  
  node_id <- seq(0, 999)
  feature_output <- cbind(node_id, x1, x2, y0, y1, influence, total_neighbor,iv_avg_nn)
  colnames(feature_output)[4] <- "y0"
  colnames(feature_output)[5] <- "y1"
  colnames(feature_output)[6] <- "influence"
  
  output<- list(edges, feature_output)
  return(output)
}


install.packages("stringr")
library("stringr")

#-----------------------------------#
#  1. Simulate pure homophily case  #
#         True beta = 0             #
#-----------------------------------#
filepathname = "data/Final_simulated_data_pure_homophily"
setwd(filepathname)

for (i in 1:100) {
  num <-toString(i)
  result_1 <- network.sim(num.nodes=1000, scale=4, offset=-1, auto_correlation=0.1, peer_beta=0, time.trend=2)
  edges<-result_1[1]
  filename1<-str_c("Simulated_networks/",num,"_pure_homophily_beta0_edge_list_undirected.txt")
  write.table(edges, filename1, sep = "\t", row.names = FALSE,col.names= FALSE)
  
  simulated_features<-result_1[2]
  filename2<-str_c("Final_regression_features/",num,"_pure_homophily_beta0_features.csv")
  write.csv(simulated_features, file=filename2, row.names = T)
}


#----------------------------------------#
# 2. Simulate positive peer effect case  #
#             True beta = 0.2            #
#----------------------------------------#
filepathname = "data/Final_simulated_data_pure_homophily"
setwd(filepathname)

for (i in 1:100) {
  num <-toString(i)
  result_1 <- network.sim(num.nodes=1000, scale=4, offset=-1, auto_correlation=0.1, peer_beta=0.2, time.trend=2)
  edges<-result_1[1]
  filename1<-str_c("Simulated_networks/",num,"_positive_peer_beta0.2_edge_list_undirected.txt")
  write.table(edges, filename1,  sep = "\t",row.names = FALSE,col.names= FALSE)
  
  simulated_features<-result_1[2]
  filename2<-str_c("Final_regression_features/",num,"_positive_peer_beta0.2_features.csv")
  write.csv(simulated_features, file=filename2, row.names = T)
}

