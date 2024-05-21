#-----------------------------------------Declare-------------------------------------------------------# 
# F.aac.iter function has been slightly modified on the basis of Zhu et al.(2015)                       
#
# Reference: Zhu Y, Coffman DL, Ghosh D (2015) A boosting algorithm for estimating generalized propensity scores with continuous treatments. 
#            Journal of Causal Inference 3(1):25–40.
#--------------------------------------------------------------------------------------------------# 

F.aac.iter=function(i,data,ps.model,ps.num,rep,criterion) {
  # i: number of iterations (trees)
  # data: dataset containing the treatment and the covariates
  # ps.model: the boosting model to estimate p(T_i,X_i)
  # ps.num: the estimated p(T_i)
  # rep: number of replications in bootstrap
  # criterion: the correlation metric used as the stopping criterion
  GBM.fitted=predict(ps.model,newdata=data,n.trees=floor(i),type= "response")
  
  ps.den=dnorm((data$T-GBM.fitted)/sd(data$T-GBM.fitted),0,1)
  wt=ps.num/ps.den
  
  aac_iter=rep(NA,rep)
  
  for (i in 1:rep){
    
    bo=sample(1:dim(data)[1],replace=TRUE,prob=wt)
    newsample=data[bo,]
    j.drop=match(c("T"),names(data))
    j.drop=j.drop[!is.na(j.drop)]
    x=newsample[,-j.drop]
    
    if(criterion== "spearman"| criterion== "kendall"){
      if (dim(data)[2]-1==1){
        dim(x) <- c(1, length(x))
        ac=apply(x, MARGIN=1, FUN=cor, y=newsample$T,method=criterion)
      }
      else {
        ac=apply(x, MARGIN=2, FUN=cor, y=newsample$T,method=criterion)
      }
      
    } else if (criterion== "distance"){
      if (dim(data)[2]-1==1){
        ac=apply(x, MARGIN=1, FUN=dcor, y=newsample$T)
      }
      else {
        ac=apply(x, MARGIN=2, FUN=dcor, y=newsample$T)
      }
      
      
      
    } else if (criterion== "pearson"){
      #ac=matrix(NA,dim(x)[2],1)
      ac = matrix(NA,dim(data)[2]-1,1)
      
      for (j in 1:dim(data)[2]-1){
        if (dim(data)[2]-1==1){
          #print("Only one input variable is used!")
          
          ac[j] =ifelse (!is.factor(x), cor(newsample$T, x,
                                            method=criterion),polyserial(newsample$T, x))}
        else {
          ac[j] =ifelse (!is.factor(x[,j]), cor(newsample$T, x[,j],
                                                method=criterion),polyserial(newsample$T, x[,j]))}
      }
    } else print("The criterion is not correctly specified")
    aac_iter[i]=mean(abs(1/2*log((1+ac)/(1-ac))),na.rm=TRUE)
  }
  aac=mean(aac_iter)
  return(aac)
}

#install.packages("openxlsx")
library(openxlsx)
library(readr)
library(polycor)
library(gbm)
library(survey)
library("stringr")

inputfolder = "data/Final_simulated_data_positive_peer_effect/Final_regression_feature"
outfolder = "data/Final_simulated_data_positive_peer_effect/"
file_name = '_positve_peer_effect_beta0.2_final_data.csv'
outfilename = str_c(outfolder, "Table2(c)_GPSW_estimation_results.xlsx")

#inputfolder = "data/Final_simulated_data_pure_homophily/Final_regression_feature"
#outfolder = "data/Final_simulated_data_pure_homophily/"
#file_name = '_pure_homophily_beta0_final_data.csv'
#outfilename = str_c(outfolder, "Table2(c)_GPSW_estimation_results.xlsx")

setwd(inputfolder)
N = 10
wb <- createWorkbook()

#--------------------------1. Results for no unobservable case --------------------------#
coef_list <- vector("list", N)
std_list <- vector("list", N)
pvalue_list <- vector("list", N)
sig_list <- vector("list", N)

for (i in 1:N){
  num <-toString(i)
  print(paste("new dataset" , num))
  filename2<-str_c(num,file_name)
  data <- read_csv(filename2,col_names = TRUE,show_col_types = FALSE)
  num_data <- data.frame(data.matrix(data))
  
  x = data.frame(y0 = data$y0,x1=data$x1,x2=data$x2) # No unobservable
  formula = as.formula("y1 ~ influence + y0 + x1 + x2")
  
  mydata = data.frame(T=num_data$influence,X=x)
  model.num=lm(T~1,data=mydata)
  ps.num=dnorm((mydata$T-model.num$fitted)/(summary(model.num))$sigma,0,1)
  model.den=gbm(T~.,data=mydata, shrinkage=0.0005,interaction.depth=6, distribution= "gaussian",n.trees=500)
  
  opt=optimize(F.aac.iter,interval=c(1,500), data=mydata, ps.model=model.den,
               ps.num=ps.num,rep=50,criterion= "pearson")
  # Find the optimal number of trees using Pearson/polyserial/spearman correlation
  best.aac.iter=opt$minimum
  best.aac=opt$objective
  
  # Calculate the inverse probability weights
  model.den$fitted=predict(model.den,newdata=mydata,n.trees=floor(best.aac.iter), type= "response")
  ps.den=dnorm((mydata$T-model.den$fitted)/sd(mydata$T-model.den$fitted),0,1)
  weight.gbm=ps.num/ps.den

  # Outcome analysis using survey package
  dataset=data.frame(y1=data$y1,influence=data$influence, weight.gbm, x)
  
  design.b=svydesign(ids=~1, weights=~weight.gbm, data=dataset)
  fit=svyglm(formula, design=design.b,rescale=FALSE)
  summary(fit)
  coef = summary(fit)$coefficients[2,1]
  std = summary(fit)$coefficients[2,2]
  pvalue = summary(fit)$coefficients[2,4]
  coef_list[[i]] = coef
  std_list[[i]] = std
  pvalue_list[[i]] = pvalue
  sig_list[[i]] = ifelse(pvalue < 0.05, 1, 0)
  
  #-------------------------- using trimmed data to estimated peer effect --------------------------#
  # lower_cutoff <- quantile(dataset$weight.gbm, 0.01)
  # upper_cutoff <- quantile(dataset$weight.gbm, 0.99)
  # trimmed_data <- dataset[dataset$weight.gbm >= lower_cutoff & dataset$weight.gbm <= upper_cutoff, ]
  # design.b2 =svydesign(ids=~1, weights=~weight.gbm, data=trimmed_data)
  # fit2=svyglm(formula, design=design.b2,rescale=FALSE)
  # summary(fit2)
  # coef2 = summary(fit2)$coefficients[2,1]
  # std2 = summary(fit2)$coefficients[2,2]
  # pvalue2 = summary(fit2)$coefficients[2,4]
  #-------------------------------------------------------------------------------------------------#
  }
  
est_results <- cbind(coef_list,std_list,pvalue_list,sig_list)
colnames(est_results)[1] <- "coef_ipw"
colnames(est_results)[2] <- "std_ipw"
colnames(est_results)[3] <- "pvalue_ipw"
colnames(est_results)[4] <- "#pvalue<0.05"

addWorksheet(wb, "no_unobservable")
writeData(wb, "no_unobservable", est_results,rowNames = T)
saveWorkbook(wb, outfilename, overwrite = TRUE)

#--------------------------2. Results for only observable case --------------------------#

coef_list <- vector("list", N)
std_list <- vector("list", N)
pvalue_list <- vector("list", N)
sig_list <- vector("list", N)

for (i in 1:N){
  num <-toString(i)
  print(paste("new dataset" , num))
  filename2<-str_c(num,file_name)
  data <- read_csv(filename2,col_names = TRUE,show_col_types = FALSE)
  num_data <- data.frame(data.matrix(data))
  x = data.frame(y0 = data$y0) # Only observable
  formula = as.formula("y1 ~ influence + y0")      

  mydata = data.frame(T=num_data$influence,X=x)
  model.num=lm(T~1,data=mydata)
  ps.num=dnorm((mydata$T-model.num$fitted)/(summary(model.num))$sigma,0,1)
  model.den=gbm(T~.,data=mydata, shrinkage=0.0005,interaction.depth=1, distribution= "gaussian",n.trees=500)
  
  opt=optimize(F.aac.iter,interval=c(1,500), data=mydata, ps.model=model.den, ps.num=ps.num,rep=50,criterion= "pearson")
  # Find the optimal number of trees using Pearson/polyserial/spearman correlation
  best.aac.iter=opt$minimum
  best.aac=opt$objective
  
  # Calculate the inverse probability weights
  model.den$fitted=predict(model.den,newdata=mydata,n.trees=floor(best.aac.iter), type= "response")
  ps.den=dnorm((mydata$T-model.den$fitted)/sd(mydata$T-model.den$fitted),0,1)
  weight.gbm=ps.num/ps.den
  
  # Outcome analysis using survey package
  dataset=data.frame(y1=data$y1,influence=data$influence, weight.gbm, x)
  
  design.b=svydesign(ids=~1, weights=~weight.gbm, data=dataset)
  fit=svyglm(formula, design=design.b,rescale=FALSE)
  summary(fit)
  coef = summary(fit)$coefficients[2,1]
  std = summary(fit)$coefficients[2,2]
  pvalue = summary(fit)$coefficients[2,4]
  coef_list[[i]] = coef
  std_list[[i]] = std
  pvalue_list[[i]] = pvalue
  sig_list[[i]] = ifelse(pvalue < 0.05, 1, 0)
  
}

est_results <- cbind(coef_list,std_list,pvalue_list,sig_list)
colnames(est_results)[1] <- "coef_ipw"
colnames(est_results)[2] <- "std_ipw"
colnames(est_results)[3] <- "pvalue_ipw"
colnames(est_results)[4] <- "#pvalue<0.05"

addWorksheet(wb, "only observable")
writeData(wb, "only observable", est_results,rowNames = T)
saveWorkbook(wb, outfilename, overwrite = TRUE)

#--------------------------3. Results for observables + centrality case --------------------------#

coef_list <- vector("list", N)
std_list <- vector("list", N)
pvalue_list <- vector("list", N)
sig_list <- vector("list", N)

for (i in 1:N){
  num <-toString(i)
  print(paste("new dataset" , num))
  filename2<-str_c(num,file_name)
  data <- read_csv(filename2,col_names = TRUE,show_col_types = FALSE)
  num_data <- data.frame(data.matrix(data))

  C_col_names <- paste("C", 0:9, sep = "")
  C_columns <- data[, C_col_names]
  x = data.frame(y0 = data$y0,C_columns) # Observable + Centrality
  formula = as.formula(paste("y1 ~ influence + y0 + ", paste(paste0("C", 0:9),collapse = " + ")))  
  
  mydata = data.frame(T=num_data$influence,X=x)
  model.num=lm(T~1,data=mydata)
  ps.num=dnorm((mydata$T-model.num$fitted)/(summary(model.num))$sigma,0,1)
  model.den=gbm(T~.,data=mydata, shrinkage=0.0005,interaction.depth=6, distribution= "gaussian",n.trees=500)
  
  opt=optimize(F.aac.iter,interval=c(1,500), data=mydata, ps.model=model.den,
               ps.num=ps.num,rep=50,criterion= "pearson")
  # Find the optimal number of trees using Pearson/polyserial/spearman correlation
  best.aac.iter=opt$minimum
  best.aac=opt$objective
  
  # Calculate the inverse probability weights
  model.den$fitted=predict(model.den,newdata=mydata,n.trees=floor(best.aac.iter), type= "response")
  ps.den=dnorm((mydata$T-model.den$fitted)/sd(mydata$T-model.den$fitted),0,1)
  weight.gbm=ps.num/ps.den
  
  # Outcome analysis using survey package
  dataset=data.frame(y1=data$y1,influence=data$influence, weight.gbm, x)
  
  
  design.b=svydesign(ids=~1, weights=~weight.gbm, data=dataset)
  fit=svyglm(formula, design=design.b,rescale=FALSE)
  summary(fit)
  coef = summary(fit)$coefficients[2,1]
  std = summary(fit)$coefficients[2,2]
  pvalue = summary(fit)$coefficients[2,4]
  coef_list[[i]] = coef
  std_list[[i]] = std
  pvalue_list[[i]] = pvalue
  sig_list[[i]] = ifelse(pvalue < 0.05, 1, 0)

}

est_results <- cbind(coef_list,std_list,pvalue_list,sig_list)
colnames(est_results)[1] <- "coef_ipw"
colnames(est_results)[2] <- "std_ipw"
colnames(est_results)[3] <- "pvalue_ipw"
colnames(est_results)[4] <- "#pvalue<0.05"

addWorksheet(wb, "observable+centrality")
writeData(wb, "observable+centrality", est_results,rowNames = T)
saveWorkbook(wb, outfilename, overwrite = TRUE)

#--------------------------4. Results for observables + embedding case --------------------------#

coef_list <- vector("list", N)
std_list <- vector("list", N)
pvalue_list <- vector("list", N)
sig_list <- vector("list", N)

for (i in 1:N){
  num <-toString(i)
  print(paste("new dataset" , num))
  filename2<-str_c(num,file_name)
  data <- read_csv(filename2,col_names = TRUE,show_col_types = FALSE)
  num_data <- data.frame(data.matrix(data))
  
  E_col_names <- paste("E", 0:63, sep = "")
  E_columns <- data[, E_col_names]
  x = data.frame(y0 = data$y0,E_columns)  # Observable + Embedding
  formula = as.formula(paste("y1 ~ influence + y0 + ", paste(paste0("E", 0:63),collapse = " + "))) 
  
  mydata = data.frame(T=num_data$influence,X=x)
  model.num=lm(T~1,data=mydata)
  ps.num=dnorm((mydata$T-model.num$fitted)/(summary(model.num))$sigma,0,1)
  model.den=gbm(T~.,data=mydata, shrinkage=0.0005,interaction.depth=6, distribution= "gaussian",n.trees=500)
  
  opt=optimize(F.aac.iter,interval=c(1,500), data=mydata, ps.model=model.den,
               ps.num=ps.num,rep=50,criterion= "spearman")
  # Find the optimal number of trees using Pearson/polyserial/spearman correlation
  best.aac.iter=opt$minimum
  best.aac=opt$objective
  
  # Calculate the inverse probability weights
  model.den$fitted=predict(model.den,newdata=mydata,n.trees=floor(best.aac.iter), type= "response")
  ps.den=dnorm((mydata$T-model.den$fitted)/sd(mydata$T-model.den$fitted),0,1)
  weight.gbm=ps.num/ps.den
  
  # Outcome analysis using survey package
  dataset=data.frame(y1=data$y1,influence=data$influence, weight.gbm, x)
  
  design.b=svydesign(ids=~1, weights=~weight.gbm, data=dataset)
  fit=svyglm(formula, design=design.b,rescale=FALSE)
  summary(fit)
  coef = summary(fit)$coefficients[2,1]
  std = summary(fit)$coefficients[2,2]
  pvalue = summary(fit)$coefficients[2,4]
  coef_list[[i]] = coef
  std_list[[i]] = std
  pvalue_list[[i]] = pvalue
  sig_list[[i]] = ifelse(pvalue < 0.05, 1, 0)
}

est_results <- cbind(coef_list,std_list,pvalue_list,sig_list)
colnames(est_results)[1] <- "coef_ipw"
colnames(est_results)[2] <- "std_ipw"
colnames(est_results)[3] <- "pvalue_ipw"
colnames(est_results)[4] <- "#pvalue<0.05"

addWorksheet(wb, "observable+embedding")
writeData(wb, "observable+embedding", est_results,rowNames = T)
saveWorkbook(wb, outfilename, overwrite = TRUE)

