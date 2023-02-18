#import packages-----
library(DOSS)
library(MASS)
library(ks)
library(kernlab)
library(randomForest)
library(foreach)
library(reshape2)
library(ggplot2)
library(glmnet)
library(caret)
library(nnet)
library(ggpubr)
library(tidyverse)
library(kedd)
library(pbmcapply)
library(onlineFDR)
library(parallel)
library(grid)
library(gridExtra)
library(magrittr)

#setwd("E:\\Online Selection Sim")

source("functions_OnSel_0116.R")
source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF



N <- 5000 # number of total time points

# simulation setting-----
alpha <- 0.1 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 5000 # number of historical data
m <- 100 # number of selections when stop
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- 800

Diversity_constant<-1 #determine the diversity threshold
Diversity_initial_num<-1 #when rejection number exceeds this, we consider computing diversity.
algo<- new("NN-R") #algorithm used for classification or regression
lambda<- 200 #specific parameter for the algorithm


# generate data
data <- data_generation_regression(N=n)
Value=list(type="<=A",v=quantile(data$y,0.8))
#Value=list(type="==A,R",v=0)

# generate history data and estimate K (diversity threshold)---
his_data <- data_generation_regression(N=n)

p <- ncol(his_data)-1 # dimension of covariates

### some data notations, and index for null data-----
datawork=DataSplit(his_data,n,0,n_cal)
data_train=datawork$data_train

data_cal=datawork$data_cal
data_rest=datawork$data_rest

Null_cal=NullIndex(data_cal$y,Value)
Null_rest=NullIndex(data_rest$y,Value)

X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
Y_train=as.matrix(data_train$y)
X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
Y_cal=as.matrix(data_cal$y)

X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
Y_rest=as.matrix(data_rest$y)

data_test=data
Null_test=NullIndex(data_test$y,Value)
Alter_test=setdiff(1:length(data_test$y),Null_test)
X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
Y_test=as.matrix(data_test$y)

X_test_scale=scale(X_test,center = TRUE,scale = TRUE)

# model and estimating locfdr -----

model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
W_test=Pred(algo,model,X_test) #predict classfication score of test data
#estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they
#are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
#If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = FALSE)
#plot(W_test,TN)#observe the minimum value of localfdr, it should be lower than alpha

### confirm the diversity threshold

#when computing diversity, we should scale each dimension of X
X_cal_scale=scale(X_cal,center = TRUE,scale = TRUE)
X_test_scale=scale(X_test,center = TRUE,scale = TRUE)

X_cal_alter=X_cal_scale[-Null_cal,]
Diversity_Base<-diversity_true_correct_rej(X_cal_alter)
Diversity_threshold<-Diversity_Base*0.4
Diversity_threshold <- 0.015

workerFunc1 <- function(iter,n_cal=800){
  #Generate data

  data <- data_generation_regression(N=N)
  his_data <- data_generation_regression(N=n)

  ### some data notations, and index for null data-----
  datawork=DataSplit(his_data,n,0,n_cal)
  data_train=datawork$data_train

  data_cal=datawork$data_cal
  data_rest=datawork$data_rest

  Null_cal=NullIndex(data_cal$y,Value)
  Null_rest=NullIndex(data_rest$y,Value)

  X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
  Y_train=as.matrix(data_train$y)
  X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
  Y_cal=as.matrix(data_cal$y)

  X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
  Y_rest=as.matrix(data_rest$y)

  data_test=data
  Null_test=NullIndex(data_test$y,Value)
  Alter_test=setdiff(1:length(data_test$y),Null_test)
  X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
  Y_test=as.matrix(data_test$y)

  # model and estimating locfdr -----

  model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
  W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
  W_test=Pred(algo,model,X_test) #predict classfication score of test data
  #estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they
  #are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
  #If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
  TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = FALSE)
  #calculate p-values



  # raw result of DOSS
  res_DOSS.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=TRUE)
  # calculate FSP and DIV use the decision and the truth for data until time t


  res_DOSS<-  list(FDP=TimeCovert(res_DOSS.raw$FDP,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]),
                   DIV=TimeCovert(res_DOSS.raw$DIV,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]))

  t_DOSS <- res_DOSS.raw$stoptime

  return(data.frame(FDP=res_DOSS$FDP[t_DOSS],DIV=res_DOSS$DIV[t_DOSS],ncal=n_cal))
}



nrep <- 500
trails <- seq(1:nrep)
results <- lapply(trails, workerFunc1)
results1 <- results %>% unlist %>% split(.,names(.))
results1 <- as.data.frame(results1)

res1 <- colMeans(results1,na.rm=TRUE)
res1
res2 <- apply(results1,2,sd)
res2


