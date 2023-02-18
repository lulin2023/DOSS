#import packages-----

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
library(latex2exp)
library(ggsci)

setwd("E:\\Online Selection Sim")

source("functions_OnSel_mES.R")  
source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF


N <- 5000 # number of total time points

# simulation setting-----
alpha <- 0.1 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 5000 # number of historical data
m <- 100 # number of selections when stop
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- 4000

Diversity_constant<-1 #determine the diversity threshold
Diversity_initial_num<-1 #when rejection number exceeds this, we consider computing diversity.
algo<- new("NN-R") #algorithm used for classification or regression
lambda<- 200 #specific parameter for the algorithm

# generate data
data <- data_generation_regression(N=n)

# generate history data and estimate K (diversity threshold)---
his_data <- data_generation_regression(N=n)

Value=list(type="<=A",v=quantile(data$y,0.8))

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

# a worker function that runs DOSS, SAST, ST, LOND, LORD++, SAFFRON and ADDIS

workerFunc <- function(iter){
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
  
  # record the stop time for selecting m samples
  t_DOSS <- res_DOSS.raw$stoptime
  
  # calculate FSP and DIV use the decision and the truth for data until time t
  x.Real <- data$y
  
  t_DOSS.new <- c(1:t_DOSS)
  
  res_DOSS<- t_DOSS.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, res_DOSS.raw$decisions, .x, x.Real))%>% unlist  %>% split(.,names(.))
 
  res_DOSS.stop <- data.frame(FDP=res_DOSS$FDP[t_DOSS],DIV=res_DOSS$DIV[t_DOSS],Power=res_DOSS$Power[t_DOSS],
                              NS=res_DOSS$NS[t_DOSS],TS=res_DOSS$TS[t_DOSS])
 
 
  return(list(iter=iter,
              t_DOSS=t_DOSS,
              DOSS_FDP=res_DOSS$FDP,
              DOSS_DIV=res_DOSS$DIV,
              DOSS_NS=res_DOSS$NS,
              DOSS_TS=res_DOSS$TS,
              DOSS_FDP.stop=res_DOSS.stop$FDP,
              DOSS_DIV.stop=res_DOSS.stop$DIV,
              DOSS_NS.stop=res_DOSS.stop$NS,
              DOSS_TS.stop=res_DOSS.stop$TS,
              DOSS_Power.stop=res_DOSS.stop$Power
  ))
}

#res <- workerFunc(iter=1) 
#attributes(res)

#trails <- seq(1:nrep)
#results <- lapply(trails, workerFunc)
#results1 <- results %>% unlist %>% split(.,names(.))
nrep <- 100

time_DOSS <-  rep(NA,nrep)

DOSS_FDP <-  matrix(NA,N,nrep)
DOSS_DIV <-  matrix(NA,N,nrep)
DOSS_NS <- matrix(NA,N,nrep)
DOSS_TS <-  matrix(NA,N,nrep)


DOSS_FDP.stop <-  rep(NA,nrep)
DOSS_DIV.stop <-  rep(NA,nrep)
DOSS_NS.stop <- rep(NA,nrep)
DOSS_TS.stop <-  rep(NA,nrep)
DOSS_Power.stop <-  rep(NA,nrep)

# repeat implement nrep times-----

for(i in 1:nrep){
  res <- workerFunc(iter=1)
  time_DOSS[i] <- res$t_DOSS
  DOSS_FDP[1:res$t_DOSS,i] <- res$DOSS_FDP;
  DOSS_DIV[1:res$t_DOSS,i] <- res$DOSS_DIV
  DOSS_FDP.stop[i] <- res$DOSS_FDP.stop
  DOSS_DIV.stop[i] <- res$DOSS_DIV.stop
  DOSS_NS[1:res$t_DOSS,i] <- res$DOSS_NS
  DOSS_TS[1:res$t_DOSS,i] <- res$DOSS_TS
  DOSS_NS.stop[i] <- res$DOSS_NS.stop
  DOSS_TS.stop[i] <- res$DOSS_TS.stop
  DOSS_Power.stop[i] <- res$DOSS_Power.stop
 
}


# tidy the results when stop------
FDP.stop <- as.data.frame(DOSS_FDP.stop)
DIV.stop <- as.data.frame(DOSS_DIV.stop)
NS.stop <- as.data.frame(DOSS_NS.stop)
TS.stop <- as.data.frame(DOSS_TS.stop)

mES.stop <- colMeans(TS.stop)/colMeans(NS.stop)

result_stop <- as.data.frame(cbind(FDP.stop,DIV.stop))
head(result_stop)
result_stop.ave <- colMeans(result_stop,na.rm = TRUE)

stop_time <- as.data.frame(time_DOSS)
stop_time.ave <- colMeans(stop_time,na.rm = TRUE)

names(FDP.stop) <- c('DOSS')
names(DIV.stop) <- c('DOSS')

head(FDP.stop)
head(DIV.stop)

#write.csv(FDP.stop,"FDP.stop.csv")
#write.csv(DIV.stop,"DIV.stop.csv")
#write.csv(stop_time,"stop_time.csv")
Diversity_threshold
alpha

# tidy the results at every time t--------

t <- seq(50,750,10)

result_each <- as.data.frame(tidy_res.each1(t,DOSS_FDP,DOSS_DIV,DOSS_TS,DOSS_NS))


dim(result_each)


res_ES_mES <- result_each[,-1]
res_ES_mES$time <- t
names(res_ES_mES) <- c('ES','mES','Time')

# plots the results at every time t--------


p <- ggplot(res_ES_mES, aes(Time))+
  geom_line(aes(y = ES, colour = "ES"),linewidth=1.0) + 
  geom_line(aes(y = mES, colour = "mES"),linewidth=1.0)+
  theme_bw()+
  labs(y = "Value", x = "Time")+
  scale_color_nejm(palette = c("default"), alpha = 1)+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.position = "bottom",
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+
  guides(color=guide_legend(title = "Measure"))


p <- p + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")

p

#write.csv(res_ES_mES,"res_ES_mES.csv")
#res_ES_mES <- read.csv("E:\\Online Selection Sim\\Simulation results\\regression\\res_ES_mES.csv")[,-1]
