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
library(latex2exp)

#setwd("C:\\Users\\86139\\Desktop\\online selection\\simu-112")

#source("functions_OnSel.R")
#source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF


data_generation_classication1 <- function(N=5000,mu1= c(0,0,0,0),mu2=c(-3,-3,0,0),p=4,propotion=0.2,pi=0.2){
  Y <- rbinom(n=2*N, size=1, prob=pi)
  ident <- diag(p)
  X0 <- mvrnorm(n=2*N,mu1,Sigma=ident)
  X1 <- mvrnorm(n=2*N,mu2,Sigma=ident)
  id <- c(1:(2*N))
  id_0 <- id[which(Y==0)]
  id_1 <- id[which(Y==1)]
  train_0=sample(id_0,N*(1-propotion))
  train_1=sample(id_1,N*propotion)
  data <- matrix(NA, nrow = 2*N, ncol = p+1)
  data[,p+1] <- Y
  data[train_0,-(p+1)] <- X0[train_0,]
  data[train_1,-(p+1)] <- X1[train_1,]
  data <- as.data.frame(data)
  names(data)[5] <- "y"
  data1 <- data[complete.cases(data),]
  return(data=data1)
}


nrep <- 500

N <- 5000 # number of total time points

# simulation setting-----
alpha <- 0.1 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 1000 # number of historical data
m <- 100 # number of selections when stop
n_train<- round(n/2) # number of data used for training model
n_cal<- n-n_train #number of data used for estimating locfdr
Diversity_constant<-1 #determine the diversity threshold
Diversity_initial_num<-1 #when rejection number exceeds this, we consider computing diversity.
algo<- new("RFc") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm

# confirm H0, 6 choices among classification and regression settings-----

### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))

# generate data
data <- data_generation_classication1(N=n)

# generate history data and estimate K (diversity threshold)---
his_data <- data_generation_classication1(N=n)

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
TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = TRUE)
#plot(W_test,TN)#observe the minimum value of localfdr, it should be lower than alpha

### confirm the diversity threshold

#when computing diversity, we should scale each dimension of X
X_cal_scale=scale(X_cal,center = TRUE,scale = TRUE)
X_test_scale=scale(X_test,center = TRUE,scale = TRUE)

X_cal_alter=X_cal_scale[-Null_cal,]
Diversity_Base<-diversity_true_correct_rej(X_cal_alter)
Diversity_threshold<-Diversity_Base*0.3


######   #####   #####
data <- data_generation_classication1(N=N)
his_data <- data_generation_classication1(N=n)


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
TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = TRUE)
#calculate p-values

pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)

# raw result of DOSS
res_DOSS.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=TRUE)

# raw result of SAST
res_SAST.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=FALSE)


Sampled=which(res_DOSS.raw$decisions==1)



Type3=rep("Falsely-selected",N)
Type3[intersect(Sampled,Alter_test)]="Correctly-selected"
dataP1=data.frame(X_test1=X_test[Sampled,1],X_test2=X_test[Sampled,2],Types=Type3[Sampled])
dataP1$Types=factor(dataP1$Types,levels = c("Falsely-selected","Correctly-selected"))

P1<-dataP1 %>%
  ##x,y
  ggplot(aes(x =X_test1, y = X_test2, color=Types,size=Types,shape=Types)) +
  geom_point()+ggtitle("DOSS") +
  scale_y_continuous(name=TeX("$X_{2}$"),position="left",limits=c(-6.34,1))+scale_x_continuous(name=TeX("$X_{1}$"),limits=c(-5.57,1.6))+
  scale_color_manual(values=c("red","darkgreen"),name = "",
                     labels = c("Falsely-selected","Correctly-selected"))+
  scale_size_manual(values=c(1.5,1.5),labels = c("Falsely-selected","Correctly-selected"))+
  scale_shape_manual(values = c(17,16), name = "",labels = c("Falsely-selected","Correctly-selected"))+
  theme(plot.title = element_text(color = 'black', hjust = 0.5),
        legend.position="top",legend.spacing.x = unit(0.4, 'cm'),legend.text=element_text(size=20))+
  guides(size="none", color = guide_legend(reverse=TRUE), shape = guide_legend(reverse=TRUE))+
  theme_bw()+
  theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5),legend.position="top",legend.text = element_text(size=12))
P1

Sampled=which(res_SAST.raw$decisions==1)


Type3=rep("Falsely-selected",N)
Type3[intersect(Sampled,Alter_test)]="Correctly-selected"
dataP2=data.frame(X_test1=X_test[Sampled,1],X_test2=X_test[Sampled,2],Types=Type3[Sampled])
dataP2$Types=factor(dataP2$Types,levels = c("Falsely-selected","Correctly-selected"))


P2<-dataP2 %>%
  ##x,y
  ggplot(aes(x =X_test1, y = X_test2, color=Types,size=Types,shape=Types)) +
  geom_point()+ggtitle("SAST") +
  scale_y_continuous(name=TeX("$X_{2}$"),position="left",limits=c(-6.34,1))+scale_x_continuous(name=TeX("$X_{1}$"),limits=c(-5.57,1.6))+
  scale_color_manual(values=c("red","darkgreen"),name = "",
                     labels = c("Falsely-selected","Correctly-selected"))+
  scale_size_manual(values=c(1.5,1.5),labels = c("Falsely-selected","Correctly-selected"))+
  scale_shape_manual(values = c(17,16), name = "",labels = c("Falsely-selected","Correctly-selected"))+
  theme(plot.title = element_text(color = 'black', hjust = 0.5),
        legend.position="top",legend.spacing.x = unit(0.4, 'cm'),legend.text=element_text(size=20))+
  guides(size="none", color = guide_legend(reverse=TRUE), shape = guide_legend(reverse=TRUE))+
  theme_bw()+
  theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5),legend.position="top",legend.text = element_text(size=12))
P2

PP=ggarrange(P1,P2,common.legend = TRUE)
PP


res_DOSS.raw$FDP[m]
res_SAST.raw$FDP[m]


ggsave('Scatter-illustration.pdf',width=7,height=4)
pdf('Scatter-illustration.pdf',width=7,height=4)
PP
dev.off()
