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
library(latex2exp)
library(ggsci)

#setwd("E:\\Online Selection Sim")

#source("functions_OnSel_0116.R")
#source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF

nrep <- 500

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

# confirm H0, 6 choices among classification and regression settings-----

### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
#Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))

# generate data
data <- data_generation_regression(N=n)
Value=list(type="<=A",v=quantile(data$y,0.8))

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
Diversity_threshold <- 0.20

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

  pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)

  # raw result of DOSS
  res_DOSS.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=TRUE)

  # raw result of SAST
  res_SAST.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=FALSE)

  # raw result of LOND, LORD++, ADDIS
  res_lond.raw <- LOND(pval,alpha)
  #sum(res_lond.raw$R)
  #res_lord_plus.raw <- LORD(pval,alpha)
  #sum(res_lord_plus.raw$R)
  res_addis.raw <- ADDIS(pval,alpha)
  #sum(res_addis.raw$R)
  res_SAFFRON.raw <- SAFFRON(pval,alpha)
  #sum(res_SAFFRON.raw$R)
  res_naive.raw <- Online_selection_naive(pval,X_test_scale,Alter_test,alpha,m,N)

  dec_lond.raw <- res_lond.raw$R
  #dec_lord_plus.raw <- res_lord_plus.raw$R
  dec_addis.raw <- res_addis.raw$R
  dec_SAFFRON.raw <- res_SAFFRON.raw$R

  # calculate the decisions and the stop time for LOND, LORD++, and ADDIS
  dec_lond <- Decision_compute(dec_lond.raw,m)
  #dec_lord_plus <- Decision_compute(dec_lord_plus.raw,m)
  dec_addis <- Decision_compute(dec_addis.raw,m)
  dec_SAFFRON <- Decision_compute(dec_SAFFRON.raw,m)

  # record the stop time for selecting m samples
  t_DOSS <- res_DOSS.raw$stoptime
  t_SAST <- res_SAST.raw$stoptime
  t_naive <- res_naive.raw$stoptime

  if(!dec_lond$isdeath){
    t_lond <- dec_lond$stoptime
  }else{t_lond <- dec_lond$deathtime}



  if(!dec_addis$isdeath){
    t_addis <- dec_addis$stoptime
  }else{
    t_addis <- dec_addis$deathtime
  }

  if(!dec_SAFFRON$isdeath){
    t_SAFFRON <- dec_SAFFRON$stoptime
  }else{
    t_SAFFRON <- dec_SAFFRON$deathtime
  }

  # calculate FSP and DIV use the decision and the truth for data until time t
  x.Real <- data$y

  t_DOSS.new <- c(1:t_DOSS)
  t_SAST.new <- c(1:t_SAST)
  t_addis.new<- c(1:t_addis)
  t_naive.new <- c(1:t_naive)
  t_SAFFRON.new <- c(1:t_SAFFRON)
  t_lond.new <- c(1:t_lond)
  #t_lord_plus.new <- c(1:t_lord_plus)

  res_DOSS<-  list(FDP=TimeCovert(res_DOSS.raw$FDP,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]),
                   DIV=TimeCovert(res_DOSS.raw$DIV,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]))
  res_SAST <- list(FDP=TimeCovert(res_SAST.raw$FDP,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   DIV=TimeCovert(res_SAST.raw$DIV,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]))
  res_naive <- list(FDP=TimeCovert(res_naive.raw$FDP,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    DIV=TimeCovert(res_naive.raw$DIV,res_naive.raw$decisions[1:res_naive.raw$stoptime]))


  res_lond <- t_lond.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_lond$decisions, .x, x.Real))%>% unlist  %>% split(.,names(.))
  res_addis <- t_addis.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_addis$decisions, .x, x.Real)) %>% unlist %>% unlist  %>% split(.,names(.))


  res_SAFFRON <- t_SAFFRON.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_SAFFRON$decisions, .x, x.Real)) %>% unlist %>% unlist  %>% split(.,names(.))


  res_DOSS.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=res_DOSS.raw$decisions[1:t_DOSS],select.num = m)
  res_SAST.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=res_SAST.raw$decisions[1:t_SAST],select.num = m)

  res_naive.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=res_naive.raw$decisions[1:t_naive],select.num = m)


  #res_SAFFRON.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=dec_SAFFRON$decisions[1:t_SAFFRON],select.num = m)
  res_SAFFRON.stop <- data.frame(FDP=res_SAFFRON$FDP[t_SAFFRON],DIV=res_SAFFRON$DIV[t_SAFFRON])

  res_addis.stop <- data.frame(FDP=res_addis$FDP[t_addis],DIV=res_addis$DIV[t_addis])
  res_lond.stop <- data.frame(FDP=res_lond$FDP[t_lond],DIV=res_lond$DIV[t_lond])


  return(list(iter=iter,
              t_DOSS=t_DOSS,t_SAST=t_SAST,t_addis=t_addis,t_naive=t_naive,t_SAFFRON=t_SAFFRON,
              t_lond=t_lond,


              DOSS_FDP=res_DOSS$FDP,
              DOSS_DIV=res_DOSS$DIV,
              SAST_FDP=res_SAST$FDP,
              SAST_DIV=res_SAST$DIV,
              lond_FDP=res_lond$FDP,
              lond_DIV=res_lond$DIV,

              addis_FDP=res_addis$FDP,
              addis_DIV=res_addis$DIV,
              naive_FDP=res_naive$FDP,
              naive_DIV=res_naive$DIV,

              SAFFRON_FDP=res_SAFFRON$FDP,
              SAFFRON_DIV=res_SAFFRON$DIV,

              lond_FDP.stop=res_lond.stop$FDP,
              lond_DIV.stop=res_lond.stop$DIV,

              DOSS_FDP.stop=res_DOSS.stop$FDP,
              DOSS_DIV.stop=res_DOSS.stop$DIV,
              SAST_FDP.stop=res_SAST.stop$FDP,
              SAST_DIV.stop=res_SAST.stop$DIV,
              SAFFRON_FDP.stop=res_SAFFRON.stop$FDP,
              SAFFRON_DIV.stop=res_SAFFRON.stop$DIV,
              addis_FDP.stop=res_addis.stop$FDP,
              addis_DIV.stop=res_addis.stop$DIV,
              naive_FDP.stop=res_naive.stop$FDP,
              naive_DIV.stop=res_naive.stop$DIV
  ))
}

#res <- workerFunc(iter=1)
#attributes(res)

#trails <- seq(1:nrep)
#results <- lapply(trails, workerFunc)
#results1 <- results %>% unlist %>% split(.,names(.))

nrep <- 10
time_DOSS <- time_SAST  <- time_addis <- time_lond <- time_lord_plus <- time_SAFFRON <- time_naive <- rep(NA,nrep)
DOSS_FDP <- SAST_FDP <- addis_FDP <- naive_FDP <- lond_FDP <- lord_plus_FDP <- SAFFRON_FDP <- matrix(NA,N,nrep)
DOSS_DIV <- SAST_DIV <- addis_DIV <- naive_DIV <- lond_DIV <- lord_plus_DIV <- SAFFRON_DIV <- matrix(NA,N,nrep)

DOSS_FDP.stop <- SAST_FDP.stop <- addis_FDP.stop <- naive_FDP.stop <- lond_FDP.stop <- lord_plus_FDP.stop <- SAFFRON_FDP.stop <- rep(NA,nrep)
DOSS_DIV.stop <- SAST_DIV.stop <- addis_DIV.stop <- naive_DIV.stop <- lond_DIV.stop <- lord_plus_DIV.stop <- SAFFRON_DIV.stop <- rep(NA,nrep)


# repeat implement nrep times-----

for(i in 1:nrep){
  res <- workerFunc(iter=1)
  time_DOSS[i] <- res$t_DOSS
  time_SAST[i] <- res$t_SAST

  time_naive[i] <- res$t_naive
  time_lond[i] <- res$t_lond

  time_SAFFRON[i] <- res$t_SAFFRON
  time_addis[i] <- res$t_addis

  DOSS_FDP[1:res$t_DOSS,i] <- res$DOSS_FDP
  SAST_FDP[1:res$t_SAST,i] <- res$SAST_FDP
  naive_FDP[1:res$t_naive,i] <- res$naive_FDP

  lond_FDP[1:res$t_lond,i] <- res$lond_FDP

  SAFFRON_FDP[1:res$t_SAFFRON,i] <- res$SAFFRON_FDP
  addis_FDP[1:res$t_addis,i] <- res$addis_FDP

  DOSS_DIV[1:res$t_DOSS,i] <- res$DOSS_DIV
  SAST_DIV[1:res$t_SAST,i] <- res$SAST_DIV
  naive_DIV[1:res$t_naive,i] <- res$naive_DIV

  lond_DIV[1:res$t_lond,i] <- res$lond_DIV

  SAFFRON_DIV[1:res$t_SAFFRON,i] <- res$SAFFRON_DIV
  addis_DIV[1:res$t_addis,i] <- res$addis_DIV

  DOSS_FDP.stop[i] <- res$DOSS_FDP.stop
  SAST_FDP.stop[i] <- res$SAST_FDP.stop
  naive_FDP.stop[i] <- res$naive_FDP.stop

  lond_FDP.stop[i] <- res$lond_FDP.stop

  SAFFRON_FDP.stop[i] <- res$SAFFRON_FDP.stop
  addis_FDP.stop[i] <- res$addis_FDP.stop


  DOSS_DIV.stop[i] <- res$DOSS_DIV.stop
  SAST_DIV.stop[i] <- res$SAST_DIV.stop
  naive_DIV.stop[i] <- res$naive_DIV.stop

  lond_DIV.stop[i] <- res$lond_DIV.stop
  SAFFRON_DIV.stop[i] <- res$SAFFRON_DIV.stop
  addis_DIV.stop[i] <- res$addis_DIV.stop
}


# tidy the results when stop------
FDP.stop <- as.data.frame(cbind(DOSS_FDP.stop,SAST_FDP.stop,naive_FDP.stop,lond_FDP.stop,SAFFRON_FDP.stop,addis_FDP.stop))
DIV.stop <- as.data.frame(cbind(DOSS_DIV.stop,SAST_DIV.stop,naive_DIV.stop,lond_DIV.stop,SAFFRON_DIV.stop,addis_DIV.stop))
result_stop <- as.data.frame(cbind(FDP.stop,DIV.stop))
head(result_stop)
result_stop.ave <- colMeans(result_stop,na.rm = TRUE)

stop_time <- as.data.frame(cbind(time_DOSS,time_SAST,time_naive,time_lond,time_SAFFRON,time_addis))
stop_time.ave <- colMeans(stop_time,na.rm = TRUE)

#write.csv(result_stop,"result_stop_regression.csv")
#write.csv(stop_time,"stop_time.csv")

names(FDP.stop) <- c('DOSS','SAST','ST','LOND','SAFFRON','ADDIS')
names(DIV.stop) <- c('DOSS','SAST','ST','LOND','SAFFRON','ADDIS')
head(FDP.stop)
head(DIV.stop)

#write.csv(FDP.stop,"FDP.stop.csv")
#write.csv(DIV.stop,"DIV.stop.csv")
#write.csv(stop_time,"stop_time.csv")
Diversity_threshold
alpha

level <- as.data.frame(cbind(Diversity_threshold,alpha))
level
#write.csv(level,"level_regression.csv")

#  plots the results when stop--------

FDP_stop_value <- melt(FDP.stop)
names(FDP_stop_value) <- c('Method','FSR')

p_FSR <- ggplot(data=FDP_stop_value,aes(x=Method,y=FSR,color=Method))
p_FSR <- p_FSR+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = "FSR", x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5))
p_FSR <- p_FSR + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

ES_stop_value <- melt(DIV.stop)
names(ES_stop_value) <- c('Method','ES')
p_ES <- ggplot(data=ES_stop_value,aes(x=Method,y=ES,color=Method))
p_ES <- p_ES+geom_boxplot()+
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = "ES", x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5))
p_ES <- p_ES + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")



names(stop_time) <- c('DOSS','SAST','ST','LOND','SAFFRON','ADDIS')


stop_time_value <- melt(stop_time)
names(stop_time_value) <- c('Method','Stoptime')


p_Time <- ggplot(data=stop_time_value,aes(x=Method,y=Stoptime,color=Method))
p_Time <- p_Time+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = TeX("$T_{m}$"), x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)

p_Time <- p_Time+theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5))
p_Time

ggarrange(p_FSR, p_ES, p_Time,ncol=3, nrow=1, common.legend = TRUE, legend="bottom",
          font.label = list(size = 12, face = "bold"))






# tidy the results at every time t--------

t <- seq(30,300,10)

CI.fun <- function(x){
  x <- na.omit(x)
  y <- mean(x)+c(-1.96,1.96)*sd(x)/sqrt(length(x))
  y[which(y<0)]=0
  y[which(y>1)]=1
  return(y)
}


tidy_res.each <- function(t, Method_FDP, Method_DIV){
  FDP_Method.ave <- rowMeans(Method_FDP[t,],na.rm = TRUE)
  DIV_Method.ave <- rowMeans(Method_DIV[t,],na.rm = TRUE)
  return(data.frame(FDP.ave=FDP_Method.ave,DIV.ave=DIV_Method.ave)
  )

}



result_each <- as.data.frame(cbind(tidy_res.each(t,DOSS_FDP,DOSS_DIV),tidy_res.each(t,SAST_FDP,SAST_DIV),
                                   tidy_res.each(t,naive_FDP,naive_DIV),tidy_res.each(t,lond_FDP,lond_DIV),
                                   #tidy_res.each(t,lord_plus_FDP,lord_plus_DIV),
                                   tidy_res.each(t,SAFFRON_FDP,SAFFRON_DIV),
                                   tidy_res.each(t,addis_FDP,addis_DIV)))
dim(result_each)

#write.csv(result_each,"result_each_regression.csv")





result_FSR <- result_each[,c(1,3,5,7,9,11)]
result_ES <- result_each[,c(2,4,6,8,10,12)]
result_FSR$time <- t
result_ES$time <- t
names(result_FSR) <- c('DOSS','SAST','ST','LOND','SAFFRON','ADDIS','time')
names(result_ES) <- c('DOSS','SAST','ST','LOND','SAFFRON','ADDIS','time')
head(result_FSR)
head(result_ES)


# plots the results at every time t--------


data_FSR <- melt(result_FSR,id="time")

colnames(data_FSR) <- c("Time","Method","Value")

#data_FSR %<>% mutate(lower = data_FSR.lower$Value, higher = data_FSR.higher$Value)

p1 <- ggplot(data = data_FSR,aes(x=Time,y=Value,group =Method,color=Method,shape=Method))+
  geom_point()+
  geom_line()+
  xlab("Time")+#横坐标名???
  ylab("FSR")+#纵坐标名???+
  ylim(0,0.3)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 1)+
  theme(axis.text = element_text(size = 16),
       axis.title = element_text(size = 20),
       legend.text = element_text(size = 16),
       legend.title = element_text(size = 16))

p1 <- p1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

p1
data_ES <- melt(result_ES,id="time")
colnames(data_ES) <-  c("Time","Method","Value")

p2 <- ggplot(data = data_ES,aes(x=Time,y=Value,group=Method,shape=Method,color=Method))+
  geom_point()+
  geom_line(aes(color=Method))+
  xlab("Time")+#横坐标名???
  ylab("ES")+#纵坐标名???
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 1)+
  theme(axis.text = element_text(size = 16),
       axis.title = element_text(size = 20),
       legend.text = element_text(size = 16),
       legend.title = element_text(size = 16))

p2 <- p2 + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")

p2



ggarrange(p1, p2, ncol=1, nrow=2, common.legend = TRUE, legend="bottom", # 添加标签
          font.label = list(size = 20, face = "bold"))


# save the results---

#write.csv(result_each,"result_each_reg.csv")
#write.csv(result_FSR,"result_FSR_reg.csv")
#write.csv(result_ES,"result_ES_reg.csv")
#write.csv(FDP.stop,"FDP_stop_reg.csv")
#write.csv(DIV.stop,"DIV_stop_reg.csv")
#write.csv(stop_time,"stop_time_reg.csv")

setwd("E:\\Online Selection Sim\\Simulation results\\regression")
result_each <- read.csv("result_each_reg_NN.csv")[,-1]
result_FSR <- read.csv("result_FSR_reg_NN.csv")[,-1]
result_ES <- read.csv("result_ES_reg_NN.csv")[,-1]
FDP.stop <- read.csv("FDP_stop_reg_NN.csv")[,-1]
DIV.stop <- read.csv("DIV_stop_reg_NN.csv")[,-1]
stop_time <- read.csv("stop_time_reg_NN.csv")[,-1]


# plots---------
Diversity_threshold <- 0.015
alpha <- 0.1

level <- as.data.frame(cbind(Diversity_threshold,alpha))
level
#write.csv(level,"level_regression.csv")

#  plots the results when stop--------

FDP_stop_value <- melt(FDP.stop[,-3])
names(FDP_stop_value) <- c('Method','FSR')

p_FSR <- ggplot(data=FDP_stop_value,aes(x=Method,y=FSR,color=Method))
p_FSR <- p_FSR+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = "FSR", x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())
p_FSR <- p_FSR + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

ES_stop_value <- melt(DIV.stop[,-3])
names(ES_stop_value) <- c('Method','ES')
p_ES <- ggplot(data=ES_stop_value,aes(x=Method,y=ES,color=Method))
p_ES <- p_ES+geom_boxplot()+
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = "ES", x = "Method")+
  ylim(0,0.2)+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())
p_ES <- p_ES + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")



names(stop_time) <- c('DOSS','SAST','ST','LOND','SAFFRON','ADDIS')


stop_time_value <- melt(stop_time[,-3])
names(stop_time_value) <- c('Method','Stoptime')


p_Time <- ggplot(data=stop_time_value,aes(x=Method,y=Stoptime,color=Method))
p_Time <- p_Time+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = TeX("$T_{m}$"), x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)

p_Time <- p_Time+theme(axis.text = element_text(size = 16),
                       axis.title = element_text(size = 20),
                       axis.text.x = element_text(angle=90, hjust=1, vjust=.5),
                       panel.grid.major=element_line(colour=NA),
                       panel.background = element_rect(fill = "transparent",colour = NA),
                       plot.background = element_rect(fill = "transparent",colour = NA),
                       panel.grid.minor = element_blank())
p_Time

ggarrange(p_FSR, p_ES, p_Time,ncol=3, nrow=1, common.legend = TRUE, legend="bottom",
          font.label = list(size = 12, face = "bold"))


t <- seq(30,300,10)
result_FSR <- result_each[,c(1,3,7,9,11)]
result_ES <- result_each[,c(2,4,8,10,12)]
result_FSR$time <- t
result_ES$time <- t
names(result_FSR) <- c('DOSS','SAST','LOND','SAFFRON','ADDIS','time')
names(result_ES) <- c('DOSS','SAST','LOND','SAFFRON','ADDIS','time')
head(result_FSR)
head(result_ES)




# plots the results at every time t--------


data_FSR <- melt(result_FSR,id="time")

colnames(data_FSR) <- c("Time","Method","Value")

#data_FSR %<>% mutate(lower = data_FSR.lower$Value, higher = data_FSR.higher$Value)

p1 <- ggplot(data = data_FSR,aes(x=Time,y=Value,group =Method,color=Method,shape=Method))+
  geom_point()+
  geom_line(aes(linetype=Method,color=Method))+
  xlab("Time")+#横坐标名???
  ylab("FSR")+#纵坐标名???+
  ylim(0,0.2)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 1)+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())

p1 <- p1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

p1
data_ES <- melt(result_ES,id="time")
colnames(data_ES) <-  c("Time","Method","Value")

p2 <- ggplot(data = data_ES,aes(x=Time,y=Value,group=Method,shape=Method,color=Method))+
  geom_point()+
  geom_line(aes(linetype=Method,color=Method))+
  xlab("Time")+#横坐标名???
  ylab("ES")+#纵坐标名???
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 1)+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())

p2 <- p2 + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")

p2



ggarrange(p1, p2, ncol=1, nrow=2, common.legend = TRUE, legend="bottom", # 添加标签
          font.label = list(size = 20, face = "bold"))







