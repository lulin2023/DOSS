# Candidate Selection Analysis

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

setwd("C:\\Users\\86139\\Desktop\\online selection\\realdata-census")

source("functions_OnSel.R")
source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF


# import the Candidate Selection data----

data0<-read.table("adult.csv",header=T,sep = ",")

data0[data0=="?"]=NA

data1<-na.omit(data0)

USindex=which(data1$native.country=="United-States")
data1=select(data1[USindex,],select=-c(native.country))


varcontinue <- c("age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week")

colname=colnames(data1)
y=as.numeric(data1$income==factor(">50K",levels=c("<=50K",">50K")))
data1 <- cbind(lapply(data1[,varcontinue],function(x) as.numeric(as.character(x))),as.data.frame(lapply(data1[,setdiff(colname,varcontinue)],function(x) factor(x))))
dummy <- dummyVars(" ~ .", data=data1[,-length(colname)])

#perform one-hot encoding on data frame
data <- data.frame(predict(dummy, newdata=data1))


data$y=y


# simulation setting-----
Number=dim(data)[1]
sr=1-sum(data$y==1)/Number
alpha <- 0.2 # significance level
p <- dim(data)[2]-1 # dimension of covariates
pi <- 0.2 # Bernoulli(pi)
n <- 2000 # number of historical data
N <- 8000 # number of total time points
m <- 100 # number of selections when stop
n_train<- round(n/2) # number of data used for training model
n_cal<- n-n_train #number of data used for estimating locfdr
Diversity_constant<-0.3 #determine the diversity threshold
Diversity_initial_num<-1 #when rejection number exceeds this, we consider computing diversity.
algo<- new("RFc") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm

Diversity_Base=0.02
Diversity_threshold<-Diversity_constant*Diversity_Base




# Random forest--------

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



workerFunc <- function(iter){

  Null=NullIndex(data$y,Value)
  Alter=setdiff(1:Number,Null)

  IndexSample=c(sample(Null,round((n+N)*sr),replace = FALSE),sample(Alter,n+N-round((n+N)*sr),replace = FALSE))
  newdata=data[sample(IndexSample,n+N,replace = FALSE),]


  datawork=DataSplit(newdata,n+N,N,n_cal,n)


  ### some data notations, and index for null data-----
  data_train=datawork$data_train

  data_cal=datawork$data_cal
  data_rest=datawork$data_rest
  data_test=datawork$data_test

  Null_cal=NullIndex(data_cal$y,Value)
  Null_rest=NullIndex(data_rest$y,Value)

  X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
  Y_train=as.matrix(data_train$y)
  X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
  Y_cal=as.matrix(data_cal$y)

  X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
  Y_rest=as.matrix(data_rest$y)

  Null_test=NullIndex(data_test$y,Value)
  Alter_test=setdiff(1:length(data_test$y),Null_test)
  X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
  Y_test=as.matrix(data_test$y)

  X_test_scale=scale(X_test,center = TRUE,scale = TRUE)
  X_test_scale[which(is.na(X_test_scale))]=0
  X_test_scale[,7:13]=X_test_scale[,7:13]/7
  X_test_scale[,14:29]=X_test_scale[,14:29]/16
  X_test_scale[,30:36]=X_test_scale[,30:36]/7
  X_test_scale[,37:50]=X_test_scale[,37:50]/14
  X_test_scale[,51:56]=X_test_scale[,51:56]/6
  X_test_scale[,57:61]=X_test_scale[,57:61]/5
  X_test_scale[,62:63]=X_test_scale[,62:63]/2
  # model and estimating locfdr -----

  model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
  W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
  W_test=Pred(algo,model,X_test) #predict classfication score of test data
  #estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they
  #are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
  #If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
  TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = TRUE)
  plot(W_test,TN)
  #calculate p-values

  pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)

  # raw result of DOSS
  res_DOSS.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=TRUE)

  # raw result of SAST
  res_SAST.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=FALSE)

  # raw result of LOND, LORD++, ADDIS
  #sum(res_lond.raw$R)
  #res_lord_plus.raw <- LORD(pval,alpha)

  #sum(res_SAFFRON.raw$R)
  res_naive.raw <- Online_selection_naive(pval,X_test_scale,Alter_test,alpha,m,N)




  # record the stop time for selecting m samples
  t_DOSS <- res_DOSS.raw$stoptime
  t_SAST <- res_SAST.raw$stoptime
  t_naive <- res_naive.raw$stoptime



  # calculate FSP and DIV use the decision and the truth for data until time t
  x.Real <- data$y

  t_DOSS.new <- c(1:t_DOSS)
  t_SAST.new <- c(1:t_SAST)
  t_naive.new <- c(1:t_naive)

  #t_lord_plus.new <- c(1:t_lord_plus)

  res_DOSS<-  list(FDP=TimeCovert(res_DOSS.raw$FDP,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]),
                   DIV=TimeCovert(res_DOSS.raw$DIV,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]),
                   Power=TimeCovert(res_DOSS.raw$Power,res_DOSS.raw$decisions[1:res_DOSS.raw$stoptime]))
  res_SAST <- list(FDP=TimeCovert(res_SAST.raw$FDP,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   DIV=TimeCovert(res_SAST.raw$DIV,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   Power=TimeCovert(res_SAST.raw$Power,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]))
  res_naive <- list(FDP=TimeCovert(res_naive.raw$FDP,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    DIV=TimeCovert(res_naive.raw$DIV,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    Power=TimeCovert(res_naive.raw$Power,res_naive.raw$decisions[1:res_naive.raw$stoptime]))



  res_DOSS.stop <- data.frame(FDP=res_DOSS$FDP[t_DOSS],DIV=res_DOSS$DIV[t_DOSS],Power=res_DOSS$Power[t_DOSS])
  res_SAST.stop <- data.frame(FDP=res_SAST$FDP[t_SAST],DIV=res_SAST$DIV[t_SAST],Power=res_SAST$Power[t_SAST])
  res_naive.stop <- data.frame(FDP=res_naive$FDP[t_naive],DIV=res_naive$DIV[t_naive],Power=res_naive$Power[t_naive])

  #res_lord_plus.stop <- data.frame(FDP=res_lord_plus$FDP[t_lord_plus],DIV=res_lord_plus$DIV[t_lord_plus],Power=res_lord_plus$Power[t_lord_plus])

  #tidy the final results
  res_DOSS_final <- list(method='DOSS',t_stop=t_DOSS,FDP=res_DOSS$FDP,DIV=res_DOSS$DIV,FDP.stop=res_DOSS.stop$FDP,DIV.stop=res_DOSS.stop$DIV)
  res_SAST_final <- list(method='SAST',t_stop=t_SAST,FDP=res_SAST$FDP,DIV=res_SAST$DIV,FDP.stop=res_SAST.stop$FDP,DIV.stop=res_SAST.stop$DIV)
  res_ST_final <- list(method='ST',t_stop=t_naive,FDP=res_naive$FDP,DIV=res_naive$DIV,FDP.stop=res_naive.stop$FDP,DIV.stop=res_naive.stop$DIV)


  ClassData=data1[rownames(data_test),]
  DOSS_X=ClassData[intersect(which(res_DOSS.raw$decisions==1),Alter_test),]
  SAST_X=ClassData[intersect(which(res_SAST.raw$decisions==1),Alter_test),]
  naive_X=ClassData[intersect(which(res_naive.raw$decisions==1),Alter_test),]
  #  return(list(iter=iter,
  #              result_DOSS=res_DOSS_final,
  #              result_SAST=res_SAST_final,
  #              result_ST=res_ST_final,
  #              result_LOND=res_LOND_final,
  #              result_LORD_plus=res_LORD_plus_final,
  #             result_SAFFRON=res_SAFFRON_final,
  #              result_ADDIS=res_ADDIS_final
  #  ))
  return(list(iter=iter,
              t_DOSS=t_DOSS,t_SAST=t_SAST,t_naive=t_naive,
              DOSS_FDP=res_DOSS$FDP,
              DOSS_DIV=res_DOSS$DIV,
              SAST_FDP=res_SAST$FDP,
              SAST_DIV=res_SAST$DIV,

              naive_FDP=res_naive$FDP,
              naive_DIV=res_naive$DIV,


              DOSS_FDP.stop=res_DOSS.stop$FDP,
              DOSS_DIV.stop=res_DOSS.stop$DIV,
              DOSS_Power.stop=res_DOSS.stop$Power,

              SAST_FDP.stop=res_SAST.stop$FDP,
              SAST_DIV.stop=res_SAST.stop$DIV,
              SAST_Power.stop=res_SAST.stop$Power,


              naive_FDP.stop=res_naive.stop$FDP,
              naive_DIV.stop=res_naive.stop$DIV,
              naive_Power.stop=res_naive.stop$Power,

              DOSS_X=DOSS_X,
              SAST_X=SAST_X,
              naive_X=naive_X
  ))
}

#res <- workerFunc(iter=1)
#attributes(res)

#trails <- seq(1:nrep)
#results <- lapply(trails, workerFunc)
#results1 <- results %>% unlist %>% split(.,names(.))

nrep <- 500
time_DOSS <- time_SAST  <- time_addis <- time_lond <- time_lord_plus <- time_SAFFRON <- time_naive <- rep(NA,nrep)
DOSS_FDP <- SAST_FDP <- addis_FDP <- naive_FDP <- lond_FDP <- lord_plus_FDP <- SAFFRON_FDP <- matrix(NA,N,nrep)
DOSS_DIV <- SAST_DIV <- addis_DIV <- naive_DIV <- lond_DIV <- lord_plus_DIV <- SAFFRON_DIV <- matrix(NA,N,nrep)

DOSS_FDP.stop <- SAST_FDP.stop <- addis_FDP.stop <- naive_FDP.stop <- lond_FDP.stop <- lord_plus_FDP.stop <- SAFFRON_FDP.stop <- rep(NA,nrep)
DOSS_DIV.stop <- SAST_DIV.stop <- addis_DIV.stop <- naive_DIV.stop <- lond_DIV.stop <- lord_plus_DIV.stop <- SAFFRON_DIV.stop <- rep(NA,nrep)
DOSS_Power.stop <- SAST_Power.stop <- addis_Power.stop <- naive_Power.stop <- lond_Power.stop <- lord_plus_Power.stop <- SAFFRON_Power.stop <- rep(NA,nrep)

reslist<-list()
# repeat implement nrep times-----

for(i in 1:nrep){
  res <- workerFunc(iter=1)
  time_DOSS[i] <- res$t_DOSS
  time_SAST[i] <- res$t_SAST
  time_naive[i] <- res$t_naive


  reslist=append(reslist,list(res=res))
  DOSS_FDP[1:res$t_DOSS,i] <- res$DOSS_FDP
  SAST_FDP[1:res$t_SAST,i] <- res$SAST_FDP
  naive_FDP[1:res$t_naive,i] <- res$naive_FDP



  DOSS_DIV[1:res$t_DOSS,i] <- res$DOSS_DIV
  SAST_DIV[1:res$t_SAST,i] <- res$SAST_DIV
  naive_DIV[1:res$t_naive,i] <- res$naive_DIV



  DOSS_FDP.stop[i] <- res$DOSS_FDP.stop
  SAST_FDP.stop[i] <- res$SAST_FDP.stop
  naive_FDP.stop[i] <- res$naive_FDP.stop



  DOSS_DIV.stop[i] <- res$DOSS_DIV.stop
  SAST_DIV.stop[i] <- res$SAST_DIV.stop
  naive_DIV.stop[i] <- res$naive_DIV.stop



  DOSS_Power.stop[i] <- res$DOSS_Power.stop
  SAST_Power.stop[i] <- res$SAST_Power.stop
  naive_Power.stop[i] <- res$naive_Power.stop

}


# tidy the results when stop------
FDP.stop <- as.data.frame(cbind(DOSS_FDP.stop,SAST_FDP.stop,naive_FDP.stop))
DIV.stop <- as.data.frame(cbind(DOSS_DIV.stop,SAST_DIV.stop,naive_DIV.stop))
Power.stop <- as.data.frame(cbind(DOSS_Power.stop,SAST_Power.stop,naive_Power.stop))
result_stop <- as.data.frame(cbind(FDP.stop,DIV.stop,Power.stop))
head(result_stop)
result_stop.ave <- colMeans(result_stop,na.rm = TRUE)

stop_time <- as.data.frame(cbind(time_DOSS,time_SAST,time_naive))
stop_time.ave <- colMeans(stop_time,na.rm = TRUE)

names(FDP.stop) <- c('DOSS','SAST','ST')
names(DIV.stop) <- c('DOSS','SAST','ST')
names(Power.stop) <- c('DOSS','SAST','ST')



### output table results and figure

#table results
apply(stop_time, 2, mean)
apply(FDP.stop, 2, mean)
apply(DIV.stop, 2, mean)*1000

# female proportion in table
sexdata=data.frame()
for (res in reslist) {
  if(dim(res$DOSS_X)[1]!=0){
    aa=count(res$DOSS_X,sex)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="DOSS"
    aa=melt(aa,measure.vars="sex",value.name = "sex")
    sexdata=rbind(sexdata,aa)
  }


  if(dim(res$SAST_X[1]!=0)){
    aa=count(res$SAST_X,sex)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="SAST"
    aa=melt(aa,measure.vars="sex",value.name = "sex")
    sexdata=rbind(sexdata,aa)
  }
  aa=count(res$naive_X,sex)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="ST"
  aa=melt(aa,measure.vars="sex",value.name = "sex")
  sexdata=rbind(sexdata,aa)
}

pp4=sexdata%>%
  group_by(sex,method)%>%
  dplyr::summarize(n=mean(n),prop=mean(prop))
pp4[pp4$sex=="Female",c(1,2,4)]


# Minority proportion in table

racedata=data.frame()
for (res in reslist) {
  if(dim(res$DOSS_X)[1]!=0){
    aa=count(res$DOSS_X,race)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="DOSS"
    aa=melt(aa,measure.vars="race",value.name = "race")
    racedata=rbind(racedata,aa)
  }


  if(dim(res$SAST_X[1]!=0)){
    aa=count(res$SAST_X,race)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="SAST"
    aa=melt(aa,measure.vars="race",value.name = "race")
    racedata=rbind(racedata,aa)
  }


  aa=count(res$naive_X,race)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="ST"
  aa=melt(aa,measure.vars="race",value.name = "race")
  racedata=rbind(racedata,aa)

}

pp2=racedata%>%
  group_by(race,method)%>%
  dplyr::summarize(n=mean(n),prop=mean(prop))

pp3=pp2[pp2$race=="White",c(1,2,4)]
pp3$race="Minority"
pp3$prop=1-pp3$prop
pp3



## figure for eduaction length

edudata=data.frame()
for (res in reslist) {
  if(dim(res$DOSS_X)[1]!=0){
    aa=count(res$DOSS_X,education.num)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="DOSS"
    aa=melt(aa,measure.vars="education.num",value.name = "education")
    edudata=rbind(edudata,aa)
  }


  if(dim(res$SAST_X[1]!=0)){
    aa=count(res$SAST_X,education.num)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="SAST"
    aa=melt(aa,measure.vars="education.num",value.name = "education")
    edudata=rbind(edudata,aa)
  }

}

pp5=edudata%>%
  group_by(education,method)%>%
  dplyr::summarize(n=mean(n),prop=mean(prop))
pp5

pp5["label"]=round(pp5$prop,2)
pp5$label[pp5$method=="SAST"]=NA
P3<-ggplot(data = pp5, aes(x = education, y = prop,fill=method)) +
  geom_bar(stat = "identity",position = position_dodge(0.9),alpha=1)+
  scale_fill_nejm(palette = c("default"),name="Method")+
  scale_y_continuous(name = "Porpotion")+
  scale_x_continuous(name = "Education Length") +
  geom_text(mapping = aes(label = pp5$label,y=prop+0.02),size = 2.9, colour = 'black', vjust = 1, hjust = .5, position = position_dodge(0.9))+
  theme_classic()+
  theme(legend.position="top",legend.text=element_text(size=14),legend.title=element_text(size=13),axis.text.y=element_text(size=13),
        axis.text.x = element_text(size=13,angle = 0,hjust=0.5),axis.title.x = element_text(size=13),axis.title.y = element_text(size=13))
P3

ggsave('income.pdf',width=7,height=4)
pdf('income.pdf',width=7,height=4)
P3
dev.off()
