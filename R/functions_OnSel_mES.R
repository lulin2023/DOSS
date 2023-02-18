data_generation_classication1 <- function(N=5000,mu1= c(5,0,0,0),mu2=c(0,0,-3,-2),p=4,propotion=0.2,pi=0.2){
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



data_generation_regression<- function(N=5000){
  X1 <- rnorm(n=N,0,1)
  X2 <- rnorm(n=N,0,1)
  X3 <- rnorm(n=N,0,1)
  X4 <- rnorm(n=N,0,1)
  epsilon <- rnorm(n=N,0,1)
  Y <- -7*X1^2+5*exp(X2)+10*(X3+X4)^2+epsilon
  data <- cbind(X1,X2,X3,X4,Y)
  data <- as.data.frame(data)
  names(data)[5] <- "y"
  return(data=data)
}


# data <- data_generation_regression()
# hist(data$y)
# sum(data$y>quantile(data$y,0.8))


DataSplit<-function(data,n,n_test,n_cal,n_rest)
{
  if(n_test>0)
  {  index_test=sample(1:n,n_test,replace=FALSE)
  data_test=data[index_test,]
  data_rest2=data[-index_test,]
  data_rest=data_rest2[sample(1:dim(data_rest2)[1],n_rest),]
  index_cal=sample(1:dim(data_rest)[1],n_cal)
  data_train=data_rest[-index_cal,]
  data_cal=data_rest[index_cal,]
  return(list(data_train=data_train,data_cal=data_cal,data_test=data_test,data_rest=data_rest))}else
  {
    data_rest=data[sample(1:dim(data)[1],n_rest,replace=FALSE),]
    index_cal=sample(1:dim(data_rest)[1],n_cal)
    data_train=data_rest[-index_cal,]
    data_cal=data_rest[index_cal,]
    return(list(data_train=data_train,data_cal=data_cal,data_test=0,data_rest=data_rest))
  }

}

confomalPvalue<-function(W_cal,W_test,Null_cal,Value)
{
  Phi_cal=-ScoreCompute(W_cal,Value)
  Phi_test=-ScoreCompute(W_test,Value)
  Phi_Null=Phi_cal[Null_cal]
  n2=length(Phi_Null)
  pvalue=sapply(Phi_test,function(t){
    (sum(Phi_Null<=t)+1)/(n2+1)
  })
  return(pvalue)
}

ScoreCompute<-function(pred,Value)
{
  if(Value$type=="==A,S")
  {Phi=-pred}else if(Value$type=="==A,R")
  {Phi=pred}else if(Value$type=="<=A")
  {Phi=pred}else if(Value$type==">=B")
  {Phi=-pred}else if(Value$type=="<=A|>=B")
  {Phi=pmin(pred-Value$v[1],Value$v[2]-pred)}else if(Value$type==">=A&<=B")
  {Phi=pmax(Value$v[1]-pred,pred-Value$v[2])}
  return(Phi)
}



NullIndex<-function(y,Value)
{
  if(Value$type=="==A,S"|Value$type=="==A,R")
  {index=which(y==Value$v)}else if(Value$type=="<=A")
  {index=which(y<=Value$v)}else if(Value$type==">=B")
  {index=which(y>=Value$v)}else if(Value$type=="<=A|>=B")
  {index=which(y<=Value$v[1]|y>=Value$v[2])}else if(Value$type==">=A&<=B")
  {index=which(y>=Value$v[1]&y<=Value$v[2])}
  return(index)
}



LocalFDRcompute<-function(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame=FALSE,IsCali=FALSE)
{
  if(h1==0){h1=density(W_cal[Null_cal])$bw}
  if(h2==0){h2=density(W_cal)$bw}
  if(IsSame){h1=h2}
  f0=kde(W_cal[Null_cal],h=h1,eval.points =W_test)
  f=kde(W_cal,h=h2,eval.points =W_test)
  phat=1-length(Null_cal)/length(W_cal)
  TN=(1-phat)*f0$estimate/f$estimate
  TN[which(TN>1)]<-1
  TN[which(TN<0)]<-0
  #TN[f$estimate<0.2]=1
  #plot(W_test,TN)
  if(IsCali){
    Ind=order(W_test)
    TNW=TN[Ind]
    for (i in (length(TNW)-20):1) {
      if(TNW[i]<TNW[i+1]){TNW[i]=TNW[i+1]}
    }
    TN[Ind]=TNW
  }

  return(TN)
}



KfoldSemiDensity<-function(X_rest,Y_rest,X_test,Null_rest,W_test_whole,K,lambda,algo,h1=0,h2=0,IsSame=FALSE,IsCali=FALSE)
{
  Folds=createFolds(1:length(Y_rest),K)
  m=length(W_test_whole)
  W_cross=rep(0,length(Y_rest))
  for (i in 1:K) {
    model=fitting(algo,X_rest[-Folds[[i]],],Y_rest[-Folds[[i]]],lambda = lambda)
    lens=length(Folds[[i]])
    if (lens==1)
    {W_cross[Folds[[i]]]=Pred(algo,model,t(X_rest[Folds[[i]],]),lens)
    }else
    {W_cross[Folds[[i]]]=Pred(algo,model,X_rest[Folds[[i]],],lens) }
  }
  TN_cross=LocalFDRcompute(W_cross,W_test_whole,Null_rest,algo,h1,h2,IsSame,IsCali)

  return(TN_cross)
}

L2.norm<-function(x){
  return(sum(x^2))
}

g.function<-function(Xi,Xj)
{
  return(exp(-(L2.norm(Xi-Xj))))
}



diversity_true_correct_rej <- function(X_cal_alter){
## calculate the true empirical diversity
## here we do not use local fdr since we know the true theta
## offline paradigm
  if(nrow(X_cal_alter)>2){
  M <- matrix(0,nrow(X_cal_alter),nrow(X_cal_alter))
  for(i in 2:nrow(X_cal_alter)){
    #M[i,i]<- g.function(X_cal_alter[i,],X_cal_alter[i,])
    for(j in 1:(i-1)){
      M[i,j] <- g.function(X_cal_alter[i,],X_cal_alter[j,])
      M[j,i] <- M[i,j]
    }
  }
  #diversity <- sum(M)/(nrow(X_cal_alter)*(nrow(X_cal_alter)))
  diversity <- sum(M)/(nrow(X_cal_alter)*(nrow(X_cal_alter)-1))
  }else{
    diversity=0
  }
  return(diversity=diversity)
}


NS_true_correct_rej <- function(X_cal_alter){
  NS <- nrow(X_cal_alter)*(nrow(X_cal_alter)-1)
  return(NS=NS)
}

TS_true_correct_rej <- function(X_cal_alter){
  if(nrow(X_cal_alter)>2){
  M <- matrix(0,nrow(X_cal_alter),nrow(X_cal_alter))
  for(i in 2:nrow(X_cal_alter)){
    #M[i,i]<- g.function(X_cal_alter[i,],X_cal_alter[i,])
    for(j in 1:(i-1)){
      M[i,j] <- g.function(X_cal_alter[i,],X_cal_alter[j,])
      M[j,i] <- M[i,j]
    }
  }
  TS <- sum(M)
  }else{
    TS <- 0
  }
  return(TS)
}





diversity_estimated <- function(X_test_scale,TN,Current_selected){
  Xd=X_test_scale[Current_selected,]
  Td=TN[Current_selected]
  M <- matrix(0,nrow(Xd),nrow(Xd))
  for(i in 2:nrow(Xd)){
    #M[i,i]<- g.function(X_cal_alter[i,],X_cal_alter[i,])
    for(j in 1:(i-1)){
      M[i,j] <- g.function(Xd[i,],Xd[j,])*(1-Td[i])*(1-Td[j])
      M[j,i] <- M[i,j]
    }
  }
  #diversity <- sum(M)/(nrow(X_cal_alter)*(nrow(X_cal_alter)))
  diversity <- sum(M)/(nrow(Xd)*(nrow(Xd)-1))
  return(diversity=diversity)
}



diversity_true_online_correct_rej<-function(DIV,X_past_Alter,past.num,X_now,Alter_now)
{##computing true diversity (theta) in online paradigm
  if(past.num==0)
  {return(0)}
  if(Alter_now)
  { if(past.num==1){
    Delta=2*g.function(X_past_Alter,X_now)
  }else{Delta=2*sum(sapply(1:past.num,function(i){g.function(X_past_Alter[i,],X_now)}))}
    DIV=DIV*(past.num-1)/(past.num+1)+Delta/((past.num+1)*past.num)
    return(DIV)
  }else{return(DIV)}
}


NS_true_online_correct_rej <- function(NS,past_Alter,past.num,Alter_now){
  if(past.num==0){
    return(0)
  }
    Delta=sum(past_Alter)*Alter_now
    NS=NS+Delta
  return(NS)
}


TS_true_online_correct_rej <- function(TS,X_past_Alter,past.num,X_now,Alter_now){
  if(past.num==0)
  {return(0)}
  if(Alter_now){
    if(past.num==1){
      Delta=g.function(X_past_Alter,X_now)
    }else{
    Delta=sum(sapply(1:past.num,function(i){g.function(X_past_Alter[i,],X_now)}))
    }
    TS=TS+Delta
    return(TS)
  }else{return(TS)}
}


diversity_Delta_compute<-function(X_past,TN_past,X_now,TN_now,select.num)
{### online computing estimated diversity (locfdr)

#part1=g.function(X_now,X_now)*(1-TN_now)^2
if(select.num==0)
{#part2=2*g.function(X_past,X_now)*(1-TN_past)*(1-TN_now)
  part2=0
  }else if(select.num==1){part2=2*g.function(X_past,X_now)*(1-TN_past)*(1-TN_now)}else{
  #part2=2*sum(sapply(1:select.num,function(i){g.function(X_past[i,],X_now)*(1-TN_past[i])}))*(1-TN_now)
  part2=2*sum(sapply(1:select.num,function(i){g.function(X_past[i,],X_now)*(1-TN_past[i])}))*(1-TN_now)
}
#return(part1+part2)
  return(part2)
}


CiterionCompute_mES<-function(X_test_scale,Alter_test,decisions,select.num)
{
  TrueSignal=intersect(which(decisions==1),Alter_test)
  FDP=1-length(TrueSignal)/select.num
  Power=length(TrueSignal)/length(Alter_test)
  TS=TS_true_correct_rej(X_test_scale[TrueSignal,])
  NS=NS_true_correct_rej(X_test_scale[TrueSignal,])

  if(select.num==1){DIV=0}else{
    DIV=diversity_true_correct_rej(X_test_scale[TrueSignal,])
  }
  return(data.frame(FDP=FDP,DIV=DIV,Power=Power,NS=NS,TS=TS))
}


CiterionCompute_each_mES<-function(X_test_scale,Alter_test,decisions,t,x.Real)
{TrueSignal=intersect(which(decisions[1:t]==1),Alter_test)
  if (sum(decisions[1:t])!=0){
    FDP=1-length(TrueSignal)/sum(decisions[1:t])
    TS=TS_true_correct_rej(as.matrix(X_test_scale[TrueSignal,]))
    NS=NS_true_correct_rej(as.matrix(X_test_scale[TrueSignal,]))
    Power=length(TrueSignal)/length(intersect(1:t,Alter_test))
  } else {
    FDP=0
    TS=0
    NS=0
    Power=0
  }
  if (length(TrueSignal)<=1){DIV=0}else{
      DIV=diversity_true_correct_rej(X_test_scale[TrueSignal,])
  }
  return(data.frame(FDP=FDP,DIV=DIV,Power=Power,NS=NS,TS=TS))
}



### main procedure, true diversity is over all correct rejections
Online_selection_correct_rej_mES<-function(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity)
{
  select.num <- 0
  decisions <- rep(0,N)
  FDP<-rep(0,m)
  DIV<-rep(0,m)
  NS <- rep(0,m)
  TS <- rep(0,m)
  DIVloc<-rep(0,m)
  Power<-rep(0,m)
  active.correct.reject<-0
  active.Lfdr.mva <- 0
  Active.Diversity<- 0

  i <- 1

  while (select.num<m&i<=N) {
    fake_active.Lfdr.mva=(active.Lfdr.mva*select.num+TN[i])/(select.num+1)
    if(fake_active.Lfdr.mva<=alpha) # judge by local fdr
    {
      # compute true diversity in online paradigm
      Current_selected=which(decisions==1)
      X_past_Alter=X_test_scale[intersect(Current_selected,Alter_test),]
      past_Alter=intersect(Current_selected,Alter_test)
      past.num=length(intersect(Current_selected,Alter_test))
      if(past.num==0){
        DIV[select.num+1]=0
        TS[select.num+1]=0
        NS[select.num+1]=0
        }else{
          DIV[select.num+1]=diversity_true_online_correct_rej(DIV[max(select.num,1)],X_past_Alter,past.num,X_test_scale[i,],i %in% Alter_test)
          TS[select.num+1]=TS_true_online_correct_rej(TS[max(select.num,1)],X_past_Alter,past.num,X_test_scale[i,],i %in% Alter_test)
          NS[select.num+1]=NS_true_online_correct_rej(NS[max(select.num,1)], past_Alter, past.num, i %in% Alter_test)
          }

      if(IsDiversity)
      {
        #compute estimated diversity for each decision point in online paradigm

        Delta=diversity_Delta_compute(X_test_scale[Current_selected,],TN[Current_selected],X_test_scale[i,],TN[i],select.num)
        Fake_active.correct.reject=active.correct.reject+2*(1-TN[i])*sum(1-if(select.num==0){1}else{TN[Current_selected]} )
        Fake_Active.Diversity=Active.Diversity*(active.correct.reject)/max(Fake_active.correct.reject,1)+Delta/max(Fake_active.correct.reject,1)


        if(select.num<Diversity_initial_num){## for the first value, we do not consider diversity
          select.num=select.num+1
          active.Lfdr.mva=fake_active.Lfdr.mva
          Active.Diversity=Fake_Active.Diversity
          active.correct.reject=Fake_active.correct.reject
          decisions[i]=1
          TrueSignal=intersect(which(decisions==1),Alter_test)
          FDP[select.num]=1-length(TrueSignal)/select.num
          Power[select.num]=length(TrueSignal)/length(intersect(1:i,Alter_test))
        }else{
          if(Fake_Active.Diversity<=Diversity_threshold) #judge by diversity
          {select.num=select.num+1
          active.Lfdr.mva=fake_active.Lfdr.mva
          decisions[i]=1
          Active.Diversity=Fake_Active.Diversity
          active.correct.reject=Fake_active.correct.reject
          DIVloc[select.num]=Active.Diversity

          TrueSignal=intersect(which(decisions==1),Alter_test)
          FDP[select.num]=1-length(TrueSignal)/select.num
          Power[select.num]=length(TrueSignal)/length(intersect(1:i,Alter_test))}
        }

      }else{ # For case we only consider Local fdr
        select.num=select.num+1
        active.Lfdr.mva=fake_active.Lfdr.mva
        decisions[i]=1
        TrueSignal=intersect(which(decisions==1),Alter_test)
        FDP[select.num]=1-length(TrueSignal)/select.num
        Power[select.num]=length(TrueSignal)/length(intersect(1:i,Alter_test))
      }
    }

    i=i+1
  }
  stoptime <- i-1

  return(list(FDP=FDP[1:select.num],DIV=DIV[1:select.num],TS=TS[1:select.num],NS=NS[1:select.num],
              Power=Power[1:select.num],decisions=decisions,select.num=select.num,stoptime=stoptime))
}





TimeCovert<-function(FDP,decisions)
{#convert FDP or diversity to the whole time
  FDP_time=rep(0,length(decisions))
  current_FDP=0
  select.num=0
  for (i in 1:length(decisions)) {
    if(decisions[i]==1){
      select.num=select.num+1
      current_FDP=FDP[select.num]
    }
    FDP_time[i]=current_FDP
  }
  return(FDP_time)
}


removeColsAllNa  <- function(x){x[, apply(x, 2, function(y) any(!is.na(y)))]}



Decision_compute <- function(full_dec, m){
  for(i in 1:length(full_dec)){
    if(sum(full_dec)<m){
      if(sum(full_dec[1:i])==sum(full_dec)){
        deathtime=i
        select.num=sum(full_dec[1:deathtime])
        stoptime=NA
        decisions=full_dec[1:i]
        isdeath=TRUE
        #break
      }
    }else{
      if(sum(full_dec[1:i])==m){
        stoptime=i
        select.num=sum(full_dec[1:stoptime])
        deathtime=NA
        decisions=full_dec[1:i]
        isdeath=FALSE
        #break
      }
    }
  }
  return(list(decisions=decisions,stoptime=stoptime,deathtime=deathtime,isdeath=isdeath,select.num=select.num))
}





Online_selection_naive <- function(pval,X_test_scale,Alter_test,alpha,m,N){
  N <- length(pval)
  select.num <- 0
  decisions <- rep(0,N)
  FDP <- rep(0,m)
  DIV <- rep(0,m)
  NS <- rep(0,m)
  TS <- rep(0,m)
  Power <- rep(0,m)

  i <- 1
  while (select.num<m&i<=N) {
    if(pval[i]<=alpha){

      # compute true diversity in online paradigm
      Current_selected=which(decisions==1)
      X_past_Alter=X_test_scale[intersect(Current_selected,Alter_test),]
      past_Alter=intersect(Current_selected,Alter_test)
      past.num=length(intersect(Current_selected,Alter_test))
      if(past.num==0){
        DIV[select.num+1]=0
        TS[select.num+1]=0
        NS[select.num+1]=0
      }else{
        DIV[select.num+1]=diversity_true_online_correct_rej(DIV[max(select.num,1)],X_past_Alter,past.num,X_test_scale[i,],i %in% Alter_test)
        TS[select.num+1]=TS_true_online_correct_rej(TS[max(select.num,1)],X_past_Alter,past.num,X_test_scale[i,],i %in% Alter_test)
        NS[select.num+1]=NS_true_online_correct_rej(NS[max(select.num,1)], past_Alter, past.num, i %in% Alter_test)
      }

      select.num <- select.num+1
      decisions[i] <- 1
      TrueSignal=intersect(which(decisions==1),Alter_test)
      FDP[select.num]=1-length(TrueSignal)/select.num
      Power[select.num]=length(TrueSignal)/length(intersect(1:i,Alter_test))
    }
    i <- i+1
    stoptime <- i-1
  }
  return(list(FDP=FDP[1:select.num],DIV=DIV[1:select.num],TS=TS[1:select.num],NS=NS[1:select.num],
              Power=Power[1:select.num],decisions=decisions[1:stoptime],stoptime=stoptime,select.num=select.num))
}


tidy_res.each <- function(t, Method_FDP, Method_DIV){
  FDP_Method.ave <- rowMeans(Method_FDP[t,],na.rm = TRUE)
  DIV_Method.ave <- rowMeans(Method_DIV[t,],na.rm = TRUE)
  return(data.frame(FDP.ave=FDP_Method.ave,DIV.ave=DIV_Method.ave)
  )

}

tidy_res.each1 <- function(t, Method_FDP, Method_DIV,Method_TS,Method_NS){
  FDP_Method.ave <- rowMeans(Method_FDP[t,],na.rm = TRUE)
  DIV_Method.ave <- rowMeans(Method_DIV[t,],na.rm = TRUE)
  TS_Method.ave <- rowMeans(Method_TS[t,],na.rm = TRUE)
  NS_Method.ave <- rowMeans(Method_NS[t,],na.rm = TRUE)
  mES_Method.ave <- TS_Method.ave/NS_Method.ave
  return(data.frame(FDP.ave=FDP_Method.ave,DIV.ave=DIV_Method.ave,
                    mES.ave=mES_Method.ave)
  )
}



CI.fun <- function(x){
  x <- na.omit(x)
  y <- mean(x)+c(-1.96,1.96)*sd(x)/sqrt(length(x))
  y[which(y<0)]=0
  y[which(y>1)]=1
  return(y)
}
