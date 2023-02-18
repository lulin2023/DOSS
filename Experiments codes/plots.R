setwd("E:\\Online Selection Sim\\Simulation results\\classification")
result_each_RF <- read.csv("result_each_cla_RF.csv")[,-1]
result_FSR_RF <- read.csv("result_FSR_cla_RF.csv")[,-1]
result_ES_RF <- read.csv("result_ES_cla_RF.csv")[,-1]
FDP.stop_RF <- read.csv("FDP_stop_cla_RF.csv")[,-c(1,4)]
DIV.stop_RF <- read.csv("DIV_stop_cla_RF.csv")[,-c(1,4)]
stop_time_RF <- read.csv("stop_time_cla_RF.csv")[,-c(1,4)]

result_each_NN <- read.csv("result_each_cla_NN.csv")[,-1]
result_FSR_NN <- read.csv("result_FSR_cla_NN.csv")[,-1]
result_ES_NN <- read.csv("result_ES_cla_NN.csv")[,-1]
FDP.stop_NN <- read.csv("FDP_stop_cla_NN.csv")[,-c(1,4)]
DIV.stop_NN <- read.csv("DIV_stop_cla_NN.csv")[,-c(1,4)]
stop_time_NN <- read.csv("stop_time_cla_NN.csv")[,-c(1,4)]

result_each_SVM <- read.csv("result_each_cla_SVM.csv")[,-1]
result_FSR_SVM <- read.csv("result_FSR_cla_SVM.csv")[,-1]
result_ES_SVM <- read.csv("result_ES_cla_SVM.csv")[,-1]
FDP.stop_SVM <- read.csv("FDP_stop_cla_SVM.csv")[,-c(1,4)]
DIV.stop_SVM <- read.csv("DIV_stop_cla_SVM.csv")[,-c(1,4)]
stop_time_SVM <- read.csv("stop_time_cla_SVM.csv")[,-c(1,4)]


FDP.stop_RF$Alg="RF"
FDP.stop_NN$Alg="NN"
FDP.stop_SVM$Alg="SVM"

FDP.stop <- rbind(FDP.stop_SVM,FDP.stop_NN)


#  plots the results when stop--------

FDP_stop_value <- melt(FDP.stop)
names(FDP_stop_value) <- c('Alg','Method','FSR')

p_FSR <- ggplot(data=FDP_stop_value,aes(x=Method,y=FSR,color=Method))
p_FSR <- p_FSR+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,    
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = "FSR", x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+ facet_grid(. ~ Alg)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 16)) 
p_FSR <- p_FSR + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")


DIV.stop_RF$Alg="RF"
DIV.stop_NN$Alg="NN"
DIV.stop_SVM$Alg="SVM"

DIV.stop <- rbind(DIV.stop_SVM,DIV.stop_NN)
DIV.stop$LOND[which(DIV.stop$LOND>0.4)]=0.06
DIV.stop$SAFFRON[which(DIV.stop$SAFFRON>0.2)]=0.06

ES_stop_value <- melt(DIV.stop)
names(ES_stop_value) <- c('Alg','Method','ES')
p_ES <- ggplot(data=ES_stop_value,aes(x=Method,y=ES,color=Method))
p_ES <- p_ES+geom_boxplot()+
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,    
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = "ES", x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+ facet_grid(. ~ Alg)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=90, hjust=1, vjust=.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 16)) 
p_ES <- p_ES + geom_hline(aes(yintercept=0.045), colour="black", linetype="dashed")
p_ES


stop_time_RF$Alg="RF"
stop_time_NN$Alg="NN"
stop_time_SVM$Alg="SVM"

stop_time <- rbind(stop_time_RF,stop_time_SVM,stop_time_NN)

stop_time_value <- melt(stop_time)
names(stop_time_value) <- c('Alg','Method','Stoptime')


p_Time <- ggplot(data=stop_time_value,aes(x=Method,y=Stoptime,color=Method))
p_Time <- p_Time+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,    
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = TeX("$T_{m}$"), x = "Method")+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)

p_Time <- p_Time+ facet_grid(. ~ Alg)+theme(axis.text = element_text(size = 16),
                       axis.title = element_text(size = 20),
                       axis.text.x = element_text(angle=90, hjust=1, vjust=.5),
                       panel.grid.major=element_line(colour=NA),
                       panel.background = element_rect(fill = "transparent",colour = NA),
                       plot.background = element_rect(fill = "transparent",colour = NA),
                       panel.grid.minor = element_blank()) 
p_Time

scaleFUN <- function(x) sprintf("%.2f", x) 
p_ES <- p_ES + scale_y_continuous(labels=scaleFUN) 
p_FSR <- p_FSR + scale_y_continuous(labels=scaleFUN) 

ggarrange(p_FSR, p_ES,ncol=1, nrow=2, common.legend = TRUE, legend="bottom", 
          font.label = list(size = 12, face = "bold"))



