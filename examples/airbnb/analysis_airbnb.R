library(ggplot2)
library(ggpubr)
library(rpart)
library(rpart.plot)

df <- read.csv("combined.csv", 
                 header = TRUE,
                 sep = ",")

sds <- read.csv("syn_train_users.csv", 
               header = TRUE,
               sep = ",")

ofit <- rpart(V5 ~ V1 + V2 + V3 + V4,
              method="class", data=df)
sfit <- rpart(V5 ~ V1 + V2 + V3 + V4,
              method="class", data=sds)

par(mfrow = c(1,2))
par(mar=c(1,1,1,1))

rpart.plot(ofit)
rpart.plot(sfit)



