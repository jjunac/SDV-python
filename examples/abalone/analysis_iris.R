library(ggplot2)
library(ggpubr)
library(rpart)
library(rpart.plot)

df <- read.csv("iris.data", 
                 header = FALSE,
                 sep = ",")

sds <- read.csv("synthetic_iris_simple.data", 
               header = FALSE,
               sep = ",")

df$set = rep("observed", nrow(df))
sds$set = rep("synthetised", nrow(sds))
all = rbind(df, sds)

p1 <- ggplot(all, aes(x=V1, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p2 <- ggplot(all, aes(x=V2, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p3 <- ggplot(all, aes(x=V3, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p4 <- ggplot(all, aes(x=V4, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
ggarrange(p1, p2, p3, p4)

readline("Press <return to continue")

df <- read.csv("iris.data", 
               header = FALSE,
               sep = ",")

sds <- read.csv("synthetic_iris_expon_beta.data", 
                header = FALSE,
                sep = ",")

df$set = rep("observed", nrow(df))
sds$set = rep("synthetised", nrow(sds))
all = rbind(df, sds)

p1 <- ggplot(all, aes(x=V1, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p2 <- ggplot(all, aes(x=V2, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p3 <- ggplot(all, aes(x=V3, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p4 <- ggplot(all, aes(x=V4, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
ggarrange(p1, p2, p3, p4)

readline("Press <return to continue")

p1 <- ggplot(all, aes(x=V1, colour=set)) +
  geom_density()
p2 <- ggplot(all, aes(x=V2, colour=set)) +
  geom_density()
p3 <- ggplot(all, aes(x=V3, colour=set)) +
  geom_density()
p4 <- ggplot(all, aes(x=V4, colour=set)) +
  geom_density()
ggarrange(p1, p2, p3, p4)

readline("Press <return to continue")

df <- read.csv("iris.data", 
               header = FALSE,
               sep = ",")

sds <- read.csv("synthetic_iris_final.data", 
                header = FALSE,
                sep = ",")

df$set = rep("observed", nrow(df))
sds$set = rep("synthetised", nrow(sds))
all = rbind(df, sds)

p1 <- ggplot(all, aes(x=V1, colour=set)) +
  geom_density()
p2 <- ggplot(all, aes(x=V2, colour=set)) +
  geom_density()
p3 <- ggplot(all, aes(x=V3, colour=set)) +
  geom_density()
p4 <- ggplot(all, aes(x=V4, colour=set)) +
  geom_density()
ggarrange(p1, p2, p3, p4)

readline("Press <return to continue")

p1 <- ggplot(all, aes(x=V1, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p2 <- ggplot(all, aes(x=V2, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p3 <- ggplot(all, aes(x=V3, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
p4 <- ggplot(all, aes(x=V4, colour=set)) +
  geom_density() + 
  facet_wrap(~V5)
ggarrange(p1, p2, p3, p4)

readline("Press <return to continue")

ofit <- rpart(V5 ~ V1 + V2 + V3 + V4,
              method="class", data=df)
sfit <- rpart(V5 ~ V1 + V2 + V3 + V4,
              method="class", data=sds)

par(mfrow = c(1,2))
par(mar=c(1,1,1,1))

rpart.plot(ofit)
rpart.plot(sfit)



