df <- read.csv("iris.data", 
                 header = FALSE,
                 sep = ",")

sds <- read.csv("synthetic_iris3.data", 
               header = FALSE,
               sep = ",")

df$set = rep("observed", nrow(df))
sds$set = rep("synthetised", nrow(sds))
all = rbind(df, sds)

library(ggplot2)
library(ggpubr)

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