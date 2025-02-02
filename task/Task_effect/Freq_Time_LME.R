library(readr)
library(reshape2)
library(dplyr)
library("ggpubr")
library(rstatix)
library(reprex)
library(DescTools)
library(R.matlab)

## BH
GPi_BH = read.table(paste0("./new_data_1204/GRA0.1dict500Freq_BH/GPi_cbeta.txt"), sep = ',')
GPi_BH = GPi_BH[2:dim(GPi_BH)[1],]
GPi_BH$V1 <- as.numeric(GPi_BH$V1)
GPi_BH$V2 <- as.numeric(GPi_BH$V2)
GPi_BH$V3 <- as.numeric(GPi_BH$V3)
GPi_BH$V4 <- as.numeric(GPi_BH$V4)
GPi_BH = rowMeans(GPi_BH)

## BL
GPi_BL = read.table(paste0("./new_data_1204/GRA0.1dict500Freq_BL/GPi_cbeta.txt"), sep = ',')
GPi_BL = GPi_BL[2:dim(GPi_BL)[1],]
GPi_BL$V1 <- as.numeric(GPi_BL$V1)
GPi_BL$V2 <- as.numeric(GPi_BL$V2)
GPi_BL$V3 <- as.numeric(GPi_BL$V3)
GPi_BL$V4 <- as.numeric(GPi_BL$V4)
GPi_BL = rowMeans(GPi_BL)

## BV
GPi_BV = read.table(paste0("./new_data_1204/GRA0.1dict500Freq_BV/GPi_cbeta.txt"), sep = ',')
GPi_BV = GPi_BV[2:dim(GPi_BV)[1],]
GPi_BV$V1 <- as.numeric(GPi_BV$V1)
GPi_BV$V2 <- as.numeric(GPi_BV$V2)
GPi_BV$V3 <- as.numeric(GPi_BV$V3)
GPi_BV$V4 <- as.numeric(GPi_BV$V4)
GPi_BV = rowMeans(GPi_BV)

t.test(GPi_BH, GPi_BL, paired = TRUE)
t.test(GPi_BH, GPi_BV, paired = TRUE)
t.test(GPi_BL, GPi_BV, paired = TRUE)

d <- data.frame(BH = GPi_BH, BV = GPi_BV, BL = GPi_BL, id = seq(1:14))

d <- melt(d, id.vars = 'id')

ggpaired(d, x = "variable", y = "value",
         color = "variable", palette = "jco", line.color = "#0000ff00")
res.aov <- anova_test(data = d, dv = value, wid = id, within = variable)
get_anova_table(res.aov)

summary(res)
TukeyHSD(res)

## BH
M1_BH = read.table(paste0("./new_data_1204/GRA0.1dict500Freq_BH/M1_cbeta.txt"), sep = ',')
M1_BH = M1_BH[2:dim(M1_BH)[1],]
M1_BH$V1 <- as.numeric(M1_BH$V1)
M1_BH$V2 <- as.numeric(M1_BH$V2)
M1_BH$V3 <- as.numeric(M1_BH$V3)
M1_BH = rowMeans(M1_BH)

## BL
M1_BL = read.table(paste0("./new_data_1204/GRA0.1dict500Freq_BL/M1_cbeta.txt"), sep = ',')
M1_BL = M1_BL[2:dim(M1_BL)[1],]
M1_BL$V1 <- as.numeric(M1_BL$V1)
M1_BL$V2 <- as.numeric(M1_BL$V2)
M1_BL$V3 <- as.numeric(M1_BL$V3)
M1_BL = rowMeans(M1_BL)

## BV
M1_BV = read.table(paste0("./new_data_1204/GRA0.1dict500Freq_BV/M1_cbeta.txt"), sep = ',')
M1_BV = M1_BV[2:dim(M1_BV)[1],]
M1_BV$V1 <- as.numeric(M1_BV$V1)
M1_BV$V2 <- as.numeric(M1_BV$V2)
M1_BV$V3 <- as.numeric(M1_BV$V3)
M1_BV = rowMeans(M1_BV)

t.test(M1_BH, M1_BL, paired = TRUE)
t.test(M1_BL, M1_BV, paired = TRUE)
t.test(M1_BH, M1_BV, paired = TRUE)

d <- data.frame(BH = M1_BH, BV = M1_BV, BL = M1_BL, id = seq(1, 14))

d <- melt(d, id.vars = "id")
ggpaired(d, x = "variable", y = "value",
         color = "variable", palette = "jco", line.color = "#0000ff00")

res.aov <- anova_test(data = d, dv = value, wid = id, within = variable)
get_anova_table(res.aov)

summary(res)
TukeyHSD(res)

##### Month ####
## 1m
M11m = read.table(paste0("./new_data_1204/GRA0.1dict5001m/M1_cbeta.txt"), sep = ',')
M11m = M11m[2:dim(M11m)[1],]
M11m$V1 <- as.numeric(M11m$V1)
M11m$V2 <- as.numeric(M11m$V2)
M11m$V3 <- as.numeric(M11m$V3)
M11m = rowMeans(M11m)

GPi1m = read.table(paste0("./new_data_1204/GRA0.1dict5001m/GPi_cbeta.txt"), sep = ',')
GPi1m = GPi1m[2:dim(GPi1m)[1],]
GPi1m$V1 <- as.numeric(GPi1m$V1)
GPi1m$V2 <- as.numeric(GPi1m$V2)
GPi1m$V3 <- as.numeric(GPi1m$V3)
GPi1m$V4 <- as.numeric(GPi1m$V4)
GPi1m = rowMeans(GPi1m)

## 3m
M13m = read.table(paste0("./new_data_1204/GRA0.1dict5003m/M1_cbeta.txt"), sep = ',')
M13m = M13m[2:dim(M13m)[1],]
M13m$V1 <- as.numeric(M13m$V1)
M13m$V2 <- as.numeric(M13m$V2)
M13m$V3 <- as.numeric(M13m$V3)
M13m = rowMeans(M13m)

GPi3m = read.table(paste0("./new_data_1204/GRA0.1dict5003m/GPi_cbeta.txt"), sep = ',')
GPi3m = GPi3m[2:dim(GPi3m)[1],]
GPi3m$V1 <- as.numeric(GPi3m$V1)
GPi3m$V2 <- as.numeric(GPi3m$V2)
GPi3m$V3 <- as.numeric(GPi3m$V3)
GPi3m$V4 <- as.numeric(GPi3m$V4)
GPi3m = rowMeans(GPi3m)

## 6m
M16m = read.table(paste0("./new_data_1204/GRA0.1dict5006m/M1_cbeta.txt"), sep = ',')
M16m = M16m[2:dim(M16m)[1],]
M16m$V1 <- as.numeric(M16m$V1)
M16m$V2 <- as.numeric(M16m$V2)
M16m$V3 <- as.numeric(M16m$V3)
M16m = rowMeans(M16m)

GPi6m = read.table(paste0("./new_data_1204/GRA0.1dict5006m/GPi_cbeta.txt"), sep = ',')
GPi6m = GPi6m[2:dim(GPi6m)[1],]
GPi6m$V1 <- as.numeric(GPi6m$V1)
GPi6m$V2 <- as.numeric(GPi6m$V2)
GPi6m$V3 <- as.numeric(GPi6m$V3)
GPi6m$V4 <- as.numeric(GPi6m$V4)
GPi6m = rowMeans(GPi6m)

t.test(GPi1m, GPi3m, paired = TRUE)
t.test(GPi1m, GPi6m, paired = TRUE)
t.test(GPi3m, GPi6m, paired = TRUE)

d <- data.frame(M1 = M1_BH, GPi = GPi_BH, id = seq(1, 14))

d <- melt(d, id.vars = "id")
ggpaired(d, x = "variable", y = "value",
         color = "variable", palette = "jco", line.color = "#0000ff00")

res.aov <- anova_test(data = d, dv = value, wid = id, within = variable)
get_anova_table(res.aov)

summary(res)
TukeyHSD(res)
