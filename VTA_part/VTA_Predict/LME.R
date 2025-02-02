# install.packages("lmerTest")
library(lme4)
library(lmerTest)

df <- read.csv("/home/ssli/Nutstore Files/Nutstore/MyResearches/DBS_2024/绘图整理/UPDRS/prediction/patient_idealmap_predict_N140_reorg.csv")

model <- lmer(gt1 ~ preds1 + (1 | sub), data = df)
anova(model)
