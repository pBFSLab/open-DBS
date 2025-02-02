# %%
## Compare different conditions at various time points using repeated-measures ANOVA
library(readr)
library(reshape2)
library(dplyr)
library("ggpubr")
library(rstatix)
library(reprex)
library(DescTools)
library(R.matlab)

# Read the CSV file
df <- read.csv("./data/UPDRS_long.csv")
### Output the first few rows of the data frame
head(df)

### Set 'time' and 'Freq' as factors
df$time <- factor(df$time, levels = c("1m", "3m", "6m", "12m"))
df$Freq <- factor(df$Freq, levels = c("off", "HFS", "VFS", "LFS"))
df$sub <- as.factor(df$sub)

### Compare different time points under the 'off' condition
df_off <- df[df$Freq == "off", ]
anova_iff <- anova_test(data = df_off, dv = score, wid = sub, within = c(time))
get_anova_table(anova_iff)

### Compare different time points under the 'HFS' condition
df_HFS <- df[df$Freq == "HFS", ]
anova_iff_HFS <- anova_test(data = df_HFS, dv = score, wid = sub, within = c(time))
get_anova_table(anova_iff_HFS)

### Compare different time points under the 'VFS' condition
df_VFS <- df[df$Freq == "VFS", ]
anova_iff_VFS <- anova_test(data = df_VFS, dv = score, wid = sub, within = c(time))
get_anova_table(anova_iff_VFS)

### Compare different time points under the 'LFS' condition
df_LFS <- df[df$Freq == "LFS", ]
anova_iff_LFS <- anova_test(data = df_LFS, dv = score, wid = sub, within = c(time))
get_anova_table(anova_iff_LFS)
