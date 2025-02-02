# %%
# Install and load lme4 and lmerTest packages
# install.packages("lmerTest")
library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)

# %%
# Read the CSV file
df <- read.csv("./data/UPDRS_long.csv")
### Output the first few rows of the data frame
head(df)

### Set 'time' and 'Freq' as factors
df$time <- factor(df$time, levels = c("1m", "3m", "6m", "12m"))
df$Freq <- factor(df$Freq, levels = c("off", "HFS", "VFS", "LFS"))
df$sub <- as.factor(df$sub)

# %%
### Perform comparison between ON and OFF conditions
# Create a new condition variable (Freq2: ON vs OFF)
df$Freq2 <- ifelse(df$Freq %in% c("HFS", "VFS", "LFS"), "ON", "OFF")
df$Freq2 <- factor(df$Freq2, levels = c("OFF", "ON"))  # Ensure OFF is the reference level

# Fit a new model
model3 <- lmer(score ~ time * Freq2 + (1 | sub), data = df)
summary(model3)
anova(model3)

### Perform post-hoc comparisons
marginal_means3 <- emmeans(model3, ~ Freq2)
marginal_means_time3 <- emmeans(model3, ~ time | Freq2)
marginal_means_freq3 <- emmeans(model3, ~ Freq2 | time)

### Perform pairwise comparisons
pairwise_comparisons3 <- contrast(marginal_means3, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons3)

pairwise_comparisons_time3 <- contrast(marginal_means_time3, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_time3)

pairwise_comparisons_freq3 <- contrast(marginal_means_freq3, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_freq3)

# %%
### Keep only HFS and LFS data, build a new model
df2 <- df[df$Freq %in% c("HFS", "LFS"), ]
model4_HL <- lmer(score ~ time * Freq + (1 | sub), data = df2)
summary(model4_HL)
anova(model4_HL)

### Perform post-hoc comparisons
marginal_means4_HL <- emmeans(model4_HL, ~ Freq | time)
marginal_means_Freq4_HL <- emmeans(model4_HL, ~ Freq)

### Perform pairwise comparisons
pairwise_comparisons4_HL <- contrast(marginal_means4_HL, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons4_HL)

pairwise_comparisons_Freq4_HL <- contrast(marginal_means_Freq4_HL, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_Freq4_HL)

# %%
### Keep only HFS and VFS data, build a new model
df3 <- df[df$Freq %in% c("HFS", "VFS"), ]
model4_HV <- lmer(score ~ time * Freq + (1 | sub), data = df3)
summary(model4_HV)
anova(model4_HV)

### Perform post-hoc comparisons
marginal_means4_HV <- emmeans(model4_HV, ~ Freq | time)
marginal_means_Freq4_HV <- emmeans(model4_HV, ~ Freq)

### Perform pairwise comparisons
pairwise_comparisons4_HV <- contrast(marginal_means4_HV, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons4_HV)

pairwise_comparisons_Freq4_HV <- contrast(marginal_means_Freq4_HV, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_Freq4_HV)

# %%
### Keep only VFS and LFS data, build a new model
df4 <- df[df$Freq %in% c("VFS", "LFS"), ]
model4_VL <- lmer(score ~ time * Freq + (1 | sub), data = df4)
summary(model4_VL)
anova(model4_VL)

# %%
### Keep only off and HFS data, build a new model
df5 <- df[df$Freq %in% c("off", "HFS"), ]
model4_OH <- lmer(score ~ time * Freq + (1 | sub), data = df5)
summary(model4_OH)
anova(model4_OH)

### Perform post-hoc comparisons
marginal_means4_OH <- emmeans(model4_OH, ~ Freq | time)
marginal_means_Freq4_OH <- emmeans(model4_OH, ~ time | Freq)

### Perform pairwise comparisons
pairwise_comparisons4_OH <- contrast(marginal_means4_OH, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons4_OH)

pairwise_comparisons_Freq4_OH <- contrast(marginal_means_Freq4_OH, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_Freq4_OH)

# %%
### Keep only off and VFS data, build a new model
df6 <- df[df$Freq %in% c("off", "VFS"), ]
model4_OV <- lmer(score ~ time * Freq + (1 | sub), data = df6)

summary(model4_OV)
anova(model4_OV)

### Perform post-hoc comparisons
marginal_means4_OV <- emmeans(model4_OV, ~ Freq | time)
marginal_means_Freq4_OV <- emmeans(model4_OV, ~ time | Freq)

### Perform pairwise comparisons
pairwise_comparisons4_OV <- contrast(marginal_means4_OV, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons4_OV)

pairwise_comparisons_Freq4_OV <- contrast(marginal_means_Freq4_OV, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_Freq4_OV)

# %%
### Keep only off and LFS data, build a new model
df7 <- df[df$Freq %in% c("off", "LFS"), ]
model4_OL <- lmer(score ~ time * Freq + (1 | sub), data = df7)

summary(model4_OL)
anova(model4_OL)

### Perform post-hoc comparisons
marginal_means4_OL <- emmeans(model4_OL, ~ Freq | time)
marginal_means_Freq4_OL <- emmeans(model4_OL, ~ time | Freq)

### Perform pairwise comparisons
pairwise_comparisons4_OL <- contrast(marginal_means4_OL, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons4_OL)

pairwise_comparisons_Freq4_OL <- contrast(marginal_means_Freq4_OL, interaction = "pairwise", adjust = "tukey")
summary(pairwise_comparisons_Freq4_OL)
