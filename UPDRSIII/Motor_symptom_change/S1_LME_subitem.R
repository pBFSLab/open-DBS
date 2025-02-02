# %%
# Install and load lme4 and lmerTest packages
# install.packages("lmerTest")
library(lme4)
library(lmerTest)
library(dplyr)
library(tidyverse)
library(emmeans)

# %%
# Read the CSV file
df <- read.csv("./Reliability/UPDRS_bradykinesia.csv")
### Output the first few rows of the data frame
head(df)

# Use the separate() function to split the 'condition' column into 'time' and 'Freq' columns
df <- df %>%
  separate(condition, into = c("time", "Freq"), sep = "(?<=m)")

### Set 'time' and 'Freq' as factors
df$time <- factor(df$time, levels = c("1m", "3m", "6m", "12m"))
df$Freq <- factor(df$Freq, levels = c("OFF", "CH", "CV", "CL"))
df$Subject <- as.factor(df$Subject)

### Calculate 'score' as the average of 'score' and 'score_retest'
df$score <- (df$score + df$score_retest) / 2

### Perform comparison between ON and OFF conditions
# Create a new condition variable (Freq2: ON vs OFF)
df$Freq2 <- ifelse(df$Freq %in% c("CH", "CV", "CL"), "ON", "OFF")
df$Freq2 <- factor(df$Freq2, levels = c("OFF", "ON"))  # Ensure OFF is the reference level

# Fit a linear mixed-effects model
model <- lmer(score ~ time * Freq2 + (1 | Subject), data = df)

# Perform analysis of variance using the lmertest package
summary(model)
anova(model)

### Perform post-hoc comparisons
marginal_means_freq <- emmeans(model, ~ Freq2 | time)

### Perform pairwise comparisons
pairwise_comparisons_freq <- pairs(marginal_means_freq)

summary(pairwise_comparisons_freq)

# %%
### Perform similar analysis for other sub-scores besides bradykinesia
sub_items <- c("gait", "posture", "tremor", "rigidity", "PIGD")

for (sub_item in sub_items) {
  print(paste("Analyzing", sub_item))
  
  # Read the CSV file for the current sub-item
  df <- read.csv(paste0("./Reliability/UPDRS_", sub_item, ".csv"))
  
  # Separate 'condition' column
  df <- df %>%
    separate(condition, into = c("time", "Freq"), sep = "(?<=m)")
  
  # Set 'time' and 'Freq' as factors
  df$time <- factor(df$time, levels = c("1m", "3m", "6m", "12m"))
  df$Freq <- factor(df$Freq, levels = c("OFF", "CH", "CV", "CL"))
  df$Subject <- as.factor(df$Subject)
  
  # Calculate 'score' as the average of 'score' and 'score_retest'
  df$score <- (df$score + df$score_retest) / 2
  
  # Create a new condition variable (Freq2: ON vs OFF)
  df$Freq2 <- ifelse(df$Freq %in% c("CH", "CV", "CL"), "ON", "OFF")
  df$Freq2 <- factor(df$Freq2, levels = c("OFF", "ON"))
  
  # Fit a linear mixed-effects model
  model <- lmer(score ~ time * Freq2 + (1 | Subject), data = df)
  
  # Perform analysis of variance
  summary(model)
  anova(model)
}
