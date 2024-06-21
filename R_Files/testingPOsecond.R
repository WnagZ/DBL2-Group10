# checking the paralleel lines assumption

library(ggplot2)                     
library(GGally)
library(forcats)

library(ordinal)
library(rcompanion)
library(MASS)
library(brant) 
library(car)

library(foreign)
library(Hmisc)
library(reshape2)
library(dplyr)

data = read.csv("PAS_data_shifted.csv")

ordinal_columns <- c("Q1", "Q3H")  

selected_var <- c("NQ135BD","Q1", "Q3H")

main <- data[, selected_var]

# Remove rows where any column equals 0
main <- main %>%
  filter_all(all_vars(. != 0))

# Print the filtered data frame
print(nrow(main))


# making a copy of the data
testing <- main
testing[ordinal_columns] <- lapply(testing[ordinal_columns], function(x) factor(x, ordered = TRUE))


print(levels(testing$NQ135BD))

testing <- testing %>%
  mutate(
    # Convert NQ135BD to a factor
    NQ135BD = as.factor(NQ135BD),
    
    # Y1 = Trust > 1
    Y1 = fct_collapse(NQ135BD,
                      ">1"  = c("2", "3", "4", "5"),
                      "<=1" = "1"),
    Y1 = relevel(Y1, ref = "<=1"),
    
    # Y2 = Trust > 2
    Y2 = fct_collapse(NQ135BD,
                      ">2"  = c("3", "4", "5"),
                      "<=2" = c("1", "2")),
    Y2 = relevel(Y2, ref = "<=2"),
    
    # Y3 = Trust > 3
    Y3 = fct_collapse(NQ135BD,
                      ">3"  = c("4", "5"),
                      "<=3" = c("1", "2", "3")),
    Y3 = relevel(Y3, ref = "<=3"),
    
    # Y4 = Trust > 4
    Y4 = fct_collapse(NQ135BD,
                      ">4"  = "5",
                      "<=4" = c("1", "2", "3", "4")),
    Y4 = relevel(Y4, ref = "<=4")
  )

table(testing$NQ135BD, testing$Y1, exclude = F)
table(testing$NQ135BD, testing$Y2, exclude = F)
table(testing$NQ135BD, testing$Y3, exclude = F)
table(testing$NQ135BD, testing$Y4, exclude = F)

fit.ordinal <- polr(as.factor(NQ135BD) ~  Q1+ Q3H, 
                    data = testing, 
                    Hess = TRUE)

fit.binary1 <- glm(Y1 ~ Q1 +  Q3H,
                   family = binomial, data = testing)

fit.binary2 <- glm(Y2 ~ Q1 +  Q3H, 
                   family = binomial, data = testing)

fit.binary3 <- glm(Y3 ~ Q1 + Q3H, 
                   family = binomial, data = testing)

fit.binary4 <- glm(Y4 ~ Q1 + Q3H, 
                   family = binomial, data = testing)


# Assuming fit.ordinal, fit.binary1, fit.binary2, fit.binary3, fit.binary4 are already fitted models

# Extract coefficients
ordinal_coefs <- coef(fit.ordinal)
binary1_coefs <- coef(fit.binary1)[-1]  # Exclude intercept
binary2_coefs <- coef(fit.binary2)[-1]  # Exclude intercept
binary3_coefs <- coef(fit.binary3)[-1]  # Exclude intercept
binary4_coefs <- coef(fit.binary4)[-1]  # Exclude intercept

# Combine coefficients into a data frame
coefficients_df <- data.frame(
  "ordinal"  = ordinal_coefs,
  "binary 1" = binary1_coefs,
  "binary 2" = binary2_coefs,
  "binary 3" = binary3_coefs,
  "binary 4" = binary4_coefs
)

# Exponentiation coefficients
exp_coefficients <- exp(coefficients_df)

# Display the exponentiation coefficients
print(exp_coefficients)

### There are too big differences between predictors which violates the assummption 
### of parallel lines. The solution can be to use less questions, and only check 
### some of the predictors. 


summary(fit.ordinal)

brant(fit.ordinal)
