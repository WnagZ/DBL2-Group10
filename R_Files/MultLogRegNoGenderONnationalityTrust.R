# Load necessary libraries
library(foreign)
library(nnet)
library(ggplot2)
library(reshape2)
library(caret)
library(dplyr)


# Multinomial regression for Trust
# Read the data
data <- read.csv("PAS_data_shifted.csv")

# remeber to add Q62e i there is no difference

# Define ordinal columns and selected predictors
ordinal_columns <- c( "Q1", "Q3C", "Q3H", "Q3J", "Q13", "Q15",  "Q39A_2", "A120", 
                      "A121","Q62A","Q62B","Q62C", "Q62D", "Q62E", "Q62F", "Q62TG",
                      "Q62TI", "Q62TJ", "Q79A", "Q79B", "Q79C", "Q79D", "Q79E", 
                      "Q79F", "Q79G", "Q79H", "Q133", "Q3L", "Q62H")

selected_predictors <- c("NQ135BD", ordinal_columns)

# Subset the data
main <- data[, selected_predictors]

# Filter out rows with zero values in any column
main <- main %>% filter_all(all_vars(. != 0))

testing <- main

# Convert ordinal columns to ordered factors
testing[ordinal_columns] <- lapply(testing[ordinal_columns], function(x) factor(x, ordered = TRUE))

# Convert target variable to factor
testing$NQ135BD <- as.factor(testing$NQ135BD)
# 
# # Relevel the target variable
testing$NQ135BD_2 <- relevel(testing$NQ135BD, ref = 3)
# 
# # Split the data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(testing$NQ135BD_2, p = 0.7, list = FALSE)
train_data <- testing[trainIndex, ]
test_data <- testing[-trainIndex, ]
# 
# # Fit the multinomial logistic regression model on the training set
# model <- multinom(NQ135BD ~ Q1 + Q3H + Q13 + Q15 + NQ21 + Q39A_2 + Q62A + A120 + 
#                     Q62B + Q62C + Q62D + Q62E + Q62F + Q62H + Q79C + A121, data = train_data)

model <- multinom(NQ135BD_2 ~ ., 
                  data = train_data[,c("NQ135BD_2", ordinal_columns)])

# 
# # Summary of the model
summary(model)

# # Calculate z-values and p-values
z <- summary(model)$coefficients / summary(model)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2

# # Print z-values and p-values
#print(z)
print(p)

# # Print exponentiated coefficients (odds ratios)
print(exp(coef(model)))

# # Get the fitted probabilities for each class
head(pp <- fitted(model))

# # Predict on the testing set
predicted_classes <- predict(model, newdata = test_data)
predicted_classes <- factor(predicted_classes, levels = levels(test_data$NQ135BD_2))
# # Create a confusion matrix to evaluate the performance
confusion_matrix <- confusionMatrix(predicted_classes, test_data$NQ135BD_2)
print(confusion_matrix)

# # Calculate and print additional metrics
accuracy <- confusion_matrix$overall['Accuracy']
print(paste("Accuracy:", accuracy))


# deviance(model)
# 
# # Perform likelihood ratio test (comparing to null model)
# null_model <- multinom(response ~ 1, data = your_data)  # Null model (intercept only)
# lr_test <- lrtest(null_model, model)
# lr_test

