# Load necessary libraries
require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)
require(caret)

# Read the data
data <- read.csv("PAS_data_shifted.csv")

# Multinimial logistic regression for the Confidence

# Define ordinal columns and selected predictors
ordinal_columns <- c( "Q1", "Q3C", "Q3H", "Q3J", "Q13", "Q15",  "Q39A_2", "A120", 
                      "A121","Q62A","Q62B","Q62C", "Q62D", "Q62E", "Q62F", "Q62TG",
                      "Q62TI", "Q62TJ", "Q79A", "Q79B", "Q79C", "Q79D", "Q79E", 
                      "Q79F", "Q79G", "Q79H", "Q133", "Q3L", "Q62H")

selected_predictors <- c("Q61", ordinal_columns)

# Subset the data
main <- data[, selected_predictors]

# Filter out rows with zero values in any column
main <- main %>% filter_all(all_vars(. != 0))

testing <- main

# Convert ordinal columns to ordered factors
testing[ordinal_columns] <- lapply(testing[ordinal_columns], function(x) factor(x, ordered = TRUE))

# Convert target variable to factor
testing$Q61 <- as.factor(testing$Q61)
# 
# # Relevel the target variable
testing$Q61_2 <- relevel(testing$Q61, ref = 3)
# 
# # Split the data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(testing$Q61_2, p = 0.7, list = FALSE)
train_data <- testing[trainIndex, ]
test_data <- testing[-trainIndex, ]
# 
# # Fit the multinomial logistic regression model on the training set
# model <- multinom(NQ135BD ~ Q1 + Q3H + Q13 + Q15 + NQ21 + Q39A_2 + Q62A + A120 + 
#                     Q62B + Q62C + Q62D + Q62E + Q62F + Q62H + Q79C + A121, data = train_data)

model_c <- multinom(Q61_2 ~ ., 
                  data = train_data[,c("Q61_2", ordinal_columns)])

# 
# # model_c of the model
summary(model_c)

# # Calculate z-values and p-values
z_c <- summary(model_c)$coefficients / summary(model_c)$standard.errors
p_c <- (1 - pnorm(abs(z_c), 0, 1)) * 2

# # Print z-values and p-values
#print(z)
print(p_c)

# # Print exponentiated coefficients (odds ratios)
print(exp(coef(model_c)))

# # Get the fitted probabilities for each class
head(pp_c <- fitted(model_c))

# # Predict on the testing set
predicted_classes_c <- predict(model_c, newdata = test_data)
predicted_classes_c <- factor(predicted_classes_c, levels = levels(test_data$Q61_2))
# # Create a confusion matrix to evaluate the performance
confusion_matrix_c <- confusionMatrix(predicted_classes_c, test_data$Q61_2)
print(confusion_matrix_c)

# # Calculate and print additional metrics
accuracy_c <- confusion_matrix_c$overall['Accuracy']
print(paste("Accuracy:", accuracy_c))
