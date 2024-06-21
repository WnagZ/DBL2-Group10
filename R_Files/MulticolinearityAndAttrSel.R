# Load necessary libraries
library(foreign)
library(nnet)
library(ggplot2)
library(reshape2)
library(caret)
library(dplyr)

# in this script we chose an optimal selection of question to achieve big enough
# row counts: Goal have around 40000 rows, without multicolinearity


# based on the number of missing rows, first we select questions that will be 
# included. Main methods:
# 1. After dropping 0's (encoded NAN's) there should be around 40.000 rows.
# 2. Additional rows will be droopped if there is multicolinearity

# Ad. 

# Read the data
data <- read.csv("PAS_data_shifted.csv")

count_zeros <- function(column) {
    sum(column == 0)
}

zero_counts_data <- sapply(data, count_zeros)

print(zero_counts)

# we get the number of missing values per question
# questions that have the most missing values are:
# NQ21  If you are walking alone in this area and you see 
# a police officer on foot, bicycle or horseback, does it make you feel more safe, 
# less safe or does it make no difference?
# 19.340 Nan's for now wee will keep it as it was significant in past tests

# NQ57AE how satisfied you with Tram Network? 26.000

# NQ79I - J - as it was a new question there are 22000 Nan's, different analysis 
# was used for checking the importance of these questions

# NQ62A -  Respect for the police is an important value for people to have? 19315 Nan's

# NQ62B - I feel an obligation to obey the law at all times? 19314 Nan's

# NG57AA - AE - they have many missing values over all (might be worth using for Q60)

# Q60 and Q61 will be removed as they will be new dependent values for other model

# remember to add Q62e i there is no difference

# Define ordinal columns and selected predictors
ordinal_columns <- c(  "Q1", "Q3C", "Q3H", "Q3J", "Q13", "Q15", "NQ21", "NQ57AA",  
                       "NQ57AB", "NQ57AC", "NQ57AD", "Q39A_2", "Q62A", "A120", 
                       "A121", "Q60", "Q61", "Q62E", "Q62F", "Q62TG", "Q62TI", "Q62TJ", 
                       "NQ62A", "NQ62B", "Q79A", "Q79B", "Q79C", "Q79D", "Q79E", "Q79F", 
                       "Q79G", "Q79H", "Q133", "NQ133", "Q3L", "Q62B", 
                       "Q62C", "Q62D", "Q62H") #this selection only contains 17000 rows

max_rows <- c("Q1", "Q3C", "Q3H", "Q3J", "Q13", "Q15",  "Q39A_2", "A120", 
              "A121","Q62A","Q62B","Q62C", "Q62D", "Q62E", "Q62F", "Q62TG",
              "Q62TI", "Q62TJ", "Q79A", "Q79B", "Q79C", "Q79D", "Q79E", 
              "Q79F", "Q79G", "Q79H", "Q133", "Q3L", "Q62H") # this contains 40709 rows

# shows which questions were removed
removed <- setdiff(ordinal_columns, max_rows)

selected_predictors <- c("NQ135BD", max_rows)


# Subset the data
main <- data[, selected_predictors]

# Filter out rows with zero values in any column
main <- main %>% filter_all(all_vars(. != 0))

testing <- main

corr_matrix <- cor(testing, method = "spearman")

print("Spearman's correlation matrix:")

#there are no highly  correlated pairs, thus multicolinearity is not an issue
high_correlation_pairs <- which(abs(corr_matrix) > 0.9 & corr_matrix != 1, arr.ind = TRUE)


                        

