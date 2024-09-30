# Load required libraries
options(warn = -1)
library(tidyverse)
library(randomForest)
library(e1071)
library(readxl)
library(caTools)
library(glmnet)
library(rpart)
library(rpart.plot)

# Load the dataset
df <- read_excel('plane_ticket_price_dataset.xlsx')

# View the columns associated with the dataset
print(colnames(df))

# Basic information about the dataset
print(str(df))

# Check for missing values
print(colSums(is.na(df)))

# Remove missing values
df <- na.omit(df)


# Total Number of Flights for each Airline
df %>% ggplot(aes(x = "", fill = Airline)) +
  geom_bar(stat = "count") +
  coord_polar("y") +
  labs(title = "Airlines Distribution")

# Distribution of Price
ggplot(df, aes(x = Price)) +
  geom_histogram(binwidth = 1000, fill = "#070750", color = "#d61212", alpha = 0.7) +
  labs(title = "Distribution of Price", x = "Price", y = "Frequency")

# Boxplot of Price vs. Airline
ggplot(df, aes(x = Airline, y = Price)) +
  geom_boxplot(fill = "#0ea830", color = "#302828") +
  labs(title = "Boxplot of Price vs. Airline", x = "Airline", y = "Price")


# Convert Date_of_Journey to datetime format
df$Date_of_Journey <- as.Date(df$Date_of_Journey, format="%d/%m/%Y")

# Extract information from Date_of_Journey
df$Journey_day <- weekdays(df$Date_of_Journey)
df$Journey_month <- months(df$Date_of_Journey)
df$Journey_year <- as.numeric(format(df$Date_of_Journey, "%Y"))

# Convert Journey_day to categorical type
df$Journey_day <- factor(df$Journey_day, levels=c('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))

# Number of Flights each day of the week
df %>% ggplot(aes(x = Journey_day)) +
  geom_bar() +
  labs(title = "Number of Flights each day of the week")

# Convert categorical features to numerical using LabelEncoder
categorical_columns <- c('Airline', 'Source', 'Total_Stops', 'Destination')
for (col in categorical_columns) {
  df[[col]] <- as.numeric(factor(df[[col]]))
}

# Convert month to numerical values
day_mapping <- c('Monday' = 0, 'Tuesday' = 1, 'Wednesday' = 2, 'Thursday' = 3, 'Friday' = 4, 'Saturday' = 5, 'Sunday' = 6)
df$Journey_day <- as.numeric(factor(df$Journey_day, levels=names(day_mapping), labels=day_mapping))

# Drop unnecessary columns
df <- df %>%
  select(-c('Date_of_Journey', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info'))

# Split the data into features and target
X <- df %>%
  select(-Price)  # 'Price' is the target variable
y <- df$Price

# Split the data into training and testing sets
set.seed(42)
split <- sample.split(y, SplitRatio = 0.8)
X_train <- subset(X, split == TRUE)
y_train <- y[split == TRUE]
X_test <- subset(X, split == FALSE)
y_test <- y[split == FALSE]

# Handle missing values in the training set
X_train <- na.omit(X_train)
y_train <- na.omit(y_train)

# Combine X_train and y_train into a data frame
train_data <- cbind(X_train, Price = y_train)

# Linear Regression
linear_reg <- lm(Price ~ ., data = train_data)

# Decision Tree
tree_reg <- rpart(Price ~ ., data = train_data)

# Tree Plot for Decision Tree with enhanced appearance
prp(tree_reg, main = "Decision Tree Plot", extra = 1, fallen.leaves = FALSE,
    branch.lty = 3, shadow.col = "gray", box.col = "#128cb4", cex = 0.8)

# Lasso Regression
lasso_reg <- glmnet(as.matrix(X_train), y_train, alpha = 1)

# Random Forest
rf_reg <- randomForest(Price ~ ., data = train_data)

# Ridge Regression
ridge_reg <- glmnet(as.matrix(X_train), y_train, alpha = 0)

# Make predictions
linear_reg_preds <- predict(linear_reg, newdata = X_test)
ridge_reg_preds <- predict(ridge_reg, newx = as.matrix(X_test))
lasso_reg_preds <- predict(lasso_reg, newx = as.matrix(X_test))
rf_reg_preds <- predict(rf_reg, newdata = X_test)
tree_reg_preds <- predict(tree_reg, newdata = X_test)

# Calculate and print RMSE for each model
calculate_rmse <- function(predictions, actual) {
  mse <- mean((actual - predictions)^2)
  rmse <- sqrt(mse)
  return(rmse)
}

linear_reg_rmse <- calculate_rmse(linear_reg_preds, y_test)
ridge_reg_rmse <- calculate_rmse(ridge_reg_preds, y_test)
lasso_reg_rmse <- calculate_rmse(lasso_reg_preds, y_test)
rf_reg_rmse <- calculate_rmse(rf_reg_preds, y_test)
tree_reg_rmse <- calculate_rmse(tree_reg_preds, y_test)

cat("\nRoot Mean Squared Error (RMSE):\n")
cat("\tLinear Regression RMSE:", linear_reg_rmse, "\n")
cat("\tRidge Regression RMSE:", ridge_reg_rmse, "\n")
cat("\tLasso Regression RMSE:", lasso_reg_rmse, "\n")
cat("\tRandom Forest Regression RMSE:", rf_reg_rmse, "\n")
cat("\tDecision Tree Regression RMSE:", tree_reg_rmse, "\n")


# Visualizations
# Create a bar chart for RMSE comparison
rmse_values <- c(linear_reg_rmse, ridge_reg_rmse, lasso_reg_rmse, rf_reg_rmse, tree_reg_rmse)
models <- c("Linear", "Ridge", "Lasso", "Random Forest", "Decision Tree")

barplot(rmse_values, names.arg = models, col = "#df3551", main = "RMSE Comparison of Regression Models",
        ylab = "Root Mean Squared Error (RMSE)")


# Variable Importance Plot for Random Forest
var_importance <- importance(rf_reg)

# Create a variable importance plot
barplot(var_importance[, "IncNodePurity"], names.arg = rownames(var_importance),
        col = "chartreuse", main = "Variable Importance Plot for Random Forest",
        ylab = "IncNodePurity")

# Residuals for Linear Regression
linear_reg_residuals <- residuals(linear_reg)

# Create a residual plot
plot(linear_reg_residuals, col = "blue", pch = 16, xlab = "Index", ylab = "Residuals",
     main = "Residual Plot for Linear Regression")
abline(h = 0, col = "red", lty = 2)


