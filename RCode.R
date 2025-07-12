# Load Dataset
data <- read.csv("C:\\Users\\Shobah\\Downloads\\student-por.csv")

# Struktur Dataset
str(data)

# Melihat 6 baris pertama
head(data)

# Ringkasan Dataset
summary(data)

# Cek Missing Values
colSums(is.na(data))

# Cek Outliers 
detect_outliers_iqr <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  which(x < lower_bound | x > upper_bound)
}

numeric_vars <- sapply(data, is.numeric)
outliers_index_list <- lapply(data[, numeric_vars], detect_outliers_iqr)
sapply(outliers_index_list, length)

# Melihat distribusi dari nilai G3 (target)
table(data$G3)
hist(data$G3, main = "Distribusi Nilai G3", xlab = "Nilai G3", col = "skyblue")

# B. Pre-Processing

# Handling Outliers
# B. Pre-Processing

# 1. Handling Outliers 
winsorize <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}

numeric_vars <- sapply(data, is.numeric)
numeric_vars["G3"] <- FALSE  

data_winsorized <- data
data_winsorized[, numeric_vars] <- lapply(data[, numeric_vars], winsorize)

summary(data_winsorized)

# Transform
data_clean <- data_winsorized
numeric_vars <- sapply(data_clean, is.numeric)
numeric_vars["G3"] <- FALSE  

data_clean[, numeric_vars] <- lapply(data_clean[, numeric_vars], sqrt)
summary(data_clean)

# Konversi Variabel Kategorik ke Faktor
data_clean[] <- lapply(names(data_clean), function(colname) {
  x <- data_clean[[colname]]
  if (is.character(x) && colname != "G3") return(as.factor(x)) else return(x)
})
str(data_clean)

# Klasifikasi Target G3
data_clean$G3_class <- cut(data_clean$G3,
                           breaks = c(-1, 9.9, 13.9, 15.9, 17.9, 20),
                           labels = c("Fail", "Sufficient", "Good", "Very Good", "Excellent"),
                           ordered_result = TRUE)
table(data_clean$G3_class)

# Normalisasi Variabel Numerik
numeric_vars <- sapply(data_clean, is.numeric)
data_clean[, numeric_vars] <- scale(data_clean[, numeric_vars])

summary(data_clean)

# C. Seleksi Fitur (variabel)
# a. Variabel numerik
numeric_vars <- sapply(data_clean, is.numeric)
numeric_vars["G3"] <- FALSE  
numeric_vars["G3_class"] <- FALSE  

spearman_result <- sapply(names(data_clean)[numeric_vars], function(var) {
  cor(as.numeric(data_clean[[var]]), as.numeric(data_clean$G3_class), method = "spearman")
})

spearman_result <- sort(abs(spearman_result), decreasing = TRUE)
print(spearman_result)

# b. Variabel faktor
factor_vars <- sapply(data_clean, is.factor)
chisq_result <- sapply(names(data_clean)[factor_vars], function(var) {
  tbl <- table(data_clean[[var]], data_clean$G3_class)
  test <- chisq.test(tbl)
  return(test$p.value)
})

chisq_result <- sort(chisq_result)
print(chisq_result)

# c. Penggambungan variabel
selected_numeric <- names(spearman_result[spearman_result > 0.3])
selected_factor <- names(chisq_result[chisq_result < 0.01])

selected_features <- c(selected_numeric, selected_factor)
selected_features <- setdiff(selected_features, "G3_class")

print(selected_features)


# E. Uji Asumsi 
# a. Regresi Logistik Ordinal
# Uji Multikolinearitas (VIF)
library(car)

vif_data <- data_clean[, selected_features]
vif_data$G3_class <- data_clean$G3_class  

vif_model <- lm(as.numeric(G3_class) ~ ., data = vif_data)
vif_values <- vif(vif_model)
print("Nilai VIF:")
print(vif_values)


# LDA
# a. Uji Normalitas Multivariat (Mardia Test)
library(MVN)

numeric_features <- c("G1", "G2") 
mardia_result <- mvn(data_clean[, numeric_features], mvnTest = "mardia")

print(mardia_result$multivariateNormality)

# b. Uji Homogenitas Kovarian (Box's M Test)
library(biotools)

boxm_result <- boxM(data_clean[, numeric_features], grouping = data_clean$G3_class)
print(boxm_result)


# E. Pemodelan
# a. Regresi Logistik Ordinal (polr)
library(MASS)

model_polr <- polr(G3_class ~ ., data = data_clean[, c(selected_features, "G3_class")], Hess = TRUE)
summary(model_polr)
coef(model_polr)

# b. Linear Discriminant Analysis (lda)
model_lda <- lda(G3_class ~ ., data = data_clean[, c(selected_features, "G3_class")])

print(model_lda)
print(model_lda$scaling)


# F. Uji Signifikansi Variabel
# Regresi Logistik Ordinal
# a. Uji Serentak (Likelihood Ratio Test)
model_null <- polr(G3_class ~ 1, data = data_clean[, c(selected_features, "G3_class")], Hess = TRUE)

lr_stat <- 2 * (logLik(model_polr) - logLik(model_null))
df_lr <- attr(logLik(model_polr), "df") - attr(logLik(model_null), "df")
p_value_lr <- pchisq(lr_stat, df = df_lr, lower.tail = FALSE)

cat("Likelihood Ratio Statistic:", lr_stat, "\n")
cat("Degrees of Freedom:", df_lr, "\n")
cat("p-value:", p_value_lr, "\n")

# b. Uji Wald (uji parsial per-koefisien)
summary_polr <- summary(model_polr)
wald_stat <- (summary_polr$coefficients[, "Value"] / summary_polr$coefficients[, "Std. Error"])^2
wald_p_values <- 1 - pchisq(wald_stat, df = 1)

wald_table <- data.frame(
  Estimate = summary_polr$coefficients[, "Value"],
  StdError = summary_polr$coefficients[, "Std. Error"],
  p_value = round(wald_p_values, 4)
)
print(wald_table)

# LDA
# Uji Wilks' Lambda
eigen_values <- model_lda$svd^2
wilks_lambda <- prod(1 / (1 + eigen_values))

cat("Wilks' Lambda:", wilks_lambda, "\n")


# G. Evaluasi Model
# a. Prediksi dan Confusion Matrix
pred_polr <- predict(model_polr, newdata = data_clean[, selected_features])
pred_lda <- predict(model_lda)$class

conf_matrix_polr <- table(Predicted = pred_polr, Actual = data_clean$G3_class)
conf_matrix_lda <- table(Predicted = pred_lda, Actual = data_clean$G3_class)

cat("Confusion Matrix - Regresi Logistik Ordinal:\n")
print(conf_matrix_polr)

cat("\nConfusion Matrix - LDA:\n")
print(conf_matrix_lda)

# b. Akurasi, Precision, Recall, F1-score
evaluate_model <- function(conf_matrix) {
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- diag(conf_matrix) / rowSums(conf_matrix)
  recall <- diag(conf_matrix) / colSums(conf_matrix)
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  hasil <- data.frame(
    Class = names(precision),
    Precision = round(precision, 2),
    Recall = round(recall, 2),
    F1_Score = round(f1, 2)
  )
  hasil$Accuracy <- round(accuracy, 2)
  return(hasil)
}

cat("\nEvaluasi Regresi Logistik Ordinal:\n")
print(evaluate_model(conf_matrix_polr))

cat("\nEvaluasi LDA:\n")
print(evaluate_model(conf_matrix_lda))

# H. Interpretasi Hasil
# Regresi Logistik Ordinal (Koefisien & Odds Ratio)
summary(model_polr)
coef_polr <- coef(model_polr)
odds_ratio <- exp(coef_polr)

cat("Koefisien Regresi Logistik:\n")
print(coef_polr)

cat("\nOdds Ratio:\n")
print(round(odds_ratio, 2))

# LDA (Koefisien Fungsi Diskriminan)
cat("Koefisien Fungsi Diskriminan:\n")
print(model_lda$scaling)

library(ggplot2)
lda_pred <- predict(model_lda)
lda_df <- data.frame(LD1 = lda_pred$x[, 1], G3_class = data_clean$G3_class)

ggplot(lda_df, aes(x = LD1, fill = G3_class)) +
  geom_histogram(binwidth = 0.5, position = "identity", alpha = 0.6) +
  labs(title = "Distribusi Fungsi Diskriminan Pertama", x = "LD1", y = "Frekuensi")


