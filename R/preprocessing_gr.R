# Load necessary libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(mclust)

# Set working directory to the script location (if running interactively)
# setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Create visuals directory
if (!dir.exists("visuals")) {
  dir.create("visuals")
}

save_plot <- function(filename) {
  ggsave(file.path("visuals", filename), width = 10, height = 6)
  print(paste("Saved plot:", filename))
}

# --- Load Data ---
print("Loading dataset...")
df <- read.csv("../project_cluster.csv")

# --- Preprocessing ---
print("Preprocessing data...")

# Feature Engineering
df <- df %>%
  mutate(
    Total_Guests = number.of.adults + number.of.children,
    Is_Family = as.integer(number.of.children > 0),
    Total_Nights = number.of.weekend.nights + number.of.week.nights,
    Cancellation_Ratio = (P.C + 1) / (P.not.C + P.C + 2),
    Price_per_Person = average.price / Total_Guests
  )

# Handle division by zero or inf in Price_per_Person
df$Price_per_Person[is.infinite(df$Price_per_Person)] <- NA

# Date fixing (simplified for R - just dropping for now as we don't use it for clustering)
df <- df %>% select(-date.of.reservation)

# Value capping/binning
df <- df %>%
  mutate(
    number.of.children = ifelse(number.of.children == 0, 0, 1),
    number.of.weekend.nights = pmin(number.of.weekend.nights, 3),
    number.of.week.nights = pmin(number.of.week.nights, 6),
    P.C = ifelse(P.C == 0, 0, 1),
    P.not.C = ifelse(P.not.C == 0, 0, 1),
    Total_Guests = ifelse(Total_Guests == 0, 1, pmin(Total_Guests, 3)),
    Total_Nights = pmin(round(Total_Nights), 8)
  )

# Save booking status and drop unused columns
booking_status <- df$booking.status
df <- df %>% select(-Booking_ID, -booking.status)

# One-hot encoding for market.segment.type
# R handles factors automatically in many cases, but for clustering we need numeric
# We'll use model.matrix to create dummy variables
dummy <- model.matrix(~ market.segment.type - 1, data = df)
df_encoded <- cbind(df %>% select(-market.segment.type), dummy)

# Drop meal, room type, week related columns
df_encoded <- df_encoded %>% select(-starts_with("type.of.meal"), -starts_with("room.type"), -number.of.weekend.nights, -number.of.week.nights)

# Fill NaNs (with mean)
df_encoded <- df_encoded %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Scaling (MinMax -1 to 1)
# Function for MinMax scaling to [-1, 1]
minmax_scale <- function(x) {
  min_x <- min(x)
  max_x <- max(x)
  return(2 * ((x - min_x) / (max_x - min_x)) - 1)
}

df_scaled <- df_encoded %>%
  mutate(across(everything(), minmax_scale))

print("Data processed and scaled.")

# --- Plotting Distributions ---
print("Generating distribution plots...")
numeric_cols <- c(
  'Price_per_Person', 'Cancellation_Ratio', 'Total_Nights', 'Is_Family', 
  'Total_Guests', 'special.requests', 'average.price', 'P.not.C', 'P.C', 
  'repeated', 'lead.time', 'car.parking.space', 'number.of.week.nights', 
  'number.of.weekend.nights', 'number.of.children', 'number.of.adults'
)

for (col in numeric_cols) {
  if (col %in% names(df_scaled)) {
    p <- ggplot(df_scaled, aes_string(x = col)) +
      geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
      labs(title = paste("Distribution (Scaled):", col), x = "Scaled Value [-1, 1]", y = "Frequency") +
      theme_minimal()
    
    # We need to print the plot object for ggsave to pick it up or pass it explicitly
    print(p) 
    save_plot(paste0("dist_", col, ".png"))
  }
}

# --- Clustering ---
print("Running clustering algorithms...")
X <- as.matrix(df_scaled)
k_range <- 2:10

kmeans_silhouette <- numeric(length(k_range))
hierarchical_silhouette <- numeric(length(k_range))
nmi_scores <- numeric(length(k_range))
ari_scores <- numeric(length(k_range))
kmeans_inertia <- numeric(length(k_range))

kmeans_labels_list <- list()
hierarchical_labels_list <- list()

for (i in seq_along(k_range)) {
  k <- k_range[i]
  
  # K-Means
  km <- kmeans(X, centers = k, nstart = 10)
  kmeans_labels_list[[i]] <- km$cluster
  kmeans_inertia[i] <- km$tot.withinss
  
  # Silhouette (using cluster package)
  ss_km <- silhouette(km$cluster, dist(X))
  kmeans_silhouette[i] <- mean(ss_km[, 3])
  
  # Hierarchical
  # For large datasets, hclust on full distance matrix is slow. 
  # We'll use a sample or optimized method if needed, but for ~36k rows it might be heavy.
  # Let's use 'agnes' or standard 'hclust' but be aware of memory.
  # Actually, calculating dist(X) for 36k rows is huge (36000^2 doubles ~ 10GB).
  # Python's AgglomerativeClustering might be more optimized or the user had enough RAM.
  # We will skip Hierarchical for the full dataset in R to avoid crashing, 
  # OR we can use a subsample for the hierarchical part just for demonstration.
  # Let's try on full, but if it crashes, we know why. 
  # To be safe, let's use a subsample of 5000 for hierarchical metrics.
  
  # Subsample for hierarchical to avoid memory issues
  set.seed(42)
  idx <- sample(nrow(X), min(5000, nrow(X)))
  X_sub <- X[idx, ]
  
  hc <- hclust(dist(X_sub), method = "ward.D2")
  h_labels <- cutree(hc, k = k)
  
  ss_hc <- silhouette(h_labels, dist(X_sub))
  hierarchical_silhouette[i] <- mean(ss_hc[, 3])
  
  # Comparison (on subsample)
  km_labels_sub <- kmeans_labels_list[[i]][idx]
  
  nmi_scores[i] <- adjustedRandIndex(km_labels_sub, h_labels) # mclust uses ARI, NMI is separate
  # mclust::adjustedRandIndex is actually ARI. 
  # NMI is not standard in base R, aricode package has NMI.
  # We'll stick to ARI for now as it's available in mclust.
  ari_scores[i] <- adjustedRandIndex(km_labels_sub, h_labels)
}

# Evaluation Plot
eval_df <- data.frame(
  k = k_range,
  Inertia = kmeans_inertia,
  Silhouette_KMeans = kmeans_silhouette,
  Silhouette_Hierarchical = hierarchical_silhouette,
  ARI = ari_scores
)

# Plotting evaluation
p1 <- ggplot(eval_df, aes(x = k, y = Inertia)) + geom_line() + geom_point() + ggtitle("K-Means Inertia")
p2 <- ggplot(eval_df, aes(x = k)) + 
  geom_line(aes(y = Silhouette_KMeans, color = "K-Means")) + 
  geom_point(aes(y = Silhouette_KMeans, color = "K-Means")) +
  geom_line(aes(y = Silhouette_Hierarchical, color = "Hierarchical")) + 
  geom_point(aes(y = Silhouette_Hierarchical, color = "Hierarchical")) +
  ggtitle("Silhouette Score")

ggsave("visuals/clustering_evaluation_inertia.png", p1)
ggsave("visuals/clustering_evaluation_silhouette.png", p2)

# --- Final Clustering (k=4) ---
print("Finalizing clusters (k=4)...")
k_optimal <- 4
set.seed(42)
km_final <- kmeans(X, centers = k_optimal, nstart = 10)
df_encoded$Cluster <- as.factor(km_final$cluster - 1) # 0-indexed to match Python
df_encoded$booking.status <- booking_status

# Cluster Summary
cluster_means <- df_encoded %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean))

print("Cluster Summary (Means):")
print(cluster_means)

# Cancellation Rates
cancel_rates <- df_encoded %>%
  group_by(Cluster) %>%
  summarise(Cancellation_Rate = mean(booking.status == "Canceled") * 100)

print("Cancellation Rates:")
print(cancel_rates)

p_cancel <- ggplot(cancel_rates, aes(x = Cluster, y = Cancellation_Rate, fill = Cluster)) +
  geom_bar(stat = "identity") +
  labs(title = "Cancellation Rate by Cluster", y = "Cancellation Rate (%)")

save_plot("cancellation_rate.png")

print("R script completed successfully.")
