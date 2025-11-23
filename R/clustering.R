# Load necessary libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(mclust)

# Create visuals directory
if (!dir.exists("visuals")) {
  dir.create("visuals")
}

save_plot <- function(filename, plot_obj) {
  ggsave(file.path("visuals", filename), plot = plot_obj, width = 10, height = 6)

}

print("Loading dataset...")
df <- read.csv("../project_cluster.csv")

print("Preprocessing data...")

df <- df %>%
  mutate(
    Total_Guests = number.of.adults + number.of.children,
    Is_Family = as.integer(number.of.children > 0),
    Total_Nights = number.of.weekend.nights + number.of.week.nights,
    Cancellation_Ratio = (P.C + 1) / (P.not.C + P.C + 2),
    Price_per_Person = average.price / Total_Guests
  )

df$Price_per_Person[is.infinite(df$Price_per_Person)] <- NA

df <- df %>% select(-date.of.reservation)

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

# Drop meal, room type, 
df_encoded <- df_encoded %>% select(-starts_with("type.of.meal"), -starts_with("room.type"))

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
    p <- ggplot(df_scaled, aes(x = .data[[col]])) +
      geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
      labs(title = paste("Distribution (Scaled):", col), x = "Scaled Value [-1, 1]", y = "Frequency") +
      theme_minimal() +
      theme(panel.grid.major.y = element_line(colour = "grey80"),
            panel.grid.major.x = element_blank(),
            panel.grid.minor = element_blank())

    # Removed print(p) to avoid Rplots.pdf
    save_plot(paste0("dist_", col, ".png"), p)
  }
}

# --- Clustering ---
print("Running clustering algorithms...")
library(gridExtra)  

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
  
  # Silhouette 
  ss_km <- silhouette(km$cluster, dist(X))
  kmeans_silhouette[i] <- mean(ss_km[, 3])
  
  # Hierarchical
  
  set.seed(42)
  idx <- sample(nrow(X), min(5000, nrow(X)))
  X_sub <- X[idx, ]
  
  hc <- hclust(dist(X_sub), method = "ward.D2")
  h_labels <- cutree(hc, k = k)
  
  ss_hc <- silhouette(h_labels, dist(X_sub))
  hierarchical_silhouette[i] <- mean(ss_hc[, 3])
  
  km_labels_sub <- kmeans_labels_list[[i]][idx]
  
  # Calculate NMI and ARI
  if (requireNamespace("aricode", quietly = TRUE)) {
    nmi_scores[i] <- aricode::NMI(km_labels_sub, h_labels)
  } else {
    nmi_scores[i] <- adjustedRandIndex(km_labels_sub, h_labels)
  }
  ari_scores[i] <- adjustedRandIndex(km_labels_sub, h_labels)
}

# Evaluation Plot - Combined
eval_df <- data.frame(
  k = k_range,
  Inertia = kmeans_inertia,
  Silhouette_KMeans = kmeans_silhouette,
  Silhouette_Hierarchical = hierarchical_silhouette,
  NMI = nmi_scores,
  ARI = ari_scores
)

# Create 4 subplots in 2x2 grid
p1 <- ggplot(eval_df, aes(x = k, y = Inertia)) + 
  geom_line() + 
  geom_point() + 
  labs(title = "K-Means Inertia", x = "Number of Clusters (k)", y = "Inertia") +
  theme_minimal()

p2 <- ggplot(eval_df, aes(x = k)) + 
  geom_line(aes(y = Silhouette_KMeans, color = "K-Means")) + 
  geom_point(aes(y = Silhouette_KMeans, color = "K-Means")) +
  geom_line(aes(y = Silhouette_Hierarchical, color = "Hierarchical")) + 
  geom_point(aes(y = Silhouette_Hierarchical, color = "Hierarchical")) +
  labs(title = "Silhouette Score", x = "Number of Clusters (k)", y = "Silhouette Score", color = "Method") +
  theme_minimal()

p3 <- ggplot(eval_df, aes(x = k, y = NMI)) + 
  geom_line() + 
  geom_point() + 
  labs(title = "NMI Score", x = "Number of Clusters (k)", y = "NMI") +
  theme_minimal()

p4 <- ggplot(eval_df, aes(x = k, y = ARI)) + 
  geom_line() + 
  geom_point() + 
  labs(title = "ARI Score", x = "Number of Clusters (k)", y = "ARI") +
  theme_minimal()

combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
ggsave("visuals/clustering_evaluation.png", combined_plot, width = 15, height = 12)

# --- Final Clustering (k=4) ---
print("Finalizing clusters (k=4)...")
k_optimal <- 4
set.seed(42)
km_final <- kmeans(X, centers = k_optimal, nstart = 10)
df_encoded$Cluster <- as.factor(km_final$cluster - 1)
df_encoded$booking.status <- booking_status

cluster_means <- df_encoded %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean))

print("Cluster Summary (Means):")
print(cluster_means)

print("Creating cluster comparison plots...")
variables_to_analyze <- c(
  'number.of.adults', 'number.of.children', 'car.parking.space', 
  'lead.time', 'repeated', 'P.C', 'P.not.C', 'average.price', 
  'special.requests', 'Total_Guests', 'Is_Family', 'Total_Nights', 
  'Cancellation_Ratio', 'Price_per_Person'
)

market_segment_cols <- names(df_encoded)[grepl("market.segment.type", names(df_encoded))]

all_vars <- c(variables_to_analyze, market_segment_cols)

cluster_colors <- c("steelblue", "coral", "lightgreen", "gold")

for (var in all_vars) {
  if (var %in% names(cluster_means)) {
    plot_data <- cluster_means %>%
      select(Cluster, all_of(var)) %>%
      rename(value = all_of(var))
    
    p <- ggplot(plot_data, aes(x = Cluster, y = value, fill = Cluster)) +
      geom_bar(stat = "identity", color = "black", alpha = 0.8) +
      scale_fill_manual(values = cluster_colors) +
      labs(title = paste("Cluster Comparison:", var), 
           x = "Cluster", 
           y = paste("Mean (", var, ")", sep = "")) +
      theme_minimal() +
      theme(legend.position = "none",
            panel.grid.major.y = element_line(colour = "grey80"),
            panel.grid.major.x = element_blank(),
            panel.grid.minor = element_blank())
    
    ggsave(file.path("visuals", paste0("cluster_comp_", gsub("\\\\.", "_", var), ".png")), 
           p, width = 10, height = 6)
  }
}

print("Cluster comparison plots created.")

cluster_summary <- cluster_means %>%
  mutate(Count = table(df_encoded$Cluster)[as.character(Cluster)]) %>%
  select(Count, everything(), -Cluster)
print("Creating heatmap...")
# Normalize cluster_summary for heatmap
cluster_summary_matrix <- as.matrix(cluster_summary)
cluster_summary_norm <- apply(cluster_summary_matrix, 2, function(x) {
  (x - min(x)) / (max(x) - min(x))
})
rownames(cluster_summary_norm) <- levels(df_encoded$Cluster)

heatmap_data <- as.data.frame(cluster_summary_norm) %>%
  mutate(Cluster = rownames(.)) %>%
  pivot_longer(-Cluster, names_to = "Variable", values_to = "Value")

p_heatmap <- ggplot(heatmap_data, aes(x = Cluster, y = Variable, fill = Value)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Value)), color = "black", size = 3) +  # Add annotations
  scale_fill_gradient2(low = "red", mid = "yellow", high = "green", midpoint = 0.5, name = "Normalized\nValue (0-1)") +
  labs(title = "Cluster Means Heatmap (Normalized)", x = "Cluster", y = "Variable") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave("visuals/cluster_heatmap.png", p_heatmap, width = 16, height = 6)
print("Heatmap created.")

cancel_rates <- df_encoded %>%
  group_by(Cluster) %>%
  summarise(Cancellation_Rate = mean(booking.status == "Canceled") * 100)

print("Cancellation Rates:")
print(cancel_rates)

p_cancel <- ggplot(cancel_rates, aes(x = Cluster, y = Cancellation_Rate, fill = Cluster)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.8) +
  scale_fill_manual(values = c("steelblue", "coral", "lightgreen", "gold")) +
  labs(title = "Cancellation Rate by Cluster", x = "Cluster", y = "Cancellation Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.major.y = element_line(colour = "grey80"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank())

save_plot("cancellation_rate.png", p_cancel)

# Create PDF report with all plots
print("Creating PDF report...")
library(magick)
library(grid)
library(gridExtra)

pdf_path <- "clustering_report.pdf"

plot_files <- list.files("visuals", pattern = "\\.png$", full.names = FALSE)
# Sort them to have a logical order: distributions first, then others
plot_files <- plot_files[order(!grepl("^dist_", plot_files), plot_files)]

pdf(pdf_path, width = 12, height = 8)

for (filename in plot_files) {
  filepath <- file.path("visuals", filename)
  
  img <- image_read(filepath)
  
  grid.newpage()
  grid.raster(img)
  
  grid.text(filename, x = 0.5, y = 0.98, 
            gp = gpar(fontsize = 10, col = "black"))
}

dev.off()

cat(paste("PDF report saved to:", pdf_path, "\n"))
print("R script completed successfully.")
