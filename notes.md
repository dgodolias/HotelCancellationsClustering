
# Project Notes: Hotel Cancellations Clustering

## 1. Feature Engineering
- **Arrival Month**: Calculate from `date.of.reservation` + `lead.time`. Important for seasonality.
- **Total Guests**: `number.of.adults` + `number.of.children`.
- **Is_Family**: Binary feature (1 if `number.of.children` > 0).
- **Total Nights**: `weekend nights` + `week nights`. Guests staying 1 night behave differently than those staying 10.
- **Cancellation Ratio**: For returning customers.
    - Formula (smoothing): `(P_C + 1) / (P_not_C + P_C + 2)`
- **Price per Person**: `average price` / `Total Guests`.

## 2. Features to Drop / Ignore
- **Booking_ID**: Unique identifier per row — provides no information for clustering.
- **Booking Status**: Ignore during training (per project instructions). Keep aside for later evaluation/interpretation of clusters (e.g., "Cluster 1 has 80% cancellations").
- **Date of Reservation**: Use the original date string only to extract temporal features (month, year, weekday), then drop it.

## 3. Preprocessing
- **One-Hot Encoding**: Apply to categorical variables (`type of meal`, `room type`, `market segment`) so distance computations are correct.
- **Scaling**: Normalize numeric features (StandardScaler or MinMaxScaler) before clustering.

## 4. Algorithms
- K-Means
- Hierarchical Clustering

## 5. Evaluation Metrics
- Silhouette Score
- Elbow Method
- NMI (Normalized Mutual Information)
- ARI (Adjusted Rand Index)

