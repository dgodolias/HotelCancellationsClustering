# Hotel Cancellations Clustering Report

**Demosgod Panagiotis (ID: 3220031)**

## 1. Overview
The goal of this work is to analyse hotel booking data to identify distinct customer clusters based on their booking behaviour and characteristics. Understanding these segments helps uncover cancellation patterns.

## 2. Methodology

### Feature Engineering
We generated several key features to capture customer behaviour:
- **Arrival Month** – to capture seasonality.
- **Total Guests** – sum of adults and children.
- **Is Family** – binary flag for bookings with children.
- **Total Nights** – sum of weekend and weekday nights.
- **Cancellation Ratio** – Laplace‑smoothed ratio of past cancellations for repeat guests.
- **Price per Person** – average price divided by total guests.
- **Meal Type & Room Type** – initial categorical variables.
- **Encoding** – one‑hot encoding of categorical variables (`market.segment.type`, `room.type`, `type.of.meal`). The dummy‑variable trap is avoided by using `drop_first=True`, therefore the first level of each category (`room.type_Room_Type 1`, `type.of.meal_Meal Plan 1`, `market.segment.type_Aviation`) is omitted.

### Preprocessing (R Implementation Details)
- **Data Cleaning**:
    - Removal of `date.of.reservation` column as it does not offer information for clustering.
    - Handling of infinite values in `Price_per_Person` by converting them to `NA`.
    - Imputation of missing values (`NA`) with the mean of each column.

- **Binning & Outlier Handling**:
    The `pmin()` function in R was used to cap extreme values, and `ifelse()` for grouping:
    1. **number.of.children**: Converted to binary (0 or 1+) -> `ifelse(number.of.children == 0, 0, 1)`
    2. **number.of.weekend.nights**: Capped at 3 -> `pmin(..., 3)`
    3. **number.of.week.nights**: Capped at 6 -> `pmin(..., 6)`
    4. **P.C & P.not.C**: Converted to binary (0 or 1+) -> `ifelse(..., 0, 1)`
    5. **Total_Guests**: Correction of zero values to 1 and capped at 3 -> `ifelse(Total_Guests == 0, 1, pmin(Total_Guests, 3))`
    6. **Total_Nights**: Rounded and capped at 8 -> `pmin(round(Total_Nights), 8)`

- **Encoding**:
    - Usage of `model.matrix(~ market.segment.type - 1, ...)` to create dummy variables.
    - Explicit removal of `market.segment.typeAviation` to avoid multicollinearity (dummy variable trap).
    - Removal of initial categorical columns `type.of.meal` and `room.type` as well as their derivatives, based on feature elimination analysis.

- **Scaling**:
    - Implementation of a custom `minmax_scale` function for normalization in the range **[-1, 1]**:
      $$x_{scaled} = 2 \cdot \frac{x - \min(x)}{\max(x) - \min(x)} - 1$$
    - Applied to all numeric variables using `mutate(across(everything(), minmax_scale))`.
    - The choice of the [-1, 1] range instead of the typical [0, 1] was made to center the data around zero, which often aids K-Means convergence.

### Feature Distributions (Before Clustering)
![Distribution Grid 1](R/visuals/dist_grid_1.png)
![Distribution Grid 2](R/visuals/dist_grid_2.png)
![Distribution Grid 3](R/visuals/dist_grid_3.png)
![Distribution Grid 4](R/visuals/dist_grid_4.png)

### Market Segment Distribution
![Market Segment Distribution](R/visuals/dist_market_segment.png)

## 3. Clustering
Two clustering algorithms were employed to validate the results:

1. **K‑Means** – iterative algorithm that minimizes within‑cluster distance.
   - Parameters: `centers = k` (tested 2‑10), `nstart = 10`.
2. **Hierarchical (Ward.D2)** – builds a hierarchy by merging closest points, using Euclidean distance.

### Determining the Optimal *k*
Silhouette Score, Elbow Method, NMI and ARI were used. Initial analysis suggested *k* = 3, but further optimisation identified *k* = 4 as the strongest structure.

### Feature Elimination (Stepwise Process)
A stepwise removal of noisy variables was performed. Each removal that improved the Silhouette Score was kept. The process shifted the optimal *k* from 3 to 4.

#### Steps Overview
1. **Step 1 (Initial)** – all variables, no scaling.
2. **Step 2 (Scaling)** – applied scaling.
3. **Step 3 (Categorical Removal)** – dropped `type.of.meal` and `room.type` dummies.
4. **Step 4 (Temporal Removal)** – removed `arrival.year` and `date.of.reservation`.
5. **Step 5 (Arrival_Month Removal)** –
   - Observation: mean `Arrival_Month` values were very close across clusters (range 6.41‑6.67).
   - Removing it increased the Silhouette Score more than any other variable.
   - Conclusion: `Arrival_Month` is noise and was removed.

![Final Evaluation](R/visuals/clustering_evaluation.png)
*The final evaluation plot showing *k* = 4 as the optimal solution.*

## 4. Algorithm Comparison
To confirm stability, K‑Means and Hierarchical results were compared using NMI and ARI.

| k | NMI | ARI | Comments |
|---|---|---|---|
| 2 | 0.9537 | 0.9800 | High agreement |
| **3** | **0.9920** | **0.9963** | **Very high agreement** |
| **4** | **0.9726** | **0.9890** | **Excellent agreement – optimal solution** |
| 5 | 0.7569 | 0.6157 | Significant drop |

We observe that for **k=3**, the NMI and ARI indices reach their maximum values, indicating that the two algorithms agree almost perfectly on this solution. However, the final choice of **k=4** was based on a combination of all indicators:

1. **Inertia**: The Inertia curve shows a clear "elbow" at k=4, suggesting that adding a 4th cluster significantly reduces intra-cluster variance.
2. **Silhouette Score**: It reaches its maximum value at **k=4** (approximately 0.49), indicating that the clusters are better separated and more compact compared to k=3.
3. **Stability**: At k=4, NMI and ARI indices remain extremely high (> 0.98), confirming that the solution is stable and commonly accepted by both algorithms.
4. **Drop at k=5**: For k=5, we observe a sharp drop in all indices (Silhouette, NMI, ARI), making k=4 the safest and optimal choice.

**Final Results (k = 4):**
- Silhouette Score ≈ 0.49 (high quality)
- NMI / ARI > 0.98 (excellent agreement)

## 5. Cluster Visuals & Profiles

Based on the final clustering, four distinct customer profiles were identified.

### Heatmap
![Heatmap](R/visuals/cluster_heatmap.png)

# Hotel Cancellations Clustering Report

**Demosgod Panagiotis (ID: 3220031)**

## 1. Overview
The goal of this work is to analyse hotel booking data to identify distinct customer clusters based on their booking behaviour and characteristics. Understanding these segments helps uncover cancellation patterns.

## 2. Methodology

### Feature Engineering
We generated several key features to capture customer behaviour:
- **Arrival Month** – to capture seasonality.
- **Total Guests** – sum of adults and children.
- **Is Family** – binary flag for bookings with children.
- **Total Nights** – sum of weekend and weekday nights.
- **Cancellation Ratio** – Laplace‑smoothed ratio of past cancellations for repeat guests.
- **Price per Person** – average price divided by total guests.
- **Meal Type & Room Type** – initial categorical variables.
- **Encoding** – one‑hot encoding of categorical variables (`market.segment.type`, `room.type`, `type.of.meal`). The dummy‑variable trap is avoided by using `drop_first=True`, therefore the first level of each category (`room.type_Room_Type 1`, `type.of.meal_Meal Plan 1`, `market.segment.type_Aviation`) is omitted.

### Preprocessing (R Implementation Details)
- **Data Cleaning**:
    - Removal of `date.of.reservation` column as it does not offer information for clustering.
    - Handling of infinite values in `Price_per_Person` by converting them to `NA`.
    - Imputation of missing values (`NA`) with the mean of each column.

- **Binning & Outlier Handling**:
    The `pmin()` function in R was used to cap extreme values, and `ifelse()` for grouping:
    1. **number.of.children**: Converted to binary (0 or 1+) -> `ifelse(number.of.children == 0, 0, 1)`
    2. **number.of.weekend.nights**: Capped at 3 -> `pmin(..., 3)`
    3. **number.of.week.nights**: Capped at 6 -> `pmin(..., 6)`
    4. **P.C & P.not.C**: Converted to binary (0 or 1+) -> `ifelse(..., 0, 1)`
    5. **Total_Guests**: Correction of zero values to 1 and capped at 3 -> `ifelse(Total_Guests == 0, 1, pmin(Total_Guests, 3))`
    6. **Total_Nights**: Rounded and capped at 8 -> `pmin(round(Total_Nights), 8)`

- **Encoding**:
    - Usage of `model.matrix(~ market.segment.type - 1, ...)` to create dummy variables.
    - Explicit removal of `market.segment.typeAviation` to avoid multicollinearity (dummy variable trap).
    - Removal of initial categorical columns `type.of.meal` and `room.type` as well as their derivatives, based on feature elimination analysis.

- **Scaling**:
    - Implementation of a custom `minmax_scale` function for normalization in the range **[-1, 1]**:
      $$x_{scaled} = 2 \cdot \frac{x - \min(x)}{\max(x) - \min(x)} - 1$$
    - Applied to all numeric variables using `mutate(across(everything(), minmax_scale))`.
    - The choice of the [-1, 1] range instead of the typical [0, 1] was made to center the data around zero, which often aids K-Means convergence.

### Feature Distributions (Before Clustering)
![Distribution Grid 1](R/visuals/dist_grid_1.png)
![Distribution Grid 2](R/visuals/dist_grid_2.png)
![Distribution Grid 3](R/visuals/dist_grid_3.png)
![Distribution Grid 4](R/visuals/dist_grid_4.png)

### Market Segment Distribution
![Market Segment Distribution](R/visuals/dist_market_segment.png)

## 3. Clustering
Two clustering algorithms were employed to validate the results:

1. **K‑Means** – iterative algorithm that minimizes within‑cluster distance.
   - Parameters: `centers = k` (tested 2‑10), `nstart = 10`.
2. **Hierarchical (Ward.D2)** – builds a hierarchy by merging closest points, using Euclidean distance.

### Determining the Optimal *k*
Silhouette Score, Elbow Method, NMI and ARI were used. Initial analysis suggested *k* = 3, but further optimisation identified *k* = 4 as the strongest structure.

### Feature Elimination (Stepwise Process)
A stepwise removal of noisy variables was performed. Each removal that improved the Silhouette Score was kept. The process shifted the optimal *k* from 3 to 4.

#### Steps Overview
1. **Step 1 (Initial)** – all variables, no scaling.
2. **Step 2 (Scaling)** – applied scaling.
3. **Step 3 (Categorical Removal)** – dropped `type.of.meal` and `room.type` dummies.
4. **Step 4 (Temporal Removal)** – removed `arrival.year` and `date.of.reservation`.
5. **Step 5 (Arrival_Month Removal)** –
   - Observation: mean `Arrival_Month` values were very close across clusters (range 6.41‑6.67).
   - Removing it increased the Silhouette Score more than any other variable.
   - Conclusion: `Arrival_Month` is noise and was removed.

![Final Evaluation](R/visuals/clustering_evaluation.png)
*The final evaluation plot showing *k* = 4 as the optimal solution.*

## 4. Algorithm Comparison
To confirm stability, K‑Means and Hierarchical results were compared using NMI and ARI.

| k | NMI | ARI | Comments |
|---|---|---|---|
| 2 | 0.9537 | 0.9800 | High agreement |
| **3** | **0.9920** | **0.9963** | **Very high agreement** |
| **4** | **0.9726** | **0.9890** | **Excellent agreement – optimal solution** |
| 5 | 0.7569 | 0.6157 | Significant drop |

We observe that for **k=3**, the NMI and ARI indices reach their maximum values, indicating that the two algorithms agree almost perfectly on this solution. However, the final choice of **k=4** was based on a combination of all indicators:

1. **Inertia**: The Inertia curve shows a clear "elbow" at k=4, suggesting that adding a 4th cluster significantly reduces intra-cluster variance.
2. **Silhouette Score**: It reaches its maximum value at **k=4** (approximately 0.49), indicating that the clusters are better separated and more compact compared to k=3.
3. **Stability**: At k=4, NMI and ARI indices remain extremely high (> 0.98), confirming that the solution is stable and commonly accepted by both algorithms.
4. **Drop at k=5**: For k=5, we observe a sharp drop in all indices (Silhouette, NMI, ARI), making k=4 the safest and optimal choice.

**Final Results (k = 4):**
- Silhouette Score ≈ 0.49 (high quality)
- NMI / ARI > 0.98 (excellent agreement)

## 5. Cluster Visuals & Profiles

Based on the final clustering, four distinct customer profiles were identified.

### Heatmap
![Heatmap](R/visuals/cluster_heatmap.png)

### Cancellation Rates
![Cancellation Rates](R/visuals/cancellation_rate.png)

### Cluster Comparison Grids
![Cluster Comparison Grid 1](R/visuals/cluster_comp_grid_1.png)
![Cluster Comparison Grid 2](R/visuals/cluster_comp_grid_2.png)
![Cluster Comparison Grid 3](R/visuals/cluster_comp_grid_3.png)
![Cluster Comparison Grid 4](R/visuals/cluster_comp_grid_4.png)
### Market Segment per Cluster
![Market Segment per Cluster](R/visuals/cluster_comp_market_segment.png)

We will attempt to name and observe them appropriately. For confirmation of key group characteristics, refer to the heatmap above:

### Cluster 0: Premium Families
- **Profile**: Families on vacation.
- **Key Traits**: Presence of **children**, higher total price, many special requests.
- **Booking Channel**: **Online**.
- **Risk**: High cancellation rate (~40%) due to family‑related uncertainties.
- **Explanation**:
   - **Unforeseen events** (illness, schedule changes) often force a full family cancellation.
   - **Cost**: As the most expensive segment, families monitor deals and may cancel if a better offer appears.
   - **Lead Time**: Early planning provides a large window for changes.

### Cluster 1: Offline / Early Birds
- **Profile**: Older travelers or groups that plan far ahead.
- **Key Traits**: Large **Lead Time**, lower average price.
- **Booking Channel**: **Offline** (travel agents / phone).
- **Risk**: Medium‑High cancellation rate (~30%).
- **Explanation**: Offline bookings usually involve a commitment (potential penalty), but long lead times increase the chance of plans changing.

### Cluster 2: Loyal Corporate Clients
- **Profile**: Business travelers and frequent guests.
- **Key Traits**: **Repeat** guests, low cancellation history, need for parking.
- **Booking Channel**: **Corporate**, Aviation, Complementary.
- **Risk**: Low cancellation rate (~11%).
- **Explanation**: Corporate travelers book for work purposes and rarely cancel. Repeaters are familiar with the hotel and have stable booking patterns.
    - **Parking**: Higher demand for parking spaces is observed, likely due to the use of company vehicles.

### Cluster 3: Typical Online Couples
- **Profile**: Leisure‑traveling couples.
- **Key Traits**: 2 adults, 0 children, high price per person.
- **Booking Channel**: Exclusively **Online**.
- **Risk**: High cancellation rate (~37%).
- **Explanation**: Online bookings often allow free cancellation. Couples may book early for security and cancel if a cheaper or more convenient option appears.

## 6. Conclusion
The clustering analysis successfully uncovered **four distinct customer profiles** with clear differences in booking behaviour and cancellation risk. The methodology—combining K‑Means and Hierarchical clustering with stepwise feature elimination—produced very high agreement between algorithms (NMI > 0.97, ARI > 0.98), confirming the robustness of the results.

### Key Findings
1. **Clustering Quality**: Final Silhouette Score (~0.49) indicates strong separation; incremental improvement from 0.27 (initial) to 0.49 (final) demonstrates the importance of careful feature selection.
2. **Risk Differentiation**: Cancellation rates range from 11% (Corporate) to 40% (Families), enabling targeted risk mitigation strategies.
3. **Channel vs. Risk**: Online bookings (Clusters 0 & 3) consistently show higher cancellation rates compared to Offline/Corporate channels.

### Practical Applications
- **Cluster 2 (Corporate)**: Prioritise loyalty programmes and corporate packages to retain this stable revenue base.
- **Clusters 0 & 3 (High Risk)**: Implement non‑refundable rates, deposits, or early‑check-in incentives to reduce cancellations.
- **Cluster 1 (Offline/Early Birds)**: Offer early‑booking discounts to turn long lead times into a competitive advantage.

### Final Remark
The analysis demonstrates that **booking channel, group composition (family vs couple vs corporate), and lead time** are the most decisive factors influencing cancellation behaviour. In contrast, features such as `room.type`, `meal.type`, and `arrival.month` proved to be noise, underscoring the importance of feature selection for high‑quality clustering.
