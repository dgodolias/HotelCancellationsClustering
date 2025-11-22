
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

## 6. Data Binning (Ομαδοποίηση Τιμών)

Για να μειώσουμε το διάσταση και να βελτιώσουμε το clustering, ομαδοποιούμε τις εξής μεταβλητές:

### Μεταβλητές με Binning:
1. **number.of.children**: 0, 1+ (αντί για 0, 1, 2, 3, κλπ)
   - 0 παιδιά
   - 1+ παιδιά

2. **number.of.weekend.nights**: 0, 1, 2, 3+
   - Διατηρούμε τις τιμές 0, 1, 2
   - Όλες οι τιμές >= 3 γίνονται 3

3. **number.of.week.nights**: Κανονικά ως το 5, μετά 6+
   - Διατηρούμε τις τιμές 0-5
   - Όλες οι τιμές >= 6 γίνονται 6

4. **P.C (Previous Cancellations)**: 0, 1+
   - 0 cancellations
   - 1+ cancellations

5. **P.not.C (Previous Non-Cancellations)**: 0, 1+
   - 0 non-cancellations
   - 1+ non-cancellations

6. **Total_Guests**: 1, 2, 3+
   - Όσα είναι 0 γίνονται 1 (διόρθωση)
   - Διατηρούμε τις τιμές 1, 2
   - Όλες οι τιμές >= 3 γίνονται 3

7. **Total_Nights**: Κανονικά ως το 7, μετά 8+
   - Διατηρούμε τις τιμές 0-7
   - Όλες οι τιμές >= 8 γίνονται 8
   - Σημείωση: Round σε ακέραιο (αν υπάρχουν float values)

## 7. Feature Elimination & Optimization (k=3)

Για σταθερό **k=3** που έδινε μέχρι τώρα το καλύτερο Silhouette (NMI & ARI κοντά στο 1, πολύ καλά, όμως στο Elbow δεν μπορούσα να διαχωρίσω τον αγκώνα):

- Έτρεξα loop για να βρω όλες τις μεταβλητές που προκαλούν θόρυβο, αφαιρώντας τες μία-μία και ξαναμετρώντας το Silhouette.
- **Αποτέλεσμα**: Όλες οι κατηγορικές μεταβλητές που είχα φτιάξει από **room.type** και **type.of.meal** ήταν θόρυβος.
- **Ενέργεια**: Αφαίρεσα τις μεταβλητές room.type και type.of.meal.

### Αποτελέσματα μετά την αφαίρεση:
- **Silhouette Score**: Μεταπήδησε από **0.27** σε **0.33**
- **Βέλτιστο k**: Άλλαξε από **3** σε **4**
- **Οπτικοποίηση**: Πλέον οι 4 ομάδες φαίνονται ξεκάθαρα από:
  - Silhouette Score
  - NMI (Normalized Mutual Information)
  - ARI (Adjusted Rand Index)
  - **Elbow Method** (πλέον διακρίνεται καθαρά ο αγκώνας)

**Συμπέρασμα**: Η αφαίρεση των κατηγορικών μεταβλητών room.type και type.of.meal βελτίωσε σημαντικά την ποιότητα του clustering και αποκάλυψε την πραγματική δομή των δεδομένων με 4 ομάδες.

