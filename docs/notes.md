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

Για να μειώσουμε τη διάσταση και να βελτιώσουμε το clustering, ομαδοποιούμε τις εξής μεταβλητές:

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

## 8. Περαιτέρω Βελτιστοποίηση: Αφαίρεση Reservation_Weekday & Arrival_Year

Συνεχίζοντας τη διαδικασία βελτιστοποίησης για **k=4**, αφαιρέσαμε τις μεταβλητές:
- **Reservation_Weekday**: Ημέρα της εβδομάδας που έγινε η κράτηση
- **Arrival_Year**: Έτος άφιξης

### Αποτελέσματα μετά την αφαίρεση:
- **Silhouette Score**: Μεταπήδησε από **0.33** σε **0.44** (για k=4)
- **Διαχωρισμός**: Έγινε ακόμα πιο ξεκάθαρος μεταξύ:
  - k=3 και k=4
  - k=4 και k=5
- **Βέλτιστο k**: Παραμένει **4** αλλά με ακόμα υψηλότερη ποιότητα clustering

**Συμπέρασμα**: Οι χρονικές μεταβλητές (Reservation_Weekday, Arrival_Year) προσέθεταν θόρυβο στο clustering. Η αφαίρεσή τους βελτίωσε περαιτέρω την ποιότητα από 0.33 σε 0.44 (αύξηση 33%), καθιστώντας το k=4 ακόμα πιο έντονα ως τη βέλτιστη επιλογή.

## 9. Περαιτερω Βελτιστοποίηση: Αφαίρεση Arrival_Month & Week/Weekend Nights

Αναλύοντας τα αποτελέσματα του clustering και τις μέσες τιμές των μεταβλητών ανά cluster:

### Αφαίρεση Arrival_Month:
- **Παρατήρηση**: Οι μέσες τιμές του Arrival_Month σε όλα τα 4 clusters ήταν πολύ κοντά στο 6 (range: 6.41-6.67)
- **Feature Elimination Analysis**: Η αφαίρεση του Arrival_Month ανέβαζε το Silhouette περισσότερο από οποιαδήποτε άλλη μεταβλητή
- **Συμπέρασμα**: Το Arrival_Month είναι θόρυβος και δεν συμβάλλει στον διαχωρισμό των clusters
- **Ενέργεια**: Αφαιρέθηκε το Arrival_Month

### Αφαίρεση Week/Weekend Nights:
- **Λόγος**: Διατηρούμε το Total_Nights που είναι το άθροισμά τους και παρέχει την ίδια πληροφορία χωρίς να προσθέτει πολυπλοκότητα
- **Ενέργεια**: Αφαιρέθηκαν τα number.of.weekend.nights και number.of.week.nights

**Τελικό Αποτέλεσμα**: Το dataset μειώθηκε σε **19 features** (από 21), βελτιώνοντας την ποιότητα του clustering και απλοποιώντας το μοντέλο.

### Προσθήκη Market Segment Type:
- **Παρατήρηση**: Το market.segment.type ήταν one-hot encoded στην αρχή αλλά δεν συμπεριελήφθη στην τελική ανάλυση
- **Ενέργεια**: Προστέθηκαν ξεχωριστά διαγράμματα για τις 4 κατηγορίες (Complementary, Corporate, Offline, Online)
- **Αποτέλεσμα**: Τελικός αριθμός features για ανάλυση: **14 αριθμητικές** + **4 market.segment.type** (one-hot encoded)

## 10. Ερμηνεία Clusters (Profiles)

Με βάση το τελικό clustering (k=4) και την ανάλυση των μέσων όρων (Heatmap), προκύπτουν τα εξής προφίλ πελατών:

### Cluster 0: Standard Online Couples
- **Χαρακτηριστικά**: 2 ενήλικες, χωρίς παιδιά. Υψηλή τιμή ανά άτομο. Κρατήσεις αποκλειστικά Online.
- **Market Segment**: Online.
- **Προφίλ**: Ζευγάρια που ταξιδεύουν για αναψυχή (Leisure) και κλείνουν μόνοι τους μέσω πλατφορμών (Booking, Expedia).

### Cluster 1: Offline / Early Birds
- **Χαρακτηριστικά**: Το μεγαλύτερο Lead Time (κλείνουν πολύ νωρίς). Χαμηλότερη μέση τιμή.
- **Market Segment**: Offline.
- **Προφίλ**: Πελάτες μεγαλύτερης ηλικίας ή γκρουπ που κλείνουν μέσω ταξιδιωτικών γραφείων ή τηλεφώνου μήνες πριν την άφιξη, πετυχαίνοντας καλύτερες τιμές.

### Cluster 2: Premium Families
- **Χαρακτηριστικά**: Το μοναδικό cluster με παιδιά. Υψηλότερη συνολική τιμή (Average Price) και τα περισσότερα Special Requests.
- **Market Segment**: Online.
- **Προφίλ**: Οικογένειες που πηγαίνουν διακοπές. Είναι απαιτητικοί πελάτες (ειδικά αιτήματα) και πληρώνουν ακριβά λόγω μεγέθους δωματίου/ατόμων.

### Cluster 3: Loyal Corporate
- **Χαρακτηριστικά**: Επαναλαμβανόμενοι πελάτες (Repeated) με ιστορικό μη-ακυρώσεων (High P-not-C). Χαμηλή ή μηδενική τιμή (συμβόλαια/complementary) και ανάγκη για Parking.
- **Market Segment**: Corporate, Aviation, Complementary.
- **Προφίλ**: Επαγγελματίες (Business travelers), εταιρικοί πελάτες και πληρώματα αεροπορικών.

## 11. Αξιολόγηση μέσω Cancellation Rate

Η ποιότητα των clusters επιβεβαιώνεται πλήρως από τη μεταβλητή `Booking_Status` (Cancellation Rate), η οποία **δεν** χρησιμοποιήθηκε στην εκπαίδευση του αλγορίθμου, αλλά παρουσιάζει έντονη διακύμανση ανάμεσα στις ομάδες:

### Cluster 3 (11.4% - Low Risk)
- **Δικαιολόγηση**: Οι εταιρικοί πελάτες (Corporate) και οι τακτικοί επισκέπτες (Repeaters) έχουν σταθερό πρόγραμμα και ταξιδεύουν για δουλειά, άρα σπάνια ακυρώνουν. Είναι η "σταθερά" του ξενοδοχείου.

### Cluster 2 (39.9% - High Risk)
- **Δικαιολόγηση**: Οι οικογένειες έχουν το υψηλότερο ρίσκο ακύρωσης λόγω απρόβλεπτων παραγόντων (ασθένειες παιδιών) και υψηλού κόστους, που τους ωθεί να ψάχνουν για καλύτερες προσφορές μέχρι την τελευταία στιγμή.

### Cluster 0 (37.6% - High Risk)
- **Δικαιολόγηση**: Οι online κρατήσεις αναψυχής γίνονται συχνά με "δωρεάν ακύρωση". Οι τουρίστες κάνουν πολλαπλές κρατήσεις "για ασφάλεια" και ακυρώνουν εύκολα αν αλλάξουν γνώμη, καθώς δεν υπάρχει προσωπική δέσμευση.

### Cluster 1 (30.6% - Medium-High Risk)
- **Δικαιολόγηση**: Αν και οι Offline κρατήσεις (μέσω τηλεφώνου/πράκτορα) έχουν μεγαλύτερη δέσμευση από τις Online, το μεγάλο Lead Time αυξάνει στατιστικά την πιθανότητα να συμβεί κάτι που θα ανατρέψει το ταξίδι στο διάστημα που μεσολαβεί.