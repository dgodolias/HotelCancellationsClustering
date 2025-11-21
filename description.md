Τα δεδομένα αναφέρονται σε ένα τυχαίο δείγμα κρατήσεων δωματίων σε κάποιο  ξενοδοχείο και στο αν η κράτηση τελικά ακυρώθηκε ή όχι. Η περιγραφή των μεταβλητών δίνεται παρακάτω.

 

H εργασία αφορά την ομαδοποίηση (clustering).  Αγνοώντας το αν μια κράτηση ακυρώθηκε ή όχι, θέλουμε να ομαδοποιήσουμε τις κρατήσεις με βάση τα χαρακτηριστικά τους. Ενδεχομένως να χρειάζεται να φτιάξετε και άλλες μεταβλητές συνδυάζοντας αυτές που έχετε.

Α. Πρέπει να χρησιμοποιήσετε τουλάχιστον δυο μεθόδους ομαδοποίησης και να τις συγκρίνετε.

Β. Πως θα περιγράφατε τις ομάδες που βρήκατε? Ποια clusters έχουν τις περισσότερες ακυρώσεις; Πόσο καλά είναι τα clusters που βρήκατε;

Γ. Πρέπει να εξηγήσετε γιατί επιλέξατε τις συγκεκριμένες μεταβλητές από τη λίστα που σας δίνεται και, γενικότερα, να περιγράψετε, να αιτιολογήσετε και να αξιολογήσετε με επαρκή λεπτομέρεια την προσέγγισή σας.

Θα πρέπει να ανεβάσετε δυο αρχεία (και όχι ένα zip):

To word/Latex or pdf με την αναφορά σας.
Τον R κώδικα που χρησιμοποιήσατε ώστε να μπορούμε να επιβεβαιώσουμε τα αποτελέσματα
Θα ακολουθήσει προφορική εξέταση σε τυχαία επιλεγμένες εργασίες σε ημερομηνία που θα ανακοινωθεί.

H  εργασία είναι ατομική

Τα δεδομένα υπάρχουν στο επισυναπτόμενο αρχείο EXCEL. Οι διαθέσιμες μεταβλητές είναι οι ακόλουθες:

Booking_ID--Unique identifier for each booking
number of adults--Number of adults included in the booking
number of children--Number of adults included in the booking
number of weekend nights--Number of weekend nights included in the booking
number of week nights--Number of week nights included in the booking
type of meal--Τype of meal included in the booking
car parking space--Indicates whether a car parking space was requested or included in the booking
room type--Type of room booked
lead time--Number of days between the booking date and the arrival date
market segment type--Type of market segment associated with the booking
repeated--Indicates whether the booking is a repeat booking
P-C--Number of previous bookings that were canceled by the customer prior to the current booking
P-not-C--Number of previous bookings not canceled by the customer prior to the current booking
average price--Average price associated with the booking
special requests--Number of special requests made by the guest
date of reservation--Date of the reservation
booking status--Status of the booking (canceled or not canceled)