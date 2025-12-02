Examples BEFORE merge -> train/val/test counts: 656383 81650 82371
Unique cont_ids BEFORE -> 265899 69684 70032
Client features rows: 3926686
Unique cont_ids in client_features: 3926686

Inner join matches (train): 656383
Left-join total rows: 656383 Rows with NULL static_features: 5772
Any duplicated cont_id in client_features? -> 0
After dropDuplicates, client_features rows: 3926686
train_before.cont_id type: [StringType()]
client_features.cont_id type: [StringType()]
After merge counts: 656383 81650 82371
Expected counts (before merge): 656383 81650 82371
Sample missing cont_ids (should be zero rows):

Missing count: 0
Saved merged files; counts: 656383 81650 82371
