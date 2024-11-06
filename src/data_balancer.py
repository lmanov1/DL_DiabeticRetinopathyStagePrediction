

def balance_diabetic_retinopathy_data(df):  # No need for case parameter
    class_counts = df['label'].value_counts()
    print("Original class distribution:\n", class_counts)  # CHANGE: Show original distribution

    # Determine the number of images in the No DR class (label 0)
    count_no_dr = class_counts.get(0, 0)  # Get the count of No DR class

    # Check if the count exceeds the threshold for case selection
    if count_no_dr > 30000:  # More than 30,000, treat it as case 1
        target_no_dr_count = 30000  # Target for case 1: larger distribution
    else:
        target_no_dr_count = 1200  # Target for case 2: smaller distribution

    # Downsample No DR (label 0) based on calculated target
    majority_class = df[df['label'] == 0]  # Identify the majority class
    if len(majority_class) > target_no_dr_count:  # CHANGE: Check if it exceeds the target
        majority_downsampled = resample(majority_class,
                                         replace=False,
                                         n_samples=target_no_dr_count,  # Keep specified max sample
                                         random_state=123)  # reproducible results
    else:
        majority_downsampled = majority_class  # No downsampling needed if within limits

    # Start creating a balanced DataFrame with downsampled majority class
    balanced_df = majority_downsampled  # CHANGE

    # Retain all other classes without changes
    for label in class_counts.index:  # CHANGE: Iterate through all class labels
        if label != 0:  # Skip majority class
            balanced_df = pd.concat([balanced_df, df[df['label'] == label]])

    print("New class distribution after balancing:\n", balanced_df['label'].value_counts())  # CHANGE: Show new distribution
    return balanced_df.reset_index(drop=True)  # CHANGE: Reset index for a clean DataFrame

# Example usage

balanced_df = balance_diabetic_retinopathy_data(df)

# add parameter to handle imbalanced data
# add parameter to handle upscaling images


# לבדוק ע"י הכפלת קבצים של level-ים קטנים
# check on the entire dataset
# check on the original dataset 
# check with the upscale/without
# kcs

