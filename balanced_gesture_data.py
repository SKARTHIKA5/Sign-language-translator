import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
file_path = "C:/Users/dell/Videos/sign_language/gesture_data.csv"  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Separate features and target labels
X = data.drop(columns=["gesture"])  # Assuming "gesture" is the label column
y = data["gesture"]

# Apply Random Over-Sampling to balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Combine back into a DataFrame
balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
balanced_data["gesture"] = y_resampled  # Add the target column back

# Save the balanced dataset
balanced_file_path = "C:/Users/dell/Videos/sign_language/balanced_gesture_data.csv"
balanced_data.to_csv(balanced_file_path, index=False)

print(f"Balanced dataset saved as {balanced_file_path}")
