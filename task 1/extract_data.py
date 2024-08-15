import zipfile

# Define the path to the zip file
zip_file_path = "house-prices-advanced-regression-techniques.zip"
# Define the extraction directory (current directory in this case)
extract_dir = "."

# Extract the zip file
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete!")