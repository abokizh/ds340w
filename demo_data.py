import pandas as pd

# Load the Excel file
excel_file = 'data/demo.xlsx'  # Replace with your file path
df = pd.read_excel(excel_file)

# Save the dataframe to a pickle file
pickle_file = 'data/demo.pkl'  # Replace with your desired output file name
df.to_pickle(pickle_file)

print(f"Excel file has been converted to pickle and saved as {pickle_file}.")
