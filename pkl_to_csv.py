import pickle
import pandas as pd

# Load the pickle file
with open('data/V2_test.pkl', 'rb') as file:
    data = pickle.load(file)

# If the data is a DataFrame, you can directly save it as CSV
if isinstance(data, pd.DataFrame):
    data.to_csv('data/demo.csv', index=False)

# If the data is not a DataFrame but another type (e.g., list of dictionaries), 
# you can convert it into a DataFrame first
else:
    df = pd.DataFrame(data)
    df.to_csv('data/demo.csv', index=False)
