import pandas as pd

df = pd.read_csv(r'C:\Users\Harshil\Documents\aarush\hackenza2026_team_databaes\data\combined_data.csv', comment='#', skipinitialspace=True, on_bad_lines='skip')
print('Rows:', len(df))
print('\nColumns:', list(df.columns))
print('\nLanguages:')
print(df['language'].value_counts())
print('\nNativity:')
print(df['nativity_status'].value_counts())
print('\nSample:')
print(df.head(3))
