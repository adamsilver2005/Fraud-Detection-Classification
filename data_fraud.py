import pandas as pd 
import numpy as np 
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("train.csv")
    total = df['urgency_level'].isin([1,2,3]).sum()
    #All the urgency levels except for 0, (6571)
    #print(total)
    duplicate_count = df['nameOrig'].duplicated().sum()
    duplicate_rows = df.duplicated().sum()
    all_types = df['type'].drop_duplicates()
    #We have 5 types in total. Payment, Transfer, Cash out, Cash in, Debit 
    #print(all_types)
    #This is the count of same origin names (same accountt making different transactions)
    #8997
    #print(duplicate_count)
    #No duplicate data points. (0)
    #print(duplicate_rows)
#----------------------------

    multi_step_rows = df[df.groupby('nameOrig')['step'].transform('nunique') > 1]
    multi_step_rows = multi_step_rows.sort_values(['nameOrig', 'step'])

    print(multi_step_rows[['nameOrig', 'step', 'type', 'amount', 'urgency_level']])


    account_summary = (
    df.groupby('nameOrig')
      .agg(
          num_transactions=('step', 'size'),
          num_steps=('step', 'nunique'),
          max_urgency=('urgency_level', 'max'),
          avg_urgency=('urgency_level', 'mean')
      )
      .query('num_steps > 1')
      .sort_values(['num_steps', 'num_transactions'], ascending=False))

    print(account_summary)

    mask = df['urgency_level'].isin([1, 2, 3])

    meanurg = df.loc[mask, 'amount'].mean()
    medianurg = df.loc[mask, 'amount'].median()

    print(meanurg)
    print(medianurg)


if __name__ == "__main__":
    main()
