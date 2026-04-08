import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Clean raw data
    """
    
    # Copy df
    df = df.copy()

    # Yashirin NaN qiymatlarni aniqlash
    df = df.replace(r'[^a-zA-A]', np.nan, regex=True)

    # NaN qiymatlarni tashlab yuborish
    for col in df.columns:
        na = (df[col].isna().sum()) / len(df[col])

        if na > 0.25:
            df.drop(col, axis=1, inplace=True)
    
    df.drop('veil-type', axis=1, inplace=True)
    
    return df