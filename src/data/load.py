# Data bilan ishlash uchun asosiy kutubxona
import pandas as pd
# Fayl yo'llarini boshqarish
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2]/"data"/"raw"

def load_raw_data(filename: str) -> pd.DataFrame:

    """
    Load raw dataset from data/raw directory
    """

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")
    
    df = pd.read_csv(filepath)

    return df

