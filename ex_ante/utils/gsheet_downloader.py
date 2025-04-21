from io import BytesIO

import pandas as pd
import requests


def request_read_gsheet(link):
    # here the loaded csv from sheet directly is published with anyone to access the link, ONLY with ONE SHEET, we can set and restrict to Treeo only after login, but we need json api etc
    r = requests.get(link)
    print(f"reading data csv link of {link} and convert to df ")
    data = r.content
    df = pd.read_csv(BytesIO(data), index_col=0)
    return df
