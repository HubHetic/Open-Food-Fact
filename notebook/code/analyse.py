import random


def df_nan(df, spe_colonne=True):
    """
    """
    line_with_nan = df.isna().apply(lambda x: 1 if sum(x) > 0 else 0, axis=1)
    print(f"Nombre de lignes avec nan: {line_with_nan.sum()}")
    if line_with_nan.sum():
        colonne_with_nan = df.isna().apply(lambda x: 1 if sum(x) > 0 else 0,
                                           axis=0)
        print(f"Nombre de colonnes avec Nan: {colonne_with_nan.sum()}")
        if spe_colonne and line_with_nan.sum():
            print("Colonnes avec des Nan et leur nombre:")
            print(df.isnull().sum().nlargest(sum(df.isnull().any())))


def number_duplicate(df, subset=[]):
    """
    """
    if subset == []:
        print(f"nombre de ligne duplicate: {df.duplicated(keep='last').sum()}")
    else:
        print(f"nombre de ligne duplicate:  {df.duplicated(keep='last', subset=subset).sum()}")


def typage_colonne(df):
    types_column = df.columns.to_series().groupby(df.dtypes).groups
    for key, value in types_column.items():
        print(key)
        for col in value:
            list_value = df[col].unique().tolist()
            random.shuffle(list_value)
            print(f"\t{col}: five random value {list_value[:5]}")


def basic_information(df):
    """
    """
    print(f"nombre de ligne: {df.shape[0]}")
    print(f"nombre de colonne: {df.shape[1]}")
    col_quantitatif = []
    quantitatif = 0
    col_qualititatif = []
    qualitatif = 0
    for col in df.columns:
        if len(df[col].unique()) < 21:
            qualitatif += 1
            col_qualititatif.append(col)
        else:
            quantitatif += 1
            col_quantitatif.append(col)
    if quantitatif:
        print("=================================")
        print(f"nombre de colonne quantitatif: {quantitatif}")
        print(f"colonne quantitatif: {col_quantitatif}")
    if qualitatif:
        print("=================================")
        print(f"nombre de colonne qualitatif: {qualitatif}")
        print(f"colonne qualitatif: {col_qualititatif}")
    return (col_quantitatif, col_qualititatif)
