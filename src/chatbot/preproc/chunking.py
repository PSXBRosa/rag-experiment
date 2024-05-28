

def chunk_text_column(df, column_name, n=10):
    """
    Breaks the contents of a specified column in the DataFrame into chunks of sentences.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be chunked.
        n (int): The chunk size.

    Returns:
        pd.DataFrame: A new DataFrame with the original content replaced by lists of sentences.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    newdf = df.copy()  # to avoid changing the original df
    # split the entries into chunks of size n. the last chunk may be bigger to cover the whole doc.
    newdf[column_name] = newdf[column_name].apply(
            lambda x: [x[i:i+n] for i in range(0, len(x) - n, n)] + [x[len(x) - n:]] if isinstance(x, str) else x
    )
    return newdf.explode(column_name, ignore_index=True)
