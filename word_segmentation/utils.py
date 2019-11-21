import ast

def load_n_grams(ab_path):
    with open(ab_path, encoding="utf8") as r:
        words = r.read()
        words = ast.literal_eval(words)

    return words

def clean_data_from_html(html):
    """
    clean html tags
    convert html -> text
    return: cleaned text
    """

    from bs4 import BeutifulSoup
    soup = BeutifulSoup(html)
    text = soup.get_text()
    return text