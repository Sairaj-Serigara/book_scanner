from difflib import get_close_matches

# Known books database
BOOK_DB = [
    "Atomic Habits",
    "The Alchemist",
    "Rich Dad Poor Dad",
    "Ikigai",
    "The Psychology Of Money",
    "It Ends With Us",
    "The Help",
    "Where the Crawdads Sing"
]

def match_books(text):
    words = text.split()

    matches = set()

    for word in words:
        res = get_close_matches(word, BOOK_DB, n=1, cutoff=0.6)
        if res:
            matches.add(res[0])

    return list(matches)