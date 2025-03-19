import re

def clean_textdata(text):
    # Remove special placeholders
    text = re.sub(r'@\.\@|@,\@', '', text)

    # Fix inconsistent spacing before/after punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)

    # Normalize apostrophes (replace backslashes before 's)
    text = re.sub(r"\\'s", "'s", text)
    text = re.sub(r" 's", "'s", text) # replace space before 's
    
    # Remove unwanted wiki-style formatting
    text = re.sub(r"={1,}\s*([^=]+?)\s*={1,}", "", text)  # Remove section headers like = Title =, == Title ==, etc.
    text = re.sub(r"\[\[Category:.*?\]\]", "", text)  # Remove category tags
    text = re.sub(r"\[\[.*?\|", "", text)  # Remove links, keeping only the visible part
    text = re.sub(r"\]\]", "", text)  # Remove closing brackets for links
    
    # Remove multiple spaces and normalize line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r"\s?@-@\s?", "-", text)   # Fix hyphenated words (e.g., "state @-@ of" → "state-of")
    text = re.sub(r"\s?@,@\s?", ",", text)   # Fix thousands separators (e.g., "1 @,@ 000" → "1,000")

    # Normalize spacing around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"\(\s+", r"(", text)            # Fix space after '('
    text = re.sub(r"\s+\)", r")", text)            # Fix space before ')'

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text