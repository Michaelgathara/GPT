import re

# Pre-compile regex patterns
remove_placeholders = re.compile(r'@\.\@|@,\@')
fix_spacing_punctuation = re.compile(r'\s*([.,!?;:])\s*')
normalize_apostrophes_1 = re.compile(r"\\'s")
normalize_apostrophes_2 = re.compile(r" 's")
remove_section_headers = re.compile(r"={1,}\s*([^=]+?)\s*={1,}")
remove_category_tags = re.compile(r"\[\[Category:.*?\]\]")
remove_links = re.compile(r"\[\[.*?\|")
remove_closing_brackets = re.compile(r"\]\]")
remove_multiple_spaces = re.compile(r'\s+')
fix_hyphenated_words = re.compile(r"\s?@-@\s?")
fix_thousands_separators = re.compile(r"\s?@,@\s?")
normalize_spacing_before_punctuation = re.compile(r"\s+([.,!?;:])")
fix_space_after_open_paren = re.compile(r"\(\s+")
fix_space_before_close_paren = re.compile(r"\s+\)")

def clean_textdata(text):
    # Remove special placeholders
    text = remove_placeholders.sub('', text)

    # Fix inconsistent spacing before/after punctuation
    text = fix_spacing_punctuation.sub(r'\1 ', text)

    # Normalize apostrophes (replace backslashes before 's)
    text = normalize_apostrophes_1.sub("'s", text)
    text = normalize_apostrophes_2.sub("'s", text)  # replace space before 's
    
    # Remove unwanted wiki-style formatting
    text = remove_section_headers.sub("", text)  # Remove section headers like = Title =, == Title ==, etc.
    text = remove_category_tags.sub("", text)  # Remove category tags
    text = remove_links.sub("", text)  # Remove links, keeping only the visible part
    text = remove_closing_brackets.sub("", text)  # Remove closing brackets for links
    
    # Remove multiple spaces and normalize line breaks
    text = remove_multiple_spaces.sub(' ', text).strip()

    text = fix_hyphenated_words.sub("-", text)   # Fix hyphenated words (e.g., "state @-@ of" → "state-of")
    text = fix_thousands_separators.sub(",", text)   # Fix thousands separators (e.g., "1 @,@ 000" → "1,000")

    # Normalize spacing around punctuation
    text = normalize_spacing_before_punctuation.sub(r"\1", text)  # Remove space before punctuation
    text = fix_space_after_open_paren.sub(r"(", text)            # Fix space after '('
    text = fix_space_before_close_paren.sub(r")", text)            # Fix space before ')'

    # Remove excessive whitespace
    text = remove_multiple_spaces.sub(" ", text).strip()
    
    return text