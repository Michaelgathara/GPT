import re
import unicodedata

pattern_equals = re.compile(r'={2,}')  # repeated equal signs 
pattern_at = re.compile(r'@-@')        
pattern_space_before_punct = re.compile(r'\s+([.,!?])')  # extra space before punctuation
pattern_space_before_s = re.compile(r"\s+'s")  # space before "'s"
pattern_multi_space = re.compile(r'\s+')

def clean_textdata(text):
    # VERSION 1
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

    # VERSION 2
    text = re.sub(r"\s?@-@\s?", "-", text)   # Fix hyphenated words (e.g., "state @-@ of" → "state-of")
    text = re.sub(r"\s?@,@\s?", ",", text)   # Fix thousands separators (e.g., "1 @,@ 000" → "1,000")

    # 2. Normalize spacing around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"\(\s+", r"(", text)            # Fix space after '('
    text = re.sub(r"\s+\)", r")", text)            # Fix space before ')'

    # 5. Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text