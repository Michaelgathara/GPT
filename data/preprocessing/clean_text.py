import re
import unicodedata

pattern_equals = re.compile(r'={2,}')  # repeated equal signs 
pattern_at = re.compile(r'@-@')        
pattern_space_before_punct = re.compile(r'\s+([.,!?])')  # extra space before punctuation
pattern_space_before_s = re.compile(r"\s+'s")  # space before "'s"
pattern_multi_space = re.compile(r'\s+')

def clean_textdata(text):
    text = unicodedata.normalize("NFKC", text)
    
    text = pattern_equals.sub('', text)
    text = pattern_at.sub('', text)
    
    # extra spaces before punctuation and contractions and stuff 
    text = pattern_space_before_punct.sub(r'\1', text)
    text = pattern_space_before_s.sub(r"'s", text)
    
    # normalizing whitespace and trimming stuff
    text = pattern_multi_space.sub(' ', text).strip()
    
    return text
