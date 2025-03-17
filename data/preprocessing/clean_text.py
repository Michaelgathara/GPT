import re
def clean_textdata(text):
    text = re.sub(r'={2,}', '', text)  
    
    text = re.sub(r'@-@', '', text)
    
    text = re.sub(r'\s+([.,])', r'\1', text)
    text = re.sub(r'\s+\'s', r"'s", text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text