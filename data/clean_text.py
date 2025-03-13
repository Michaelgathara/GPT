import re
def clean_text():
    # Remove equal signs used for headings
    text = re.sub(r'={2,}', '', text)  # Remove multiple equal signs for headings (e.g., = Robert Boulter =)
    
    # Remove the "@-@" string
    text = re.sub(r'@-@', '', text)
    
    # Remove extra spaces before punctuation (e.g., before ',', '.', and "'s")
    text = re.sub(r'\s+([.,])', r'\1', text)  # Remove space before period and comma
    text = re.sub(r'\s+\'s', r"'s", text)  # Remove space before "'s"
    
    # Remove extra spaces in general
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text