from .fineweb_data import get_fineweb_data
from .wikitext_data import get_wikitext_data
from .clean_text import clean_text

__all__ = ['get_fineweb_data', 'get_wikitext_data', 'clean_text']
# import using `from data import <function_name>`