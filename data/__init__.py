from .fineweb_data import get_fineweb_data
from .wikitext_data import get_wikitext_data
from .utils import save_data, load_data
__all__ = ['get_fineweb_data', 'get_wikitext_data', 'save_data', 'load_data']
# import using `from data import <function_name>`