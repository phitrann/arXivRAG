# import scipdf
# import os

# os.environ["NO_PROXY"] = "localhost"

# import time

# file_path = 'samples/LLMRouting.pdf'

# start = time.time()
# article_dict = scipdf.parse_pdf_to_dict(file_path) # return dictionary
# print("Time taken: ", time.time() - start)

# print(article_dict.keys())

# import json
# with open('samples/LLMRouting.json', 'w') as f:
#     json.dump(article_dict, f)

# start = time.time()
# scipdf.parse_figures(file_path, output_folder='figures') # folder should contain only PDF files
# print("Time taken: ", time.time() - start)

from pylatexenc.latexencode import unicode_to_latex
from pylatexenc.latex2text import LatexNodes2Text
import re

def remove_ensuremath(latex_str):
    # Use a regular expression to remove the \ensuremath{...} part
    return re.sub(r'\\ensuremath\{([^}]*)\}', r'\1', latex_str)

latex = unicode_to_latex('max \u03b8 (q,li,j )\u2208Dpref log P \u03b8 (l i,j | q).(1)') # Hello, World!

print(remove_ensuremath(latex))