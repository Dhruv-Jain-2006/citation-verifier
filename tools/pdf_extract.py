import re
import fitz  # PyMuPDF
from typing import Dict, List

# Regex to catch numeric tags like [12], [1, 2], [12-15]
REGEX_NUMERIC_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+|\s*-\s*\d+)*\s*\]")

# Regex to catch (Author, 2023) or (Author et al., 2023) or (Author & Author, 2019a)
REGEX_AUTHOR_YEAR_CITATION = re.compile(
    r"\(\s*[A-Z][a-zA-Z]+(?:.*?)\s*,\s*(?:19|20)\d{2}[a-z]?\s*\)"
)

# Regex to catch a "References" or "Bibliography" header block
REGEX_REF_HEADER = re.compile(r"^\s*(?:[0-9]+\.?\s*)?(?:References|Bibliography)\s*$", re.IGNORECASE)


def extract_pdf_data(pdf_path: str) -> Dict[str, List[str] | str]:
    """
    Deterministically parses a PDF to isolate paragraphs containing citations 
    and extract the full References bibliography section.
    
    Returns:
        dict: {
            "citation_paragraphs": List[str], # Blocks of text containing citations
            "references_section": str,       # The raw bibliography block
            "full_text": str                 # Entire raw string of the document (fallback)
        }
    """
    doc = fitz.open(pdf_path)
    
    full_text_blocks = []
    citation_paragraphs = []
    references_blocks = []
    
    in_references_section = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        # "blocks" mode separates paragraphs well, preventing huge word salads
        blocks = page.get_text("blocks")
        
        for block in blocks:
            text = block[4].strip() # PyMuPDF blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
            
            if not text:
                continue
            
            # Clean up newlines inside paragraphs to normalize text
            cleaned_text = " ".join(text.split())
            full_text_blocks.append(cleaned_text)
            
            # 1. Flip toggle if we hit the references section
            if not in_references_section and REGEX_REF_HEADER.match(cleaned_text):
                in_references_section = True
                continue # Skip adding the header string itself
            
            # 2. If we are in the References section, just collect the bibliography
            if in_references_section:
                references_blocks.append(cleaned_text)
            else:
                # 3. If we are in the body, hunt for citations 
                # (Ignore blocks that are too small to be meaningful claims, e.g. < 5 words)
                if len(cleaned_text.split()) > 5:
                    has_numeric = bool(REGEX_NUMERIC_CITATION.search(cleaned_text))
                    has_author = bool(REGEX_AUTHOR_YEAR_CITATION.search(cleaned_text))
                    
                    if has_numeric or has_author:
                        citation_paragraphs.append(cleaned_text)

    # Fallback to last 15% of document if the regex missed the "References" header
    references_text = "\n\n".join(references_blocks)
    if not references_text:
        # Emergency fallback: approximate by taking the last chunks of text
        fallback_cutoff = int(len(full_text_blocks) * 0.85)
        references_text = "\n\n".join(full_text_blocks[fallback_cutoff:])
        # Remove those from citation_paragraphs if they falsely swept them up
        citation_paragraphs = [p for p in citation_paragraphs if p not in full_text_blocks[fallback_cutoff:]]

    # Deduplicate repeated citation paragraphs
    citation_paragraphs = list(dict.fromkeys(citation_paragraphs))

    doc.close()

    return {
        "citation_paragraphs": citation_paragraphs,
        "references_section": references_text,
        "full_text": "\n\n".join(full_text_blocks)
    }

if __name__ == "__main__":
    # Smoke Test stub
    print("PDF Extractor initialized. Call extract_pdf_data(path) to use.")
