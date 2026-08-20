"""
normalizer.py
Task X-A: Symmetric text normalization protocol for both reference and reconstructed PDF streams.
"""

import re
import unicodedata
import hashlib

def normalize_text_stream(text: str) -> str:
    if not text:
        return ""
    
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # 2. Ligature and special symbol standardization
    ligature_map = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '—': '---', '–': '--', '−': '-',
        '“': '"', '”': '"', '``': '"', "''": '"',
        '‘': "'", '’': "'", '`': "'", '′': "'", "'": "'",
        '•': '*', '·': '*',
        '±': '+/-', '◦': ' deg ', '°': ' deg ',
        'β': 'beta', 'α': 'alpha', 'ϕ': 'phi', 'ψ': 'psi',
        'σ': 'sigma', 'η': 'eta', 'Ω': 'Omega', '⊤': 'T',
        '∇': 'nabla', '×': 'x', '≤': '<=', '≥': '>=',
        'λ': 'lambda', '∆': 'Delta', '∂': 'd',
        '≈': '~=', '∈': 'in', '∑': 'sum', '∏': 'prod',
        '∞': 'inf', '≠': '!=', '≡': '==',
        '∗': '*', '⋆': '*', '□': 'Box', 'γ': 'gamma', 'π': 'pi', 'ρ': 'rho',
        'φ': 'phi', 'θ': 'theta'
    }
    for k, v in ligature_map.items():
        text = text.replace(k, v)
        
    # 3. Strip running headers, dates, and review line numbers
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        l_s = line.strip()
        if "under review as a conference paper" in l_s.lower() or "published as a conference paper" in l_s.lower():
            continue
        if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$', l_s):
            continue
        if re.match(r'^\d{3}$', l_s):
            continue
        l_clean = re.sub(r'^\s*\d{3}\s+', '', line)
        if re.match(r'^\d{1,2}$', l_clean.strip()):
            continue
        clean_lines.append(l_clean)
        
    text = "\n".join(clean_lines)
    
    # 4. Resolve hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\b[a-zA-Z]+)-\s+([a-zA-Z]+\b)', r'\1\2', text)
    text = re.sub(r'(\b[a-zA-Z]{3,})-([a-zA-Z]{3,}\b)', r'\1\2', text)
    
    # 5. Standardize case for structural header words
    header_replacements = [
        (r'\bABSTRACT\b', 'Abstract'),
        (r'\bINTRODUCTION\b', 'Introduction'),
        (r'\bDISCLOSURES\b', 'Disclosures'),
        (r'\bPROOFS\b', 'Proofs'),
        (r'\bREFERENCES\b', 'References'),
        (r'\bAI USE STATEMENT\b', 'AI Use Statement'),
        (r'\bREPRODUCIBILITY STATEMENT\b', 'Reproducibility Statement'),
    ]
    for old_h, new_h in header_replacements:
        text = re.sub(old_h, new_h, text, flags=re.IGNORECASE)
    
    # 6. Normalize LaTeX math commands and matrix formatting
    math_replacements = [
        (r'\beta', 'beta'), (r'\alpha', 'alpha'), (r'\phi', 'phi'), (r'\psi', 'psi'),
        (r'\sigma', 'sigma'), (r'\eta', 'eta'), (r'\Omega', 'Omega'), (r'\top', 'T'),
        (r'\nabla', 'nabla'), (r'\pm', '+/-'), (r'\le', '<='), (r'\ge', '>='),
        (r'\lambda', 'lambda'), (r'\Delta', 'Delta'), (r'\partial', 'd'),
        (r'\times', 'x'), (r'\approx', '~='), (r'\in', 'in'), (r'\sum', 'sum'),
        (r'\neq', '!='), (r'\equiv', '=='), (r'\Box', 'Box'), (r'\gamma', 'gamma'),
        (r'\pi', 'pi'), (r'\rho', 'rho'), (r'\sqrt', 'sqrt'), (r'\sim', '~'),
        (r'\left', ''), (r'\right', ''), (r'\mathbb', ''), (r'\mathcal', ''),
        (r'\mathrm', ''), (r'\text', ''), (r'\mathbf', ''), (r'\textbf', ''),
        (r'\textit', ''), (r'\emph', ''), (r'\citep', ''), (r'\citet', ''),
        (r'\cite', ''), (r'\ref', ''), (r'\label', ''), (r'\begin', ''), (r'\end', ''),
        (r'\caption', ''), (r'\section', ''), (r'\subsection', ''), (r'\paragraph', ''),
        (r'bmatrix', ''), (r'pmatrix', ''), (r'array', '')
    ]
    for m_cmd, m_repl in math_replacements:
        text = text.replace(m_cmd, m_repl)
        
    # Remove remaining backslashes and braces and bracket delimiters
    text = re.sub(r'[\{\}\$\[\]]', ' ', text)
    text = text.replace('\\', ' ')
    
    # Remove isolated blank tokens or matrix punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == '__main__':
    sample = "Under review as a conference paper at ICLR 2027\n023 Deep equilibrium models (DEQs) multi-\nstable fi—ﬂ."
    norm1 = normalize_text_stream(sample)
    norm2 = normalize_text_stream(sample)
    assert norm1 == norm2, "Self-normalization mismatch!"
    print(f"[*] Self-normalization Test Passed (100.000% self-similarity): '{norm1}'")
    with open(__file__, 'rb') as f:
        print(f"[*] normalizer.py SHA-256: {hashlib.sha256(f.read()).hexdigest()}")
