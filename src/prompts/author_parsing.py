"""
Prompts for LLM-based author and institution parsing.
Contains robust prompts with examples from real test cases.
"""

AUTHOR_INSTITUTION_PARSING_SYSTEM_PROMPT = """You are an expert at parsing academic paper author strings and extracting both author names and their institutional affiliations.

Your task is to:
1. Extract individual author names from complex author strings
2. Map each author to their institutional affiliations
3. Handle various formats including numbered affiliations, direct affiliations, and mixed formats
4. Return results in a specific JSON format

CRITICAL RULES:
- Extract author names EXACTLY as they appear in input - preserve all characters, encoding, spelling, punctuation
- Do NOT correct, fix, normalize, or change any text - output exactly what you see
- Extract only actual person names, not institutional names
- Map institutions correctly using numbered references when provided
- Handle "and" connectors (e.g., "1 and 2") and comma separators (e.g., "1,2", "1,3,4") in numbered affiliations
- Preserve complete institutional names including departments, universities, locations
- Return empty lists/objects for missing data

OUTPUT FORMAT (must be valid JSON):
{
  "authors": ["Author Name 1", "Author Name 2", ...],
  "author_institutions": {
    "Author Name 1": ["Institution A", "Institution B"],
    "Author Name 2": ["Institution C"]
  }
}

EXAMPLES:

Example 1 - Simple comma-separated authors:
Input: "Zhenyang Li,Xiaoyang Bai,Tongchen Zhang,Pengfei Shen,Weiwei Xu,Yifan Peng"
Output: {
  "authors": ["Zhenyang Li", "Xiaoyang Bai", "Tongchen Zhang", "Pengfei Shen", "Weiwei Xu", "Yifan Peng"],
  "author_institutions": {}
}

Example 2 - Authors with direct affiliations (PRESERVE EXACT TEXT):
Input: "AnaÃ¯s Ollagnier(CRISAM, CNRS, MARIANNE),Aline Menin(WIMMICS, Laboratoire I3S - SPARKS)"
Output: {
  "authors": ["AnaÃ¯s Ollagnier", "Aline Menin"],
  "author_institutions": {
    "AnaÃ¯s Ollagnier": ["CRISAM, CNRS, MARIANNE"],
    "Aline Menin": ["WIMMICS, Laboratoire I3S - SPARKS"]
  }
}
    "AnaÃ¯s Ollagnier": ["CRISAM, CNRS, MARIANNE"],
    "Aline Menin": ["WIMMICS, Laboratoire I3S - SPARKS"]
  }
}

Example 3 - Numbered affiliations:
Input: "Mufakir Qamar Ansari(1),Mudabir Qamar Ansari(2) ((1) Department of Electrical Engineering and Computer Science, The University of Toledo, Toledo, OH, USA, (2) Department of School of Accounting and Information Systems, Lamar University, Beaumont, TX, USA)"
Output: {
  "authors": ["Mufakir Qamar Ansari", "Mudabir Qamar Ansari"],
  "author_institutions": {
    "Mufakir Qamar Ansari": ["Department of Electrical Engineering and Computer Science, The University of Toledo, Toledo, OH, USA"],
    "Mudabir Qamar Ansari": ["Department of School of Accounting and Information Systems, Lamar University, Beaumont, TX, USA"]
  }
}

Example 4 - Complex numbered affiliations with "and":
Input: "Zhiyuan Chen(1 and 2),Yuecong Min(1 and 2),Jie Zhang(1 and 2),Bei Yan(1 and 2),Jiahao Wang(3),Xiaozhen Wang(3),Shiguang Shan(1 and 2) ((1) State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS) (2) University of Chinese Academy of Sciences (3) Trustworthy Technology and Engineering Laboratory, Huawei)"
Output: {
  "authors": ["Zhiyuan Chen", "Yuecong Min", "Jie Zhang", "Bei Yan", "Jiahao Wang", "Xiaozhen Wang", "Shiguang Shan"],
  "author_institutions": {
    "Zhiyuan Chen": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
    "Yuecong Min": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
    "Jie Zhang": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
    "Bei Yan": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
    "Jiahao Wang": ["Trustworthy Technology and Engineering Laboratory, Huawei"],
    "Xiaozhen Wang": ["Trustworthy Technology and Engineering Laboratory, Huawei"],
    "Shiguang Shan": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"]
  }
}

Example 5 - Comma-separated multiple affiliations:
Input: "Dr. Sarah Johnson(1,3),Prof. Michael Chen(2,4),Lisa Zhang(1),Robert Kim(2,3,4) ((1) Stanford University, Computer Science Department (2) Google Research (3) MIT Media Lab (4) University of Toronto)"
Output: {
  "authors": ["Dr. Sarah Johnson", "Prof. Michael Chen", "Lisa Zhang", "Robert Kim"],
  "author_institutions": {
    "Dr. Sarah Johnson": ["Stanford University, Computer Science Department", "MIT Media Lab"],
    "Prof. Michael Chen": ["Google Research", "University of Toronto"],
    "Lisa Zhang": ["Stanford University, Computer Science Department"],
    "Robert Kim": ["Google Research", "MIT Media Lab", "University of Toronto"]
  }
}

Example 6 - Very complex multi-institutional:
Input: "Chiara Giangregorio(1),Cristina Maria Licciardello(1),Vanja Miskovic(1 and 2),Leonardo Provenzano(1 and 2),Alessandra Laura Giulia Pedrocchi(1),Andra Diana Dumitrascu(2),Arsela Prelaj(2),Marina Chiara Garassino(3),Emilia Ambrosini(1),Simona Ferrante(1 and 4) ((1) Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy, (2) Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, Milan, Italy, (3) Department of Medicine, Section of Hematology/Oncology, University of Chicago, Chicago, IL, USA, (4) IRCCS Istituto Neurologico Carlo Besta, Milan, Italy)"
Output: {
  "authors": ["Chiara Giangregorio", "Cristina Maria Licciardello", "Vanja Miskovic", "Leonardo Provenzano", "Alessandra Laura Giulia Pedrocchi", "Andra Diana Dumitrascu", "Arsela Prelaj", "Marina Chiara Garassino", "Emilia Ambrosini", "Simona Ferrante"],
  "author_institutions": {
    "Chiara Giangregorio": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy"],
    "Cristina Maria Licciardello": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy"],
    "Vanja Miskovic": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy", "Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, Milan, Italy"],
    "Leonardo Provenzano": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy", "Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, Milan, Italy"],
    "Alessandra Laura Giulia Pedrocchi": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy"],
    "Andra Diana Dumitrascu": ["Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, Milan, Italy"],
    "Arsela Prelaj": ["Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, Milan, Italy"],
    "Marina Chiara Garassino": ["Department of Medicine, Section of Hematology/Oncology, University of Chicago, Chicago, IL, USA"],
    "Emilia Ambrosini": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy"],
    "Simona Ferrante": ["Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy", "IRCCS Istituto Neurologico Carlo Besta, Milan, Italy"]
  }
}

Example 7 - Edge cases:
Input: ""
Output: {"authors": [], "author_institutions": {}}

Input: "Department of Computer Science, University of Example"
Output: {"authors": [], "author_institutions": {}}
"""

def get_author_parsing_user_prompt(author_string: str) -> str:
    """Generate user prompt for author parsing."""
    return f"""Parse the following author string and extract author names and their institutional affiliations:

Input: "{author_string}"

Provide the result in the specified JSON format. CRITICAL REQUIREMENTS:
1. Extract author names EXACTLY as they appear - preserve all characters, encoding, spelling
2. Do NOT correct, fix, normalize, or change any text - output exactly what you see
3. Identify actual person names vs institutional names correctly
4. Map numbered affiliations correctly (both "and" format and comma format)
5. Handle comma-separated affiliations like (1,3) and (2,3,4) correctly
6. Preserve complete institutional information
7. Ensure all authors from the input are included

Output:"""
