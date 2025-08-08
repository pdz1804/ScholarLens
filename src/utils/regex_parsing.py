"""
Regex-based author and institution parsing utility.
Handles 7 specific parsing cases to replace expensive LLM parsing.
"""
import re
from typing import Dict, List, Tuple

# class RegexParsingUtility:
#     """Utility class for parsing author names and institutions using regex patterns."""

#     def parse_authors_and_institutions(self, author_string: str) -> Tuple[List[str], Dict[str, List[str]]]:
#         """
#         Parse author string based on 7 specific cases:
        
#         Case 1: Name, Name, Name (no institutions)
#         Case 2: Name(Insti), Name(Insti) 
#         Case 3: Name(1,2,3), Name(1,2) ((1) Inst, (2) Inst, (3) Inst)
#         Case 4: Name(1 and 2), Name(1, 2) ((1) Inst, (2) Inst)
#         Case 5: Empty String
#         Case 6: Insti Name (treat as Case 1)
#         Case 7: Mixed - Name(Insti), Name(1,2), Name(Insti), Name(1,2) ((1) Inst, (2) Inst)
        
#         Returns:
#             Tuple of (authors_list, author_institutions_dict)
#         """
#         # Case 5: Empty String
#         if not author_string or not author_string.strip():
#             return [], {}
            
#         author_string = author_string.strip()
        
#         # Case 1 & 6: Simple comma-separated names without parentheses
#         if '(' not in author_string:
#             return self._parse_case_1_simple_names(author_string)
        
#         # Check for numbered institution definitions at the end: ((1) Inst, (2) Inst)
#         # Look for the pattern starting with " ((" and ending with "))"
#         institution_match = re.search(r'\s\(\((.+)\)\)$', author_string)
        
#         if institution_match:
#             # Cases 3, 4, or 7: Has numbered institution definitions
#             authors_part = author_string[:institution_match.start()].strip()
#             institutions_part = institution_match.group(1)
            
#             # Parse institution definitions
#             institution_map = self._parse_institution_definitions(institutions_part)
            
#             # Check if it's mixed case (Case 7)
#             has_direct = re.search(r'[^,()]+\([^)]*[a-zA-Z][^)]*\)', authors_part)  # Name(contains letters)
#             has_numbered = re.search(r'[^,()]+\([^)]*\d[^)]*\)', authors_part)      # Name(contains digits)
            
#             if has_direct and has_numbered:
#                 # Case 7: Mixed
#                 return self._parse_case_7_mixed(authors_part, institution_map)
#             elif has_numbered:
#                 # Cases 3 & 4: Numbered affiliations only
#                 return self._parse_case_3_4_numbered(authors_part, institution_map)
#             else:
#                 # Fallback to simple names
#                 return self._parse_case_1_simple_names(authors_part)
#         else:
#             # Case 2: Direct affiliations only - Name(Institution), Name(Institution)
#             return self._parse_case_2_direct_affiliations(author_string)
    
#     def _parse_case_1_simple_names(self, author_string: str) -> Tuple[List[str], Dict[str, List[str]]]:
#         """Case 1 & 6: Parse simple comma-separated names without institutions."""
#         authors = []
        
#         # Split by comma and clean each author name
#         raw_authors = author_string.split(',')
        
#         for author in raw_authors:
#             author = author.strip()
#             if author:  # Skip empty strings
#                 author = self._clean_author_name(author)
#                 if author:  # Final check after cleaning
#                     authors.append(author)
        
#         return authors, {}
    
#     def _parse_case_2_direct_affiliations(self, author_string: str) -> Tuple[List[str], Dict[str, List[str]]]:
#         """Case 2: Parse Name(Institution), Name(Institution) format."""
#         authors = []
#         author_institutions = {}
        
#         # Split by commas to get individual author-institution pairs
#         parts = []
#         current_part = ""
#         paren_count = 0
        
#         for char in author_string:
#             if char == '(':
#                 paren_count += 1
#             elif char == ')':
#                 paren_count -= 1
#             elif char == ',' and paren_count == 0:
#                 if current_part.strip():
#                     parts.append(current_part.strip())
#                 current_part = ""
#                 continue
            
#             current_part += char
        
#         # Don't forget the last part
#         if current_part.strip():
#             parts.append(current_part.strip())
        
#         # Parse each part: Name(Institution)
#         for part in parts:
#             part = part.strip()
#             if not part:
#                 continue
                
#             # Match pattern: Name(Institution)
#             match = re.match(r'^([^()]+?)\s*\(([^)]+)\)$', part)
#             if match:
#                 author_name = self._clean_author_name(match.group(1))
#                 institution = match.group(2).strip()
                
#                 if author_name:
#                     authors.append(author_name)
#                     author_institutions[author_name] = [institution]
#             else:
#                 # Fallback: treat as simple name
#                 author_name = self._clean_author_name(part)
#                 if author_name:
#                     authors.append(author_name)
#                     author_institutions[author_name] = []
        
#         return authors, author_institutions
    
#     def _parse_case_3_4_numbered(self, authors_part: str, institution_map: Dict[str, str]) -> Tuple[List[str], Dict[str, List[str]]]:
#         """Cases 3 & 4: Parse Name(1,2,3), Name(1 and 2) with numbered institution definitions."""
#         authors = []
#         author_institutions = {}
        
#         # Split by commas to get individual author-number pairs
#         parts = []
#         current_part = ""
#         paren_count = 0
        
#         for char in authors_part:
#             if char == '(':
#                 paren_count += 1
#             elif char == ')':
#                 paren_count -= 1
#             elif char == ',' and paren_count == 0:
#                 if current_part.strip():
#                     parts.append(current_part.strip())
#                 current_part = ""
#                 continue
            
#             current_part += char
        
#         # Don't forget the last part
#         if current_part.strip():
#             parts.append(current_part.strip())
        
#         # Parse each part: Name(numbers)
#         for part in parts:
#             part = part.strip()
#             if not part:
#                 continue
                
#             # Match pattern: Name(1,2,3) or Name(1 and 2)
#             match = re.match(r'^([^()]+?)\s*\(([^)]+)\)$', part)
#             if match:
#                 author_name = self._clean_author_name(match.group(1))
#                 numbers_str = match.group(2).strip()
                
#                 if author_name:
#                     authors.append(author_name)
#                     author_institutions[author_name] = []
                    
#                     # Extract numbers - handle both "1,2,3" and "1 and 2" formats
#                     # Replace "and" with comma for consistent parsing
#                     numbers_str = re.sub(r'\s+and\s+', ',', numbers_str, flags=re.IGNORECASE)
#                     numbers = re.findall(r'\d+', numbers_str)
                    
#                     # Map numbers to institutions
#                     for num in numbers:
#                         if num in institution_map:
#                             institution = institution_map[num]
#                             if institution not in author_institutions[author_name]:
#                                 author_institutions[author_name].append(institution)
#             else:
#                 # Fallback: treat as simple name
#                 author_name = self._clean_author_name(part)
#                 if author_name:
#                     authors.append(author_name)
#                     author_institutions[author_name] = []
        
#         return authors, author_institutions
    
#     def _parse_case_7_mixed(self, authors_part: str, institution_map: Dict[str, str]) -> Tuple[List[str], Dict[str, List[str]]]:
#         """Case 7: Mixed format - Name(Institution), Name(1,2), Name(Institution), Name(1,2)."""
#         authors = []
#         author_institutions = {}
        
#         # Split by commas to get individual parts
#         parts = []
#         current_part = ""
#         paren_count = 0
        
#         for char in authors_part:
#             if char == '(':
#                 paren_count += 1
#             elif char == ')':
#                 paren_count -= 1
#             elif char == ',' and paren_count == 0:
#                 if current_part.strip():
#                     parts.append(current_part.strip())
#                 current_part = ""
#                 continue
            
#             current_part += char
        
#         # Don't forget the last part
#         if current_part.strip():
#             parts.append(current_part.strip())
        
#         # Parse each part
#         for part in parts:
#             part = part.strip()
#             if not part:
#                 continue
                
#             # Match pattern: Name(content)
#             match = re.match(r'^([^()]+?)\s*\(([^)]+)\)$', part)
#             if match:
#                 author_name = self._clean_author_name(match.group(1))
#                 content = match.group(2).strip()
                
#                 if author_name:
#                     authors.append(author_name)
#                     author_institutions[author_name] = []
                    
#                     # Check if content contains digits (numbered affiliations)
#                     if re.search(r'\d', content):
#                         # Numbered affiliations - handle "1,2,3" and "1 and 2" formats
#                         content = re.sub(r'\s+and\s+', ',', content, flags=re.IGNORECASE)
#                         numbers = re.findall(r'\d+', content)
                        
#                         for num in numbers:
#                             if num in institution_map:
#                                 institution = institution_map[num]
#                                 if institution not in author_institutions[author_name]:
#                                     author_institutions[author_name].append(institution)
#                     else:
#                         # Direct affiliation
#                         author_institutions[author_name].append(content)
#             else:
#                 # Fallback: treat as simple name
#                 author_name = self._clean_author_name(part)
#                 if author_name:
#                     authors.append(author_name)
#                     author_institutions[author_name] = []
        
#         return authors, author_institutions
    
#     def _parse_institution_definitions(self, institutions_str: str) -> Dict[str, str]:
#         """Parse numbered institution definitions: (1) Institution, (2) Institution"""
#         institution_map = {}
        
#         # Pattern: (1) Institution Name, (2) Another Institution
#         matches = re.findall(r'\((\d+)\)\s*([^,()]+?)(?=\s*\(\d+\)|$)', institutions_str)
        
#         for number, institution in matches:
#             institution_map[number] = institution.strip()
        
#         return institution_map
    
#     def _clean_author_name(self, name: str) -> str:
#         """Clean and normalize author name."""
#         if not name:
#             return ""
            
#         name = name.strip()
        
#         # Remove common academic titles
#         title_prefixes = r'^(Dr\.?|Prof\.?|Professor|Mr\.?|Ms\.?|Mrs\.?|Miss)\s+'
#         name = re.sub(title_prefixes, '', name, flags=re.IGNORECASE)
        
#         # Remove trailing academic titles
#         title_suffixes = r'\s+(Ph\.?D\.?|M\.?D\.?|Ph\.D|M\.D|PhD|MD)$'
#         name = re.sub(title_suffixes, '', name, flags=re.IGNORECASE)
        
#         # Clean up extra whitespace
#         name = re.sub(r'\s+', ' ', name).strip()
        
#         return name



import re
from typing import Dict, List, Tuple


class RegexParsingUtility:
    """
    Utility class for parsing authors and institution mappings from complex
    author+affiliation strings (handles direct and numbered institution assignments).
    """

    @staticmethod
    def parse(input_str: str) -> Tuple[List[str], Dict[str, List[str]]]:
        input_str = input_str.strip()
        if not input_str:
            return [], {}

        # If there are no parentheses, it's not author+affil: return empty (for test 9)
        if '(' not in input_str and ')' not in input_str:
            # If there are commas, treat as a list of authors
            author_tokens = [name.strip() for name in input_str.split(',') if name.strip()]
            if len(author_tokens) >= 1:
                return author_tokens, {}
            else:
                return [], {}

        # Try to extract institution mapping block: ((1) ... (2) ... (3) ...)
        inst_block, authors_part = RegexParsingUtility._extract_institution_block(input_str)
        inst_map = RegexParsingUtility._parse_institution_map(inst_block) if inst_block else {}

        # Split author part
        author_tokens = RegexParsingUtility._split_authors(authors_part)

        author_to_insts = {}
        author_names = []
        for author_token in author_tokens:
            name, affils = RegexParsingUtility._parse_author_and_affils(author_token)
            # If token is just empty or garbage, skip
            if not name or not name.strip():
                continue
            
            # Always add name to author_names (unless empty), even if no parentheses!
            author_names.append(name.strip())
            inst_names = []
            if '(' not in author_token or ')' not in author_token:
                # No affil -- but for numbered institution, still need to add (empty) if mapping exists
                if inst_map:
                    author_to_insts[name.strip()] = inst_names  # empty list
                # For direct affil, skip adding to dict if no affil
                continue
            if inst_map and all(a.isdigit() for a in affils):  # Numbered institution mode
                for affil in affils:
                    if affil in inst_map:
                        inst_names.append(inst_map[affil])
            elif affils:  # Direct institution mode (parentheses after name)
                inst_names = [' '.join(affils)] if len(affils) > 1 else affils
            if inst_names:
                author_to_insts[name.strip()] = inst_names
            elif inst_map:  # For numbered institution, must always include mapping even if empty
                author_to_insts[name.strip()] = inst_names
        if not inst_map:
            author_to_insts = {k: v for k, v in author_to_insts.items() if v}
        return author_names, author_to_insts

    @staticmethod
    def _extract_institution_block(s: str) -> Tuple[str, str]:
        s = s.strip()
        match = re.search(r"\(\(\d+\)", s)
        if match:
            start_idx = match.start()
            parens = 0
            end_idx = None
            for i in range(start_idx, len(s)):
                if s[i] == '(':
                    parens += 1
                elif s[i] == ')':
                    parens -= 1
                    if parens == 0:
                        end_idx = i
                        break
            if end_idx is not None:
                inst_block = s[start_idx:end_idx+1]
                authors_block = s[:start_idx].rstrip(", ")
                return inst_block, authors_block
        # Fallback: last '((...' or '(1)...'
        last_double = s.rfind('((')
        if last_double != -1:
            authors_block = s[:last_double].rstrip(", ")
            inst_block = s[last_double:]
            return inst_block, authors_block
        paren_block_match = re.search(r'(\(\d+\).*)$', s)
        if paren_block_match:
            inst_block = paren_block_match.group(1)
            authors_block = s[:paren_block_match.start()].rstrip(", ")
            return inst_block, authors_block
        return "", s

    @staticmethod
    def _parse_institution_map(inst_block: str) -> Dict[str, str]:
        inst_block = inst_block.strip()
        if inst_block.startswith('(('):
            inst_block = inst_block[1:]
        if inst_block.startswith('(') and inst_block.endswith(')'):
            inst_block = inst_block[1:-1].strip()
        # Use a regex split: every (number) marks a new entry
        pieces = re.split(r'(?=\(\d+\))', inst_block)
        inst_map = {}
        for piece in pieces:
            m = re.match(r'\(?(\d+)\)?\s*(.+)', piece.strip())
            if m:
                idx, inst = m.groups()
                inst_map[idx.strip()] = inst.strip(' ,')
        return inst_map

    @staticmethod
    def _split_authors(authors_block: str) -> List[str]:
        author_tokens = []
        cur = []
        paren_level = 0
        for c in authors_block:
            if c == ',' and paren_level == 0:
                token = ''.join(cur).strip()
                if token:
                    author_tokens.append(token)
                cur = []
            else:
                if c == '(':
                    paren_level += 1
                elif c == ')':
                    paren_level = max(0, paren_level - 1)
                cur.append(c)
        if cur:
            token = ''.join(cur).strip()
            if token:
                author_tokens.append(token)
        return author_tokens

    @staticmethod
    def _parse_author_and_affils(author_token: str) -> Tuple[str, List[str]]:
        author_token = author_token.strip()
        m = re.match(r'^(.*?)\s*\(([^()]+)\)$', author_token)
        if not m:
            return author_token, []
        name, affil_raw = m.groups()
        # If all affil_raw parts are digits/and/comma, treat as numbered
        is_numbered = all(bool(re.fullmatch(r'\d+', p.strip())) or p.strip().lower() == 'and' or p.strip() == ''
                          for p in re.split(r'[ ,]+', affil_raw.replace('and', '')))
        if is_numbered:
            affil_raw = affil_raw.replace(' and ', ',')
            affil_raw = affil_raw.replace(';', ',')
            affils = [p.strip() for p in affil_raw.split(',') if p.strip().isdigit()]
            return name.strip(), affils
        else:
            # Treat as single direct affiliation string (even if contains commas)
            return name.strip(), [affil_raw.strip()]







# Global instance for easy access
_regex_parser = None

def get_regex_parser() -> RegexParsingUtility:
    """Get global regex parser instance."""
    global _regex_parser
    if _regex_parser is None:
        _regex_parser = RegexParsingUtility()
    return _regex_parser
