#!/usr/bin/env python3
"""
Comprehensive test script for robust author parsing functionality.
Tests various complex author string formats to ensure robust parsing of both authors and institutions.
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import Paper
from datetime import datetime


def test_author_and_institution_parsing():
    """Test comprehensive author and institution parsing scenarios."""
    
    # Test cases with expected results for both authors and institutions
    test_cases = [
        {
            "name": "Simple comma-separated authors",
            "input": "Zhenyang Li,Xiaoyang Bai,Tongchen Zhang,Pengfei Shen,Weiwei Xu,Yifan Peng",
            "expected_authors": ["Zhenyang Li", "Xiaoyang Bai", "Tongchen Zhang", "Pengfei Shen", "Weiwei Xu", "Yifan Peng"],
            "expected_institutions": {}
        },
        {
            "name": "Authors with simple affiliations",
            "input": "AnaÃ¯s Ollagnier(CRISAM, CNRS, MARIANNE),Aline Menin(WIMMICS, Laboratoire I3S - SPARKS)",
            "expected_authors": ["AnaÃ¯s Ollagnier", "Aline Menin"],
            "expected_institutions": {
                "AnaÃ¯s Ollagnier": ["CRISAM, CNRS, MARIANNE"],
                "Aline Menin": ["WIMMICS, Laboratoire I3S - SPARKS"]
            }
        },
        {
            "name": "Authors with numbered affiliations and long institutional info",
            "input": "Mufakir Qamar Ansari(1),Mudabir Qamar Ansari(2) ((1) Department of Electrical Engineering and Computer Science, The University of Toledo, Toledo, OH, USA, (2) Department of School of Accounting and Information Systems, Lamar University, Beaumont, TX, USA)",
            "expected_authors": ["Mufakir Qamar Ansari", "Mudabir Qamar Ansari"],
            "expected_institutions": {
                "Mufakir Qamar Ansari": ["Department of Electrical Engineering and Computer Science, The University of Toledo, Toledo, OH, USA"],
                "Mudabir Qamar Ansari": ["Department of School of Accounting and Information Systems, Lamar University, Beaumont, TX, USA"]
            }
        },
        {
            "name": "Author with university in parentheses",
            "input": "Padmavathi Moorthy(SUNY Buffalo)",
            "expected_authors": ["Padmavathi Moorthy"],
            "expected_institutions": {
                "Padmavathi Moorthy": ["SUNY Buffalo"]
            }
        },
        {
            "name": "Author with company affiliation",
            "input": "Louis Sugy(NVIDIA)",
            "expected_authors": ["Louis Sugy"],
            "expected_institutions": {
                "Louis Sugy": ["NVIDIA"]
            }
        },
        {
            "name": "Complex multi-author with detailed institutional info",
            "input": "Chiara Giangregorio(1),Cristina Maria Licciardello(1),Vanja Miskovic(1 and 2),Leonardo Provenzano(1 and 2),Alessandra Laura Giulia Pedrocchi(1),Andra Diana Dumitrascu(2),Arsela Prelaj(2),Marina Chiara Garassino(3),Emilia Ambrosini(1),Simona Ferrante(1 and 4) ((1) Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy, (2) Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, Milan, Italy, (3) Department of Medicine, Section of Hematology/Oncology, University of Chicago, Chicago, IL, USA, (4) IRCCS Istituto Neurologico Carlo Besta, Milan, Italy)",
            "expected_authors": ["Chiara Giangregorio", "Cristina Maria Licciardello", "Vanja Miskovic", "Leonardo Provenzano", "Alessandra Laura Giulia Pedrocchi", "Andra Diana Dumitrascu", "Arsela Prelaj", "Marina Chiara Garassino", "Emilia Ambrosini", "Simona Ferrante"],
            "expected_institutions": {
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
        },
        {
            "name": "Authors with academy and university affiliations",
            "input": "Zhiyuan Chen(1 and 2),Yuecong Min(1 and 2),Jie Zhang(1 and 2),Bei Yan(1 and 2),Jiahao Wang(3),Xiaozhen Wang(3),Shiguang Shan(1 and 2) ((1) State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS) (2) University of Chinese Academy of Sciences (3) Trustworthy Technology and Engineering Laboratory, Huawei)",
            "expected_authors": ["Zhiyuan Chen", "Yuecong Min", "Jie Zhang", "Bei Yan", "Jiahao Wang", "Xiaozhen Wang", "Shiguang Shan"],
            "expected_institutions": {
                "Zhiyuan Chen": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
                "Yuecong Min": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
                "Jie Zhang": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
                "Bei Yan": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"],
                "Jiahao Wang": ["Trustworthy Technology and Engineering Laboratory, Huawei"],
                "Xiaozhen Wang": ["Trustworthy Technology and Engineering Laboratory, Huawei"],
                "Shiguang Shan": ["State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS)", "University of Chinese Academy of Sciences"]
            }
        },
        {
            "name": "Edge case: empty string",
            "input": "",
            "expected_authors": [],
            "expected_institutions": {}
        },
        {
            "name": "Edge case: only institutional info",
            "input": "Department of Computer Science, University of Example",
            "expected_authors": ['Department of Computer Science', 'University of Example'],
            "expected_institutions": {}
        },
        {
            "name": "Sophisticated case: Mixed direct and numbered affiliations",
            "input": "Elena Rodriguez(MIT),Zhang Wei(1),Aisha Patel(Harvard Medical School),Mohammad Al-Hassan(2) ((1) Beijing Institute of Technology, (2) King Abdullah University of Science and Technology)",
            "expected_authors": ["Elena Rodriguez", "Zhang Wei", "Aisha Patel", "Mohammad Al-Hassan"],
            "expected_institutions": {
                "Elena Rodriguez": ["MIT"],
                "Zhang Wei": ["Beijing Institute of Technology"],
                "Aisha Patel": ["Harvard Medical School"],
                "Mohammad Al-Hassan": ["King Abdullah University of Science and Technology"]
            }
        },
        {
            "name": "Sophisticated case: Multiple affiliations per author with complex numbering",
            "input": "Dr. Sarah Johnson(1,3),Prof. Michael Chen(2,4),Lisa Zhang(1),Robert Kim(2,3,4) ((1) Stanford University, Computer Science Department, (2) Google Research, (3) MIT Media Lab, (4) University of Toronto)",
            "expected_authors": ["Dr. Sarah Johnson", "Prof. Michael Chen", "Lisa Zhang", "Robert Kim"],
            "expected_institutions": {
                "Dr. Sarah Johnson": ["Stanford University, Computer Science Department", "MIT Media Lab"],
                "Prof. Michael Chen": ["Google Research", "University of Toronto"],
                "Lisa Zhang": ["Stanford University, Computer Science Department"],
                "Robert Kim": ["Google Research", "MIT Media Lab", "University of Toronto"]
            }
        },
        {
            "name": "Sophisticated case: International names with diacritics",
            "input": "José María González(Universidad Politécnica de Madrid),François Müller(ETH Zurich),李明(Tsinghua University),Иван Петров(Moscow State University)",
            "expected_authors": ["José María González", "François Müller", "李明", "Иван Петров"],
            "expected_institutions": {
                "José María González": ["Universidad Politécnica de Madrid"],
                "François Müller": ["ETH Zurich"],
                "李明": ["Tsinghua University"],
                "Иван Петров": ["Moscow State University"]
            }
        },
        {
            "name": "Sophisticated case: Very long institutional names with nested organizations",
            "input": "Amanda Thompson(1),Carlos Rodriguez(2) ((1) Department of Biomedical Engineering and Institute for Computational Medicine, Johns Hopkins University School of Medicine, Baltimore, MD, USA, (2) Laboratory for Artificial Intelligence in Medicine, Department of Computer Science and Engineering, University of California San Diego, La Jolla, CA, USA)",
            "expected_authors": ["Amanda Thompson", "Carlos Rodriguez"],
            "expected_institutions": {
                "Amanda Thompson": ["Department of Biomedical Engineering and Institute for Computational Medicine, Johns Hopkins University School of Medicine, Baltimore, MD, USA"],
                "Carlos Rodriguez": ["Laboratory for Artificial Intelligence in Medicine, Department of Computer Science and Engineering, University of California San Diego, La Jolla, CA, USA"]
            }
        },
        {
            "name": "Multi-line and extra spaces",
            "input": "Marie Curie   (Sorbonne University),\n\tIsaac Newton  (Trinity College  ) , \nAlbert Einstein(1),Niels Bohr(2) ((1) Institute for Advanced Study, Princeton, (2) University of Copenhagen)",
            "expected_authors": ["Marie Curie", "Isaac Newton", "Albert Einstein", "Niels Bohr"],
            "expected_institutions": {
                "Marie Curie": ["Sorbonne University"],
                "Isaac Newton": ["Trinity College"],
                "Albert Einstein": ["Institute for Advanced Study, Princeton"],
                "Niels Bohr": ["University of Copenhagen"]
            }
        },
        {
            "name": "Numbered affiliations with and/comma mix",
            "input": "Ada Lovelace(1 and 2),Alan Turing(1, 3),Kurt Gödel(2) ((1) University of London, (2) Princeton University, (3) University of Manchester)",
            "expected_authors": ["Ada Lovelace", "Alan Turing", "Kurt Gödel"],
            "expected_institutions": {
                "Ada Lovelace": ["University of London", "Princeton University"],
                "Alan Turing": ["University of London", "University of Manchester"],
                "Kurt Gödel": ["Princeton University"]
            }
        },
        {
            "name": "Mixed direct, numbered, and missing parentheses",
            "input": "Grace Hopper(Yale),Claude Shannon(2),John von Neumann ((2) MIT)",
            "expected_authors": ["Grace Hopper", "Claude Shannon", "John von Neumann"],
            "expected_institutions": {
                "Grace Hopper": ["Yale"],
                "Claude Shannon": ["MIT"],
                "John von Neumann": []
            }
        },
        {
            "name": "Multiple names in one parenthesis",
            "input": "Paul Dirac, Wolfgang Pauli (ETH Zurich)",
            "expected_authors": ["Paul Dirac", "Wolfgang Pauli"],
            "expected_institutions": {
                "Wolfgang Pauli": ["ETH Zurich"]
            }
        },
        # (Should not assign ETH Zurich to Paul Dirac, only Wolfgang Pauli.)
        {
            "name": "Affiliation block out-of-order numbers",
            "input": "Erwin Schrödinger(2), Werner Heisenberg(1) ((2) University of Vienna, (1) University of Leipzig)",
            "expected_authors": ["Erwin Schrödinger", "Werner Heisenberg"],
            "expected_institutions": {
                "Erwin Schrödinger": ["University of Vienna"],
                "Werner Heisenberg": ["University of Leipzig"]
            }
        },
        {
            "name": "Very long, deeply nested institution string",
            "input": "Sophie Germain(1), Évariste Galois(2), Niels Henrik Abel(3) ((1) Laboratoire de Mathématiques et Applications, Université de Poitiers, Poitiers, France, (2) Département de Mathématiques et Applications, École Normale Supérieure, Paris, France, (3) Matematisk institutt, Universitetet i Oslo, Oslo, Norway)",
            "expected_authors": ["Sophie Germain", "Évariste Galois", "Niels Henrik Abel"],
            "expected_institutions": {
                "Sophie Germain": ["Laboratoire de Mathématiques et Applications, Université de Poitiers, Poitiers, France"],
                "Évariste Galois": ["Département de Mathématiques et Applications, École Normale Supérieure, Paris, France"],
                "Niels Henrik Abel": ["Matematisk institutt, Universitetet i Oslo, Oslo, Norway"]
            }
        }
    ]
    
    print("="*80)
    print("AUTHOR AND INSTITUTION PARSING TEST RESULTS (LLM-BASED)")
    print("="*80)
    
    failed_cases = []
    total_cases = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: {test_case['name']}")
        print(f"    Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
        
        try:
            # Create a dummy paper with the author string and all required fields
            paper = Paper(
                paper_id="test-paper-id",
                title="Test Paper",
                authors=test_case["input"],  # This will trigger the parser
                abstract="Test abstract",
                domain="Computer Science",
                primary_subject="Artificial Intelligence", 
                subjects=["cs.AI", "cs.LG"],
                date_submitted=datetime.now(),
                abstract_url="https://arxiv.org/abs/test",
                pdf_url="https://arxiv.org/pdf/test.pdf"
            )
            
            # Extract the parsed authors and institutions
            parsed_authors = paper.authors
            parsed_institutions = paper.author_institutions
            
            # Check authors
            authors_match = parsed_authors == test_case["expected_authors"]
            if not authors_match:
                print(f"    ❌ AUTHORS MISMATCH")
                print(f"       Expected: {test_case['expected_authors']}")
                print(f"       Got:      {parsed_authors}")
            else:
                print(f"    ✅ Authors parsed correctly: {len(parsed_authors)} authors")
            
            # Check institutions
            institutions_match = parsed_institutions == test_case["expected_institutions"]
            if not institutions_match:
                print(f"    ❌ INSTITUTIONS MISMATCH")
                print(f"       Expected: {test_case['expected_institutions']}")
                print(f"       Got:      {parsed_institutions}")
            else:
                print(f"    ✅ Institutions parsed correctly: {len(parsed_institutions)} mappings")
            
            if not authors_match or not institutions_match:
                failed_cases.append((i, test_case["name"], "Authors" if not authors_match else "Institutions"))
            
        except Exception as e:
            print(f"    ❌ ERROR: {str(e)}")
            failed_cases.append((i, test_case["name"], f"Exception: {str(e)}"))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not failed_cases:
        print(f"✅ ALL {total_cases} TEST CASES PASSED!")
        return True
    else:
        print(f"❌ {len(failed_cases)} out of {total_cases} test cases FAILED:")
        for case_num, case_name, issue in failed_cases:
            print(f"   {case_num}. {case_name} - {issue}")
        return False


if __name__ == "__main__":
    success = test_author_and_institution_parsing()
    sys.exit(0 if success else 1)
