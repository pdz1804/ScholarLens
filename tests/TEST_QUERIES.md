# TechAuthor Test Queries

This document contains test queries for each supported query type in the TechAuthor system.

## 🔧 System Architecture Updates

**LLM-Based Extraction**: The system now uses Large Language Model-based extraction with Chain-of-Thought prompting for improved accuracy in content analysis, replacing previous pattern-based methods.

**Query Type Distinction**:

- **Technology Trends**: Focus on methodological trends, technical approaches, and tool adoption
- **Domain Evolution**: Focus on conceptual shifts, paradigm changes, and fundamental research direction evolution

## BASIC QUERY TYPES - Priority Test Cases

### 1. Author Statistics Queries (AUTHOR_STATS)

**Description**: Retrieve comprehensive statistical profiles for individual authors, including publication metrics, research domain analysis, collaboration patterns, temporal activity, and impact indicators within the dataset

**Test Query 1**: "Tell me about John Smolin's research profile"

- **Expected behavior**: System should look up the specific author in the database and return their statistics including total papers, years active, research areas, and collaborators. If not found, should suggest similar author names.
- **Parameters**: `{"author": "John Smolin"}`

**Test Query 2**: "Show me stats for Nick Johnson"

- **Expected behavior**: Same as above but for Nick Johnson
- **Parameters**: `{"author": "Nick Johnson"}`

### 2. Paper Search Queries (PAPER_SEARCH)

**Description**: Perform targeted searches for academic papers using technology names, methodologies, concepts, or research topics, with semantic matching and relevance ranking to identify the most pertinent research publications

**Test Query 1**: "Show me papers about REPRO"

- **Expected behavior**: System should search for papers that contain "REPRO" in their title or abstract, ranked by relevance
- **Parameters**: `{"technology": "REPRO"}`

**Test Query 2**: "Find papers about transformers"

- **Expected behavior**: Search for papers related to transformer architectures
- **Parameters**: `{"technology": "transformers"}`

## Query Types and Test Examples

### 1. Author Expertise Queries

**Description**: Identify and rank leading researchers within specific academic domains or research subjects based on publication volume, research impact, domain expertise depth

**Test Query 1**: "Who are the top authors in Machine Learning?"

- **Expected behavior**: System should identify and rank authors based on publication count and influence in ML
- **Parameters**: `{"top_k": 10, "domain": "cs", "subject": "Machine Learning"}`

**Test Query 2**: "Who are the leading researchers in Image and Video Processing?"

- **Expected behavior**: Focus on computer vision and image processing experts
- **Parameters**: `{"top_k": 15, "subject": "Image and Video Processing"}`

### 2. Technology Trends Queries

**Description**: Analyze temporal patterns in technology adoption, emerging methodologies, technical approach popularity, and implementation trends within research domains

**Test Query 1**: "What are the emerging methodologies in Machine Learning from 2021 to 2024?"

- **Expected behavior**: Identify trending technical approaches, popular algorithms, and methodology adoption patterns
- **Parameters**: `{"domain": "cs", "subject": "Machine Learning", "time_range": "2021-2024"}`
- **Focus**: Technical trends, methodology popularity, implementation patterns

**Test Query 2**: "What new deep learning architectures are gaining popularity in Computer Vision?"

- **Expected behavior**: Show trending architectures, technical innovations, and adoption rates
- **Parameters**: `{"subject": "Computer Vision", "time_range": "2022-2024", "focus": "technical_methods"}`
- **Focus**: Architecture trends, technical innovations, methodology evolution

### 3. Author Collaboration Queries

**Description**: Map and analyze research collaboration networks, co-authorship patterns, and collaborative relationships between researchers within specific domains or across the entire academic network

**Test Query 1**: "Who collaborates with Lei Zhang?"

- **Expected behavior**: Find co-authors and collaboration networks (if Lei Zhang is in dataset)
- **Parameters**: `{"author_name": "Lei Zhang", "top_k": 20}`

**Test Query 2**: "What are the collaboration patterns in Quantum Computing research?"

- **Expected behavior**: Identify collaboration networks in quantum computing domain
- **Parameters**: `{"domain": "quantum", "collaboration_depth": 2}`

### 4. Domain Evolution Queries

**Description**: Track long-term conceptual transformations, paradigm shifts, fundamental changes in research problem formulations, theoretical framework evolution, and shifts in core research questions within academic domains over extended time periods

**Test Query 1**: "How has the conceptual approach to Computer Vision research evolved from 2021 to 2024?"

- **Expected behavior**: Show paradigm shifts, fundamental changes in problem formulation, and conceptual transitions
- **Parameters**: `{"domain": "Computer Vision", "time_range": "2021-2024", "analysis_type": "conceptual_evolution"}`
- **Focus**: Paradigm shifts, conceptual changes, problem formulation evolution

**Test Query 2**: "What are the fundamental conceptual changes in Natural Language Processing over time?"

- **Expected behavior**: Track evolution in core concepts, theoretical approaches, and paradigm transitions
- **Parameters**: `{"subject": "Natural Language Processing", "metrics": ["conceptual_shifts", "paradigm_evolution"]}`
- **Focus**: Theoretical evolution, conceptual transitions, paradigm changes

**Key Distinction from Technology Trends**: Domain Evolution focuses on *why* and *what* researchers study (conceptual shifts), while Technology Trends focuses on *how* they study it (methodological trends).

### 5. Cross-Domain Analysis Queries (NOT DONE)

**Description**: Identify interdisciplinary researchers who contribute to multiple academic domains, analyze knowledge transfer patterns between fields, and discover researchers bridging different research areas through their diverse publication portfolios

**Test Query 1**: "Which authors work across Machine Learning and Signal Processing?" 

- **Expected behavior**: Identify interdisciplinary researchers
- **Parameters**: `{"domains": ["Machine Learning", "Signal Processing"], "min_papers_per_domain": 2}`

**Test Query 2**: "Who are the researchers bridging Image Processing and Biomolecules?"

- **Expected behavior**: Find interdisciplinary work between these fields
- **Parameters**: `{"domains": ["Image and Video Processing", "Biomolecules"]}`

### 6. Paper Impact Queries (NOT DONE)

**Description**: Assess and rank academic papers based on influence indicators, novelty measures, citation potential, research significance, and contribution to field advancement within specific domains and time periods

**Test Query 1**: "What are the most influential papers in Machine Learning published in 2022?"

- **Expected behavior**: Rank papers by various impact metrics (citations, novelty indicators)
- **Parameters**: `{"subject": "Machine Learning", "year": 2022, "metric": "influence"}`

**Test Query 2**: "Which papers in Computer Vision have had the biggest impact recently?"

- **Expected behavior**: Identify high-impact recent CV papers
- **Parameters**: `{"domain": "Computer Vision", "time_range": "2023-2024"}`

### 7. Author Productivity Queries

**Description**: Evaluate researcher productivity through publication frequency analysis, output consistency patterns, research velocity trends, and publication rate changes over time across different academic domains and career stages

**Test Query 1**: "Who are the most prolific authors in computer science from 2021-2024?"

- **Expected behavior**: Rank authors by publication count and consistency
- **Parameters**: `{"time_range": "2021-2024", "domain": "cs", "top_k": 20}`

**Test Query 2**: "Which authors have shown increasing productivity in Machine Learning?"

- **Expected behavior**: Identify authors with growing publication rates
- **Parameters**: `{"subject": "Machine Learning", "metric": "productivity_growth"}`

### 8. Unclassified Query Handling

**Description**: Test system robustness and error handling capabilities with non-research queries, ambiguous requests, out-of-scope questions, and queries that don't match any defined query types, ensuring appropriate classification and user guidance

**Test Query 1**: "What should I have for lunch?"

- **Expected behavior**: System should classify as UNCLASSIFIED and ask user to rephrase with suggestions
- **Parameters**: `{}`

**Test Query 2**: "Random nonsense query xyz"

- **Expected behavior**: Same as above - ask for clarification with helpful suggestions
- **Parameters**: `{}`

## Special Test Cases

### Edge Cases

1. **Empty Result Query**: "Who are the experts in Underwater Basket Weaving?"

   - Should handle gracefully with no results
2. **Ambiguous Query**: "Tell me about AI"

   - Should ask for clarification or provide broad overview
3. **Date Range Query**: "What happened in Machine Learning in 2019?"

   - Should handle dates outside dataset range gracefully

### Performance Tests

1. **Large Dataset Query**: "Show me all authors in computer science"

   - Test system performance with large result sets
2. **Complex Multi-filter Query**: "Find prolific authors in Machine Learning who also work in Computer Vision, published between 2022-2024, with at least 5 papers"

   - Test complex parameter handling

## Running the Tests

To run these queries, use the main system:

```bash
# Single query
python main.py "Who are the top authors in Machine Learning?"

# Interactive mode
python main.py --interactive

# Batch testing
python main.py --batch-file test_queries.txt --output results.json
```

## Expected Dataset Coverage

Based on the arxiv_cs_YYYY.csv files (2021-2025), the system should handle:

- **Domains**: Computer Science papers across multiple subfields
- **Time Range**: 2021-2025 research papers
- **Authors**: Thousands of unique researchers
- **Subjects**: Machine Learning, Computer Vision, NLP, Signal Processing, etc.
- **Paper Count**: Varies by year, expect thousands of papers per year

## Validation Checklist

For each query type, verify:

- [ ] Query classification works correctly
- [ ] Retrieval returns relevant papers
- [ ] Analysis produces meaningful insights
- [ ] Results are properly ranked/scored
- [ ] Response format is consistent
- [ ] Performance is acceptable (< 2 seconds for most queries)
- [ ] Error handling works for edge cases
