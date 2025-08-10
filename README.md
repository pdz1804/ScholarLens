# ScholarLens - Research Paper Analysis System

A sophisticated multi-agent RAG (Retrieval-Augmented Generation) system for analyzing relationships between authors, technologies, and research domains using academic papers as bridges.


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ScholarLens-blue?logo=github)](https://github.com/pdz1804/ScholarLens)

## Quick Start

<details>
<summary>Click to expand setup instructions</summary>

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/pdz1804/ScholarLens.git
cd ScholarLens

# Run development setup
python setup_dev.py
```

### 2. Configure System

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Required: OPENAI_API_KEY
# Optional: Dataset path, model settings, etc.
```

### 3. Add Dataset

Place your ArXiv Computer Science dataset files in the `data/` directory:
- `arxiv_cs_2021.csv` (2021 papers)
- `arxiv_cs_2022.csv` (2022 papers) 
- `arxiv_cs_2023.csv` (2023 papers)
- `arxiv_cs_2024.csv` (2024 papers)
- `arxiv_cs_2025.csv` (2025 papers)

The system supports both multi-file format (recommended) and legacy single-file format.

### 4. Test the System

```bash
# Basic usage example
python examples/basic_usage.py

# Command line interface
python main.py "who are the top AI researchers?"

# Run tests
python tests/test_system.py
```

</details>

## Supported Query Types

For comprehensive examples and test queries for all supported query scenarios, please refer to [tests/TEST_QUERIES.md](tests/TEST_QUERIES.md).

The system supports various query types including:
- Author Expertise Analysis
- Technology Trends Analysis  
- Domain Evolution Analysis
- Collaboration Networks
- Cross-domain Research
- Paper Impact Assessment
- Author Productivity Analysis

## Testing the System

To test the system with sample queries, check out the comprehensive test cases in [tests/TEST_QUERIES.md](tests/TEST_QUERIES.md) which contains detailed examples for each query type.

## Architecture

<details>
<summary>Click to see detailed architecture information</summary>

### Multi-Agent System

- **Query Classifier Agent**: Understands and categorizes user queries
- **Retrieval Agent**: Finds relevant papers using semantic and keyword search
- **Analysis Agent**: Performs statistical analysis on retrieved papers
- **Synthesis Agent**: Combines results into coherent insights
- **Validation Agent**: Ensures response quality and accuracy

### RAG Pipeline

1. **Query Processing**: Natural language understanding and intent classification
2. **Document Retrieval**: Semantic search using sentence transformers and FAISS
3. **Context Analysis**: Statistical analysis of authors, papers, and relationships
4. **Response Generation**: LLM-enhanced synthesis of findings
5. **Quality Validation**: Multi-dimensional validation of responses

### LLM-Based Extraction Architecture

- **ExtractionService**: Centralized LLM-based content extraction with Chain-of-Thought prompting
- **Prompt Separation**: Dedicated prompt files for extraction, analysis, and synthesis tasks
- **Fallback Mechanisms**: Pattern-based fallbacks for robust operation when LLM unavailable
- **Content Types**: Methodologies, key concepts, problems, solutions, author names, search terms
- **Quality Assurance**: Structured extraction with validation and error handling

### Key Features

- **Collaboration Network Analysis**: Analyze author collaboration patterns
- **Cross-domain Research**: Identify authors working across multiple domains
- **Temporal Analysis**: Track research trends over time
- **Impact Assessment**: Evaluate paper and author influence

</details>

## Dataset

<details>
<summary>View dataset details and supported formats</summary>

Currently supports ArXiv Computer Science papers in multiple yearly files:

- **2021-2025 Data**: `arxiv_cs_YYYY.csv` format
- **Columns**: Paper Title, Paper ID, Authors, Abstract, Domain, Primary Subject, Subjects, Date Submitted, Abstract URL, PDF URL

The system automatically combines data from multiple years and handles temporal analysis across the full dataset.

Future support planned for:

- `arxiv_math.csv` (Mathematics)
- `arxiv_stat.csv` (Statistics)
- Additional domain-specific datasets

Note that, for those data, we collect by just scraping the paper id, paper abstract, domains, subdomains, authors from the [arxiv](https://arxiv.org/) webpages.

</details>

## Installation

<details>
<summary>Step-by-step installation guide</summary>

```bash
# Clone the repository
git clone https://github.com/pdz1804/ScholarLens.git
cd ScholarLens

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize the system
python scripts/initialize_system.py
```

</details>

## Usage

<details>
<summary>Basic Usage</summary>

### Basic Usage

```python
from src.core.system import TechAuthorSystem

# Initialize system
system = TechAuthorSystem(config_path="config/config.yaml")

# Query the system
result = system.query("Your query here")
print(result)
```

### Command Line Usage

```bash
# Direct command line usage
python main.py "Your query here"
```

For comprehensive examples and sample queries, please refer to [tests/TEST_QUERIES.md](tests/TEST_QUERIES.md).

</details>

<details>
<summary>Advanced Usage</summary>

### Advanced Usage

```python
# Custom query with parameters
result = system.query(
    "Your query",
    params={
        "top_k": 15,
        "time_range": "2020-2024",
        "min_papers": 5
    }
)

# Batch queries
queries = ["Query 1", "Query 2", "Query 3"]
results = system.batch_query(queries)
```

For detailed query examples and test cases, see [tests/TEST_QUERIES.md](tests/TEST_QUERIES.md).

</details>

## Project Structure

<details>
<summary>View complete project structure</summary>

```
techauthor/
├── src/
│   ├── agents/           # Multi-agent implementations
│   ├── core/            # Core system components
│   ├── data/            # Data processing utilities
│   ├── retrieval/       # RAG implementation
│   └── utils/           # Utility functions
├── config/              # Configuration files
├── data/               # Dataset storage
├── scripts/            # Setup and utility scripts
├── tests/              # Test suite
├── docs/               # Documentation
└── examples/           # Usage examples
```

</details>

## Contributing

<details>
<summary>How to contribute to ScholarLens</summary>

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

</details>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ArXiv for providing open access to research papers
- The research community for advancing knowledge in AI and ML
- Contributors and maintainers of this project

## Contact

For questions or support, please open an issue on [GitHub](https://github.com/pdz1804/ScholarLens/issues) or contact the maintainer.
