"""
Extraction prompts for LLM-based content analysis.
Contains Chain-of-Thought prompts with examples for accurate extraction.
"""

# Methodology Extraction Prompt with Chain-of-Thought
METHODOLOGY_EXTRACTION_PROMPT = """You are an expert research methodology analyst. Your task is to extract research methodologies and approaches from academic papers.

## CHAIN-OF-THOUGHT ANALYSIS:

1. **Read and Understand**: Analyze each paper's title and abstract
2. **Identify Methods**: Look for specific methodologies, algorithms, frameworks, and approaches
3. **Categorize**: Group similar methodologies (e.g., "neural networks", "deep learning", "CNNs")  
4. **Prioritize**: Focus on core methodologies, not just tools or datasets
5. **Extract**: Return the most significant methodologies mentioned

## DOMAIN CONTEXT:
Research Domain: {domain}

## PAPER CONTENT TO ANALYZE:
{paper_content}

## EXTRACTION RULES:
- Focus on METHODOLOGICAL approaches, not just technical terms
- Include algorithm names, model architectures, and research frameworks
- Avoid generic terms like "analysis" or "evaluation"
- Extract 5-15 methodologies maximum
- Use canonical forms (e.g., "Convolutional Neural Network" not "CNN")

## EXAMPLES:

**Example 1:**
Paper: "Attention Is All You Need: Transformer Architecture for Neural Machine Translation"
Abstract: "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms..."

Thinking: This paper introduces the Transformer architecture using attention mechanisms for neural machine translation.
Methodologies: ["Transformer", "Attention Mechanism", "Neural Machine Translation", "Sequence-to-Sequence Learning"]

**Example 2:**  
Paper: "Deep Residual Learning for Image Recognition using Convolutional Networks"
Abstract: "We present a residual learning framework to ease training of deep neural networks with skip connections..."

Thinking: This focuses on residual learning and deep CNNs for computer vision tasks.
Methodologies: ["Residual Learning", "Convolutional Neural Network", "Deep Learning", "Skip Connections", "Image Recognition"]

## OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "methodologies": ["methodology1", "methodology2", "methodology3", ...]
}}

Analyze the provided papers and extract methodologies:
"""

# Concept Extraction Prompt with Chain-of-Thought
CONCEPT_EXTRACTION_PROMPT = """You are an expert research concept analyst. Your task is to extract key concepts, properties, and characteristics from academic papers.

## CHAIN-OF-THOUGHT ANALYSIS:

1. **Identify Core Concepts**: Look for research objectives, desired properties, evaluation metrics
2. **Extract Qualities**: Find adjectives and properties that describe desired outcomes
3. **Find Challenges**: Identify what researchers are trying to achieve or improve
4. **Capture Innovation**: Look for novel ideas, improvements, or breakthroughs mentioned
5. **Synthesize**: Return the most important conceptual elements

## DOMAIN CONTEXT:
Research Domain: {domain}

## PAPER CONTENT TO ANALYZE:
{paper_content}

## EXTRACTION RULES:
- Focus on CONCEPTUAL ideas, not methodological details
- Include desired properties (accuracy, efficiency, robustness, etc.)
- Include research objectives and goals
- Include innovation areas and novel contributions  
- Avoid methodology names - focus on concepts
- Extract 5-12 concepts maximum

## EXAMPLES:

**Example 1:**
Paper: "Robust Neural Networks with Improved Generalization Performance"
Abstract: "We address the challenge of model robustness and generalization in deep learning by proposing..."

Thinking: This paper focuses on robustness and generalization as key desired properties in neural networks.
Concepts: ["Robustness", "Generalization", "Model Reliability", "Performance Improvement", "Deep Learning Optimization"]

**Example 2:**
Paper: "Real-time Efficient Processing for Large-scale Data Analysis"  
Abstract: "Our framework achieves real-time processing while maintaining high accuracy for big data scenarios..."

Thinking: Key concepts are real-time processing, efficiency, accuracy, and scalability for large data.
Concepts: ["Real-time Processing", "Computational Efficiency", "Scalability", "High Accuracy", "Big Data Analysis"]

## OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "concepts": ["concept1", "concept2", "concept3", ...]
}}

Analyze the provided papers and extract key concepts:
"""

# Problem Extraction Prompt with Chain-of-Thought
PROBLEM_EXTRACTION_PROMPT = """You are an expert research problem analyst. Your task is to extract and formulate the core problems that research papers are addressing.

## CHAIN-OF-THOUGHT ANALYSIS:

1. **Identify Problems**: Look for challenges, limitations, issues mentioned in titles/abstracts
2. **Understand Context**: Consider what problems exist in the research domain
3. **Extract Root Issues**: Find the fundamental problems, not just symptoms
4. **Formulate Clearly**: State problems as clear, specific statements
5. **Prioritize Impact**: Focus on the most significant problems addressed

## DOMAIN CONTEXT:
Research Domain: {domain}

## PAPER CONTENT TO ANALYZE:
{paper_content}

## EXTRACTION RULES:
- Focus on PROBLEM statements, not solutions
- Look for words like: challenge, problem, limitation, difficulty, issue, bottleneck
- Formulate as clear problem descriptions
- Avoid methodology-specific problems - focus on domain problems
- Extract 3-8 problems maximum
- Make problems general enough to be meaningful

## EXAMPLES:

**Example 1:**
Paper: "Addressing the Vanishing Gradient Problem in Deep Neural Networks"
Abstract: "Training very deep networks suffers from vanishing gradients, making it difficult to learn..."

Thinking: The core problem is vanishing gradients preventing effective training of deep networks.
Problems: ["Vanishing gradient problem in deep network training", "Difficulty learning long-range dependencies", "Training instability in very deep architectures"]

**Example 2:**
Paper: "Handling Data Scarcity in Low-Resource Language Processing"
Abstract: "Natural language processing faces challenges with limited training data for minority languages..."

Thinking: The main issues are data scarcity and resource limitations for certain languages.
Problems: ["Limited training data for low-resource languages", "Poor performance on minority languages", "Lack of linguistic resources for underrepresented languages"]

## OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "problems": ["problem1", "problem2", "problem3", ...]
}}

Analyze the provided papers and extract core problems:
"""

# Solution Extraction Prompt with Chain-of-Thought
SOLUTION_EXTRACTION_PROMPT = """You are an expert research solution analyst. Your task is to extract solution approaches and contributions from academic papers.

## CHAIN-OF-THOUGHT ANALYSIS:

1. **Identify Solutions**: Look for proposed methods, frameworks, approaches in titles/abstracts
2. **Understand Contributions**: Find what new solutions or improvements are being offered
3. **Extract Approaches**: Focus on HOW problems are being solved, not just WHAT is solved
4. **Categorize Types**: Group solutions by approach type (algorithmic, architectural, etc.)
5. **Capture Innovation**: Highlight novel or innovative solution aspects

## DOMAIN CONTEXT:
Research Domain: {domain}

## PAPER CONTENT TO ANALYZE:
{paper_content}

## EXTRACTION RULES:
- Focus on SOLUTION approaches, not problem descriptions
- Look for words like: propose, introduce, develop, design, framework, approach, method
- Describe solutions as approaches, not just algorithm names
- Focus on the solution strategy, not implementation details
- Extract 3-8 solutions maximum
- Make solutions descriptive and informative

## EXAMPLES:

**Example 1:**
Paper: "Residual Networks: Skip Connections for Deep Learning"
Abstract: "We introduce residual learning with skip connections to enable training of very deep networks..."

Thinking: The solution is using skip connections and residual learning to address deep network training issues.
Solutions: ["Skip connection architecture for deep networks", "Residual learning framework", "Identity mapping approach for gradient flow", "Deep network training optimization strategy"]

**Example 2:**
Paper: "Attention Mechanisms for Sequence-to-Sequence Learning"
Abstract: "We propose an attention mechanism that allows models to focus on relevant input parts..."

Thinking: The solution uses attention mechanisms to improve sequence modeling and alignment.
Solutions: ["Attention-based sequence modeling", "Dynamic input focusing mechanism", "Alignment learning for sequence tasks", "Selective information processing approach"]

## OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "solutions": ["solution1", "solution2", "solution3", ...]
}}

Analyze the provided papers and extract solution approaches:
"""

# Author Name Extraction Prompt
AUTHOR_NAME_EXTRACTION_PROMPT = """You are an expert at extracting author names from research queries. 

## TASK:
Extract the author name from the user query. Look for patterns indicating they're asking about a specific researcher.

## EXTRACTION RULES:
- Look for phrases like "about [Author Name]", "tell me about [Name]", "[Name]'s research"
- Author names are usually capitalized (proper nouns)
- Return the full name as mentioned in the query
- If no author name is found, return null
- Be precise - only extract if clearly asking about a specific author

## EXAMPLES:

**Example 1:**
Query: "Tell me about John Smith's research profile"
Thinking: Clear request for information about author "John Smith"
Author: "John Smith"

**Example 2:**  
Query: "What are the trends in machine learning?"
Thinking: No specific author mentioned, asking about general trends
Author: null

**Example 3:**
Query: "Show me papers by Yann LeCun"
Thinking: Asking about specific author "Yann LeCun"
Author: "Yann LeCun"

## QUERY TO ANALYZE:
{query}

## OUTPUT FORMAT:
Return ONLY a JSON object:
{{
  "author_name": "Author Name" or null
}}
"""

# Search Term Extraction Prompt
SEARCH_TERM_EXTRACTION_PROMPT = """You are an expert at extracting search terms from research queries.

## TASK:
Extract the main search term or technology name that the user wants to find papers about.

## EXTRACTION RULES:
- Look for phrases like "papers about [term]", "research on [term]", "show me [term]"
- Extract the main topic/technology they want to search for
- Return the most specific term mentioned
- If asking about a general concept, extract the key concept
- If no clear search term, return null

## EXAMPLES:

**Example 1:**
Query: "Show me papers about transformers"
Thinking: Clear request for papers about "transformers"
Search Term: "transformers"

**Example 2:**
Query: "Find research on natural language processing"
Thinking: Looking for papers on "natural language processing"
Search Term: "natural language processing"

**Example 3:**
Query: "Who are the top authors in AI?"
Thinking: Not asking for papers about a specific topic, asking for authors
Search Term: null

## QUERY TO ANALYZE:
{query}

## OUTPUT FORMAT:
Return ONLY a JSON object:
{{
  "search_term": "search term" or null
}}
"""

# Paradigm Shift Detection Prompt
PARADIGM_SHIFT_EXTRACTION_PROMPT = """You are an expert at detecting paradigm shifts in research evolution.

## TASK:
Analyze methodology transitions between time periods to identify significant paradigm shifts.

## SHIFT DETECTION RULES:
- Look for fundamental changes in research approaches
- Identify transitions from one dominant methodology to another
- Focus on shifts that represent different ways of thinking about problems
- Consider the significance and scope of methodological changes
- Classify shift importance as "Major", "Moderate", or "Minor"

## EXAMPLES:

**Major Paradigm Shift:**
From: ["Rule-based Systems", "Expert Systems", "Symbolic AI"]  
To: ["Neural Networks", "Machine Learning", "Statistical Methods"]
Shift Type: "Symbolic to Connectionist Paradigm"
Significance: "Major"

**Moderate Paradigm Shift:**
From: ["Shallow Neural Networks", "Feature Engineering"]
To: ["Deep Learning", "Representation Learning"] 
Shift Type: "Shallow to Deep Learning Transition"
Significance: "Moderate"

## TRANSITIONS TO ANALYZE:
{transition_data}

## OUTPUT FORMAT:
Return ONLY a JSON object:
{{
  "paradigm_shifts": [
    {{
      "shift_type": "Description of the paradigm shift",
      "from_methods": ["old_method1", "old_method2"],
      "to_methods": ["new_method1", "new_method2"],
      "significance": "Major/Moderate/Minor",
      "period": "time_period",
      "description": "Detailed explanation of the shift"
    }}
  ]
}}
"""
