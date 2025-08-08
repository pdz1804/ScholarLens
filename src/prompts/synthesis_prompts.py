"""
Synthesis prompts for TechAuthor agents.
Contains Chain-of-Thought prompts for result synthesis and insight generation.
"""

class SynthesisPrompts:
    """Container for all synthesis-related prompts."""
    
    # System Prompts
    TECHNOLOGY_TRENDS_SYSTEM = "You are an expert technology analyst providing insights about research trends."
    DOMAIN_EVOLUTION_SYSTEM = "You are an expert research analyst specializing in domain evolution and paradigm shift analysis."
    GENERAL_ANALYSIS_SYSTEM = "You are an expert research analyst providing insights on academic research patterns."
    COLLABORATION_SYSTEM = "You are an expert in research collaboration analysis and network insight generation."
    AUTHOR_EXPERTISE_SYSTEM = "You are an expert in academic expertise synthesis and research leadership assessment."

    # Domain Evolution Synthesis Prompt
    DOMAIN_EVOLUTION = """## TASK:
Synthesize domain evolution analysis results into coherent, actionable insights about how the {domain} field has fundamentally changed over time.

## DATA ANALYSIS:
Based on {total_papers} papers across {periods} time periods.

### Evolution Timeline:
{evolution_timeline}

### Conceptual Evolution:
{conceptual_evolution}

## SYNTHESIS REQUIREMENTS:

1. **Major Paradigm Shifts**: Identify 2-3 fundamental changes in how the field approaches problems
   - What changed from early approaches to current methods
   - Why these transitions occurred
   - Impact on research effectiveness and direction

2. **Methodology Evolution**: Trace how research methods have evolved
   - Key methodological transitions and their drivers
   - Adoption patterns of new approaches
   - Factors that enabled or hindered methodological changes

3. **Conceptual Developments**: Explain how problem formulations have become more sophisticated
   - Evolution of problem definitions and scope
   - Development of solution strategies
   - Increasing complexity and interdisciplinary integration

4. **Key Transition Points**: Highlight specific periods where significant changes occurred
   - Critical moments that redirected the field
   - Catalysts for major changes (technological, theoretical, practical)
   - Long-term impact of these transition points

5. **Future Trajectory**: Based on historical patterns, predict likely future directions
   - Emerging research directions suggested by evolution patterns
   - Potential future paradigm shifts
   - Areas ripe for breakthrough developments

## SYNTHESIS GUIDELINES:
- Focus on qualitative insights rather than quantitative metrics
- Emphasize paradigm shifts over simple trend counting
- Explain WHY changes occurred, not just WHAT changed  
- Connect methodology evolution to real-world research impact
- Provide actionable insights for current researchers in the domain

Original Query: "{query_text}"

Provide structured synthesis with executive summary, detailed analysis, and actionable recommendations."""

    # Technology Trends Synthesis Prompt
    TECHNOLOGY_TRENDS = """## TASK:
Analyze technology trends and emerging methodological patterns in research.

## DATA ANALYSIS:
{analysis_data}

## SYNTHESIS REQUIREMENTS:

1. **Emerging Technologies**: Identify breakthrough technologies gaining traction
2. **Methodology Adoption**: Track adoption patterns of new research methods
3. **Technical Innovation**: Highlight significant technical advances
4. **Implementation Trends**: Show how new technologies are being implemented
5. **Future Technology Directions**: Predict emerging technical directions

## OUTPUT:
Structured analysis focusing on technical trends, methodology evolution, and practical implications for researchers."""

    # Collaboration Analysis Synthesis Prompt
    COLLABORATION = """## TASK:
Synthesize collaboration network analysis into insights about research partnerships and community structure.

## DATA ANALYSIS:
{collaboration_data}

## SYNTHESIS REQUIREMENTS:

1. **Network Structure**: Analyze collaboration patterns and community formation
2. **Key Collaborators**: Identify central figures and influential partnerships
3. **Collaboration Trends**: Track how collaboration patterns have evolved
4. **Research Impact**: Connect collaboration patterns to research outcomes
5. **Future Opportunities**: Suggest potential collaboration opportunities

## OUTPUT:
Comprehensive collaboration analysis with network insights and practical recommendations."""

    # Author Expertise Synthesis Prompt  
    AUTHOR_EXPERTISE = """## TASK:
Synthesize author expertise analysis into comprehensive research leadership assessment.

## DATA ANALYSIS:
{expertise_data}

## SYNTHESIS REQUIREMENTS:

1. **Expertise Areas**: Map author's research domains and specializations
2. **Impact Assessment**: Evaluate research influence and contributions
3. **Evolution Trajectory**: Track how expertise has developed over time
4. **Collaboration Patterns**: Analyze research partnerships and networks
5. **Future Potential**: Assess emerging research directions and opportunities

## OUTPUT:
Detailed expertise profile with impact assessment and growth trajectory analysis."""

    # Cross-Domain Analysis Synthesis Prompt
    CROSS_DOMAIN = """## TASK:
Synthesize cross-domain analysis to identify interdisciplinary research patterns.

## DATA ANALYSIS:
{cross_domain_data}

## SYNTHESIS REQUIREMENTS:

1. **Interdisciplinary Connections**: Map research connections across domains
2. **Knowledge Transfer**: Identify knowledge flow between fields
3. **Emerging Intersections**: Highlight new interdisciplinary areas
4. **Research Opportunities**: Suggest promising cross-domain research directions
5. **Integration Patterns**: Analyze how different fields integrate methodologies

## OUTPUT:
Comprehensive cross-domain analysis with integration insights and future opportunities."""

    # Paper Impact Synthesis Prompt
    PAPER_IMPACT = """## TASK:
Synthesize paper impact analysis into research influence assessment.

## DATA ANALYSIS:
{impact_data}

## SYNTHESIS REQUIREMENTS:

1. **Influence Patterns**: Analyze how research influences the field
2. **Key Contributions**: Identify significant research contributions
3. **Impact Metrics**: Evaluate various measures of research impact
4. **Temporal Influence**: Track how impact evolves over time
5. **Future Influence**: Predict potential long-term impact

## OUTPUT:
Detailed impact assessment with influence patterns and contribution analysis."""

    # Author Productivity Synthesis Prompt
    AUTHOR_PRODUCTIVITY = """## TASK:
Synthesize author productivity analysis into comprehensive research output assessment.

## DATA ANALYSIS:
{productivity_data}

## SYNTHESIS REQUIREMENTS:

1. **Output Patterns**: Analyze research publication patterns
2. **Quality Assessment**: Evaluate research quality and impact
3. **Temporal Trends**: Track productivity changes over time
4. **Collaboration Impact**: Assess how collaboration affects productivity
5. **Future Trajectory**: Predict productivity trends

## OUTPUT:
Comprehensive productivity analysis with quality assessment and trend analysis."""

    # General Analysis Synthesis Prompt
    GENERAL = """## TASK:
Provide comprehensive analysis synthesis for complex research queries.

## DATA ANALYSIS:
{analysis_data}

## SYNTHESIS REQUIREMENTS:

1. **Key Insights**: Extract most significant findings
2. **Pattern Recognition**: Identify important research patterns
3. **Trend Analysis**: Analyze temporal and methodological trends
4. **Impact Assessment**: Evaluate research influence and significance
5. **Future Directions**: Suggest future research opportunities

## OUTPUT:
Structured analysis with key insights, patterns, and actionable recommendations."""

    # Enhancement Prompt for Complex Synthesis
    ENHANCEMENT = """Enhance the synthesis by:

1. **Deep Analysis**: Provide deeper insights into identified patterns
2. **Contextual Understanding**: Place findings in broader research context
3. **Practical Implications**: Connect insights to real-world applications
4. **Future Predictions**: Make informed predictions about future developments
5. **Actionable Recommendations**: Provide specific, implementable recommendations

Focus on creating actionable insights that researchers and practitioners can use to inform their work."""
