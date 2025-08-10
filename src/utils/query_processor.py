"""
Query processing utilities for TechAuthor system.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


def log_or_print(message: str, logger=None):
    """Utility function to log or print a message."""
    if logger:
        logger.info(message)
    else:
        print(message)


async def process_single_query(system, query: str, params: Dict[str, Any], logger) -> object:
    """Process a single query and return response."""
    logger.info(f"Processing query: {query}")
    start_time = datetime.now()
    
    try:
        response = await system.aquery(query, params)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        logger.info(f"Response type: {response.response_type}")
        logger.info(f"Confidence: {response.confidence:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise


async def process_batch_queries(system, batch_file: str, params: Dict[str, Any], logger) -> List[object]:
    """Process multiple queries from file."""
    logger.info(f"Processing batch queries from: {batch_file}")
    
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Found {len(queries)} queries to process")
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query}")
            try:
                response = await process_single_query(system, query, params, logger)
                results.append(response)
            except Exception as e:
                logger.error(f"Failed to process query {i}: {str(e)}")
                # Create error response matching the system's error response format
                from src.core.models import Query, SystemResponse
                from src.core.system import QueryResponseAdapter
                
                error_query = Query(text=query, parameters=params)
                error_system_response = SystemResponse(
                    query=error_query,
                    response_type="error",
                    result={"error": str(e)},
                    processing_time=0.0,
                    agent_chain=["error"],
                    confidence=0.0,
                    sources=[]
                )
                # Wrap in adapter to match the expected interface
                error_response = QueryResponseAdapter(error_system_response)
                results.append(error_response)
        
        logger.info(f"Batch processing completed: {len(results)} queries processed")
        return results
        
    except FileNotFoundError:
        logger.error(f"Batch file not found: {batch_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing batch file: {str(e)}")
        raise


async def interactive_mode(system, params: Dict[str, Any], logger) -> None:
    """Run interactive query mode."""
    logger.info("Starting interactive mode")
    
    logger.info("=" * 60)
    logger.info("TechAuthor Interactive Mode")
    logger.info("=" * 60)
    logger.info("Enter your research queries. Type 'quit' or 'exit' to stop.")
    logger.info("Type 'help' for example queries.")
    logger.info("-" * 60)
    
    while True:
        try:
            query = input("\nQuery> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            
            if query.lower() == 'help':
                show_help_examples()
                continue
            
            if not query:
                continue
            
            logger.info("Processing...")
            response = await process_single_query(system, query, params, logger)
            print_query_response(response, "text", logger)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Interactive query failed: {str(e)}")
            logger.error(f"Error: {str(e)}")


def show_help_examples():
    """Show example queries for interactive mode."""
    examples = [
        "Who are the top authors in AI Agents?",
        "What are emerging trends in Computer Vision?",
        "Who collaborates with Geoffrey Hinton?",
        "How has Deep Learning evolved over time?",
        "Which authors work across multiple AI domains?",
        "Most influential papers in Natural Language Processing?",
        "Most prolific authors in Machine Learning?",
        "Leading institutions in Quantum Computing research?"
    ]
    
    # For interactive mode, we still need to print to show the user
    # But we'll use a cleaner format without special characters
    print("")
    print("Example queries:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")


def print_query_response(response, format_type: str = "text", logger=None) -> None:
    """Print a single query response."""
    if format_type == "json":
        import json
        result_dict = {
            "query": response.query.text,
            "type": response.response_type,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "summary": response.summary,
            "insights": response.insights,
            "data": response.data
        }
        result_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
        if logger:
            logger.info("JSON Response:")
            for line in result_json.split('\n'):
                logger.info(line)
        else:
            print(result_json)
    else:
        separator_line = "=" * 80
        if logger:
            logger.info("")
            logger.info(separator_line)
            logger.info(f"QUERY: {response.query.text}")
            logger.info(separator_line)
            logger.info(f"Type: {response.response_type}")
            logger.info(f"Processing Time: {response.processing_time:.2f}s")
            logger.info(f"Confidence: {response.confidence:.2f}")
            logger.info("")
            logger.info("Summary:")
            logger.info(response.summary)
        else:
            print(f"\n{separator_line}")
            print(f"QUERY: {response.query.text}")
            print(separator_line)
            print(f"Type: {response.response_type}")
            print(f"Processing Time: {response.processing_time:.2f}s")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"\nSummary:")
            print(response.summary)
        
        if response.insights:
            if logger:
                logger.info("")
                logger.info("Key Insights:")
                for insight in response.insights:
                    logger.info(f"  - {insight}")
            else:
                print(f"\nKey Insights:")
                for insight in response.insights:
                    print(f"  - {insight}")
        
        if response.data:
            if logger:
                logger.info("")
                logger.info("Top Results:")
            else:
                print(f"\nTop Results:")
                
            if 'authors' in response.data and response.data['authors']:
                # Show top 20 authors and explain the score
                authors_to_show = min(20, len(response.data['authors']))
                log_or_print(f"Showing top {authors_to_show} authors (ranked by expertise score):", logger)
                log_or_print("  Note: Expertise score combines paper count, domain relevance, and research impact", logger)
                log_or_print("", logger)
                
                for i, author in enumerate(response.data['authors'][:authors_to_show], 1):
                    if isinstance(author, dict):
                        author_name = author.get('author', author.get('name', 'Unknown'))
                        paper_count = author.get('paper_count', author.get('count', 0))
                        expertise_score = author.get('expertise_score', 0.0)
                        subjects = author.get('subjects', [])
                        papers = author.get('papers', [])
                        
                        log_or_print(f"  {i:2d}. {author_name}", logger)
                        log_or_print(f"      Papers: {paper_count}, Expertise Score: {expertise_score:.1f}", logger)
                        if subjects:
                            log_or_print(f"      Research Areas: {', '.join(subjects)}", logger)
                        
                        # Display the papers this author contributed to
                        if papers:
                            log_or_print(f"      Published Papers:", logger)
                            for j, paper in enumerate(papers, 1):
                                if isinstance(paper, dict):
                                    title = paper.get('title', 'Unknown Title')
                                    paper_id = paper.get('paper_id', '')
                                    log_or_print(f"         {j}. Paper ID: {paper_id}", logger)
                                    log_or_print(f"            {title}", logger)
                                else:
                                    # Handle legacy format where papers were just IDs
                                    log_or_print(f"         {j}. Paper ID: {paper}", logger)
                        
                        log_or_print("", logger)  # Empty line for readability
                    else:
                        log_or_print(f"  {i}. {author}", logger)
            elif 'papers' in response.data and response.data['papers']:
                # Show more papers with abstracts
                papers_to_show = min(10, len(response.data['papers']))
                for i, paper in enumerate(response.data['papers'][:papers_to_show], 1):
                    if isinstance(paper, dict):
                        title = paper.get('title', 'Unknown Title')
                        abstract = paper.get('abstract', '')
                        authors = paper.get('authors', [])
                        log_or_print(f"  {i}. {title}", logger)
                        if authors:
                            authors_text = ', '.join(authors[:3])
                            if len(authors) > 3:
                                authors_text += "..."
                            log_or_print(f"     Authors: {authors_text}", logger)
                        if abstract:
                            abstract_excerpt = abstract[:200] + "..." if len(abstract) > 200 else abstract
                            log_or_print(f"     Abstract: {abstract_excerpt}", logger)
                        log_or_print("", logger)
                    else:
                        log_or_print(f"  {i}. {paper}", logger)
            elif 'trends' in response.data and response.data['trends']:
                for i, trend in enumerate(response.data['trends'][:10], 1):  # Show top 10 trends
                    if isinstance(trend, dict):
                        technology = trend.get('technology', trend.get('name', 'Unknown Technology'))
                        total_papers = trend.get('total_papers', 0)
                        relevance = trend.get('relevance_score', 0.0)
                        trend_direction = trend.get('trend_direction', 'current')
                        
                        if trend_direction == 'current':
                            # For current technology analysis (single year data)
                            if i == 1:
                                log_or_print("Single Year Technology Analysis:", logger)
                            log_or_print(f"  {i:2d}. {technology}", logger)
                            log_or_print(f"      Papers: {total_papers}, Relevance: {relevance:.1%}", logger)
                        else:
                            # For multi-year trend analysis
                            if i == 1:
                                log_or_print("Multi-Year Technology Trend Analysis:", logger)
                            slope = trend.get('trend_slope', 0.0)
                            log_or_print(f"  {i:2d}. {technology}", logger)
                            log_or_print(f"      Papers: {total_papers}, Trend: {trend_direction} (slope: {slope:.3f})", logger)
                    else:
                        log_or_print(f"  {i}. {trend}", logger)
            elif 'search_results' in response.data and response.data['search_results']:
                # Paper search results - handle exact matches and paper details
                search_results = response.data['search_results']
                
                if isinstance(search_results, dict) and 'search_results' in search_results:
                    # Handle nested search_results structure
                    papers = search_results.get('search_results', [])
                    search_term = search_results.get('search_term', 'Unknown')
                    total_matches = search_results.get('total_matches', len(papers))
                elif isinstance(search_results, dict) and 'matching_papers' in search_results:
                    # Handle matching_papers structure
                    papers = search_results.get('matching_papers', [])
                    search_term = search_results.get('search_term', 'Unknown')
                    total_matches = search_results.get('papers_found', len(papers))
                else:
                    # Handle direct list of papers
                    papers = search_results if isinstance(search_results, list) else []
                    search_term = 'Unknown'
                    total_matches = len(papers)
                
                if papers:
                    # Display summary statistics first
                    log_or_print(f"Paper Search Results for '{search_term}':", logger)
                    log_or_print(f"Found {total_matches} matching papers", logger)
                    
                    # Calculate and display summary statistics
                    if len(papers) > 0:
                        # Match type statistics
                        exact_matches = sum(1 for p in papers if p.get('match_type', '').startswith('exact'))
                        title_matches = sum(1 for p in papers if p.get('match_type', '') == 'title')
                        abstract_matches = sum(1 for p in papers if p.get('match_type', '') == 'abstract')
                        
                        # Year range statistics
                        years = []
                        domains = []
                        for p in papers:
                            if 'year' in p and p['year']:
                                years.append(p['year'])
                            elif 'date_submitted' in p and p['date_submitted'] not in ['Unknown Date', 'Date Not Available']:
                                try:
                                    # Extract year from date string
                                    if '-' in p['date_submitted']:
                                        year = int(p['date_submitted'].split('-')[0])
                                    else:
                                        # Might just be a year
                                        year = int(p['date_submitted'])
                                    if year and 1900 <= year <= 2030:  # Reasonable year range
                                        years.append(year)
                                except:
                                    pass
                            
                            # Handle domain extraction for statistics
                            domain = p.get('domain', '').strip()
                            if not domain or domain in ['Unknown Domain', 'Computer Science']:
                                # Try subjects as fallback
                                subjects = p.get('subjects', [])
                                if subjects and isinstance(subjects, list) and len(subjects) > 0:
                                    domain = subjects[0].strip()
                            
                            if domain and domain not in ['Unknown Domain', '']:
                                domains.append(domain)
                        
                        log_or_print("", logger)
                        log_or_print("Search Summary:", logger)
                        
                        if exact_matches > 0:
                            log_or_print(f"  - {exact_matches} exact matches found", logger)
                        if title_matches > 0:
                            log_or_print(f"  - {title_matches} title matches", logger)
                        if abstract_matches > 0:
                            log_or_print(f"  - {abstract_matches} abstract matches", logger)
                        
                        if years:
                            min_year, max_year = min(years), max(years)
                            if min_year == max_year:
                                log_or_print(f"  - All papers from year {min_year}", logger)
                            else:
                                log_or_print(f"  - Papers span from {min_year} to {max_year}", logger)
                        
                        if domains:
                            unique_domains = list(set(domains))
                            if len(unique_domains) == 1:
                                log_or_print(f"  - All papers in domain: {unique_domains[0]}", logger)
                            else:
                                domain_text = ', '.join(unique_domains[:3])
                                if len(unique_domains) > 3:
                                    domain_text += f" (and {len(unique_domains) - 3} others)"
                                log_or_print(f"  - Research domains: {domain_text}", logger)
                    
                    log_or_print("", logger)
                    log_or_print("Detailed Results:", logger)
                    
                    for i, paper in enumerate(papers[:10], 1):  # Show top 10 results
                        if isinstance(paper, dict):
                            paper_id = paper.get('paper_id', 'Unknown ID')
                            title = paper.get('title', 'Unknown Title')
                            authors = paper.get('authors', [])
                            abstract = paper.get('abstract', '')
                            
                            # Handle domain - try multiple fields and provide fallback
                            domain = paper.get('domain', '')
                            if not domain or domain == 'Unknown Domain':
                                # Try subjects as fallback for domain
                                subjects = paper.get('subjects', [])
                                if subjects and isinstance(subjects, list) and len(subjects) > 0:
                                    domain = subjects[0]  # Use first subject as domain
                                else:
                                    domain = 'Computer Science'  # Default fallback
                            
                            # Handle date - try multiple fields and provide fallback
                            date_submitted = paper.get('date_submitted', '')
                            if not date_submitted or date_submitted == 'Unknown Date':
                                # Try extracting year field
                                year = paper.get('year')
                                if year:
                                    date_submitted = str(year)
                                else:
                                    date_submitted = 'Date Not Available'
                            
                            match_type = paper.get('match_type', 'Unknown')
                            score = paper.get('score', paper.get('match_score', 0.0))
                            
                            # Format the paper information
                            log_or_print(f"  {i}. Paper ID: {paper_id}", logger)
                            log_or_print(f"     Title: {title}", logger)
                            
                            if authors:
                                if isinstance(authors, list):
                                    authors_text = ', '.join(authors[:5])
                                    if len(authors) > 5:
                                        authors_text += f" (and {len(authors) - 5} others)"
                                else:
                                    authors_text = str(authors)
                                log_or_print(f"     Authors: {authors_text}", logger)
                            
                            log_or_print(f"     Domain: {domain}", logger)
                            log_or_print(f"     Date: {date_submitted}", logger)
                            
                            if match_type and match_type != 'Unknown':
                                log_or_print(f"     Match Type: {match_type.replace('_', ' ').title()}", logger)
                            
                            if score > 0:
                                log_or_print(f"     Relevance Score: {score:.3f}", logger)
                            
                            if abstract:
                                # Truncate abstract if it's too long
                                abstract_display = abstract[:300] + "..." if len(abstract) > 300 else abstract
                                log_or_print(f"     Abstract: {abstract_display}", logger)
                            
                            log_or_print("", logger)  # Empty line between papers
                else:
                    log_or_print(f"No papers found matching '{search_term}'", logger)
                    log_or_print("Try different search terms or check the paper ID", logger)
            elif 'collaboration_overview' in response.data:
                # General collaboration analysis (when no focal author found or general analysis)
                overview = response.data['collaboration_overview']
                total_authors = overview.get('total_authors', 0)
                total_collaborations = overview.get('total_collaborations', 0)
                network_density = overview.get('network_density', 0.0)
                most_collaborative = overview.get('most_collaborative_authors', [])
                
                log_or_print("Collaboration Network Analysis:", logger)
                log_or_print(f"Total Authors: {total_authors}", logger)
                log_or_print(f"Total Collaborations: {total_collaborations}", logger)
                log_or_print(f"Network Density: {network_density:.3f}", logger)
                log_or_print("", logger)
                
                if most_collaborative:
                    log_or_print("Most Collaborative Authors:", logger)
                    for i, author_info in enumerate(most_collaborative[:10], 1):
                        if isinstance(author_info, dict):
                            name = author_info.get('author', 'Unknown')
                            centrality = author_info.get('centrality', 0.0)
                            collab_count = author_info.get('collaboration_count', 'N/A')
                            log_or_print(f"  {i:2d}. {name}", logger)
                            log_or_print(f"      Network Centrality: {centrality:.3f}", logger)
                            if collab_count != 'N/A':
                                log_or_print(f"      Collaborations: {collab_count}", logger)
                        else:
                            log_or_print(f"  {i}. {author_info}", logger)
                    log_or_print("", logger)
                    log_or_print("Note: The requested author may not be present in the dataset.", logger)
                    log_or_print("Showing general collaboration patterns from retrieved papers instead.", logger)
            elif 'focal_author' in response.data and 'top_collaborators' in response.data:
                # Author collaboration results
                focal_author = response.data.get('focal_author', 'Unknown Author')
                collaborators = response.data.get('top_collaborators', [])
                total_collaborators = response.data.get('total_collaborators', 0)
                network_size = response.data.get('network_size', 0)
                
                log_or_print(f"Collaboration Network for: {focal_author}", logger)
                log_or_print(f"Total Collaborators: {total_collaborators}, Network Size: {network_size}", logger)
                log_or_print("", logger)
                
                if collaborators:
                    log_or_print("Top Collaborators:", logger)
                    for i, collab in enumerate(collaborators[:10], 1):
                        if isinstance(collab, dict):
                            name = collab.get('collaborator', 'Unknown')
                            count = collab.get('collaboration_count', 0)
                            papers = collab.get('shared_papers', 0)
                            subjects = collab.get('common_subjects', [])
                            centrality = collab.get('centrality', 0.0)
                            
                            log_or_print(f"  {i:2d}. {name}", logger)
                            log_or_print(f"      Collaborations: {count}, Shared Papers: {papers}", logger)
                            log_or_print(f"      Network Centrality: {centrality:.3f}", logger)
                            if subjects:
                                log_or_print(f"      Common Research Areas: {', '.join(subjects)}", logger)
                            log_or_print("", logger)
                else:
                    log_or_print("No collaborators found in the retrieved papers.", logger)
            elif 'collaborators' in response.data:
                # General collaboration analysis (when no focal author found)
                collaborators = response.data.get('collaborators', [])
                if collaborators:
                    log_or_print("Top Collaborative Authors:", logger)
                    for i, collab in enumerate(collaborators[:10], 1):
                        if isinstance(collab, dict):
                            name = collab.get('author', collab.get('name', 'Unknown'))
                            count = collab.get('collaboration_count', 0)
                            centrality = collab.get('centrality', 0.0)
                            log_or_print(f"  {i:2d}. {name}", logger)
                            log_or_print(f"      Collaborations: {count}, Centrality: {centrality:.3f}", logger)
                        else:
                            log_or_print(f"  {i}. {collab}", logger)
            elif 'domain_evolution_analysis' in response.data or 'paradigm_shifts' in response.data:
                # Domain evolution results with cleaner formatting
                log_or_print("Evolution Analysis:", logger)
                
                # Show evolution analysis section if available
                if 'evolution_analysis' in response.data:
                    analysis_content = response.data['evolution_analysis']
                    if isinstance(analysis_content, str) and analysis_content.strip():
                        log_or_print(analysis_content.strip(), logger)
                    log_or_print("", logger)
                
                # Show evolution overview section if available  
                if 'evolution_overview' in response.data:
                    overview_content = response.data['evolution_overview']
                    log_or_print("Evolution Overview:", logger)
                    if isinstance(overview_content, str) and overview_content.strip():
                        log_or_print(overview_content.strip(), logger)
                    log_or_print("", logger)
                elif 'evolution_summary' in response.data:
                    # Fallback to old format
                    summary = response.data['evolution_summary']
                    log_or_print("Evolution Overview:", logger)
                    if isinstance(summary, dict):
                        time_periods = summary.get('time_periods_analyzed', [])
                        if time_periods:
                            log_or_print(f"   Time Period: {' -> '.join(time_periods)}", logger)
                        total_papers = summary.get('total_papers_analyzed', 0)
                        log_or_print(f"   Papers Analyzed: {total_papers:,}", logger)
                    log_or_print("", logger)
                
                # Show paradigm shifts in a readable format
                if 'paradigm_shifts' in response.data and response.data['paradigm_shifts']:
                    log_or_print("Key Paradigm Shifts:", logger)
                    shifts = response.data['paradigm_shifts']
                    for i, shift in enumerate(shifts[:5], 1):  # Show top 5 shifts
                        if isinstance(shift, dict):
                            period = shift.get('period', 'Unknown period')
                            shift_type = shift.get('shift_type', 'Unknown type')
                            description = shift.get('description', 'No description')
                            significance = shift.get('significance', 'Medium')
                            
                            # Clean up the description for better readability
                            if description.startswith('Transition from'):
                                description = description.replace('Transition from ', '').replace(' to ', ' -> ')
                            elif description.startswith('Problem focus changed from'):
                                description = description.replace('Problem focus changed from ', '').replace(' to ', ' -> ')
                            
                            significance_text = {'High': '[HIGH]', 'Medium': '[MEDIUM]', 'Low': '[LOW]'}.get(significance, '[MEDIUM]')
                            
                            log_or_print(f"   {i}. {significance_text} {shift_type} ({period})", logger)
                            log_or_print(f"      {description}", logger)
                            log_or_print("", logger)
                
                # Show key insights in a more narrative format
                if 'key_insights' in response.data and response.data['key_insights']:
                    insights = response.data['key_insights']
                    log_or_print("Key Insights:", logger)
                    for i, insight in enumerate(insights[:10], 1):
                        if isinstance(insight, str):
                            # Clean up technical language
                            cleaned_insight = insight.replace('Key methodology transitions identified:', 'Major research approach shifts:')
                            cleaned_insight = cleaned_insight.replace('Recent methodological shift: Introduction of', 'New methodologies emerged:')
                            cleaned_insight = cleaned_insight.replace('Conceptual evolution shows', 'Problem focus evolution:')
                            log_or_print(f"   - {cleaned_insight}", logger)
                        else:
                            log_or_print(f"   - {insight}", logger)
                    log_or_print("", logger)
                
                # Show future trajectory if available
                if 'future_trajectory_detailed' in response.data and response.data['future_trajectory_detailed']:
                    trajectory_content = response.data['future_trajectory_detailed']
                    log_or_print("Future Trajectory:", logger)
                    if isinstance(trajectory_content, str) and trajectory_content.strip():
                        log_or_print(trajectory_content.strip(), logger)
                    log_or_print("", logger)
                elif 'future_trajectory' in response.data and response.data['future_trajectory']:
                    # Fallback to old format
                    trajectory = response.data['future_trajectory']
                    log_or_print("Future Trajectory:", logger)
                    if isinstance(trajectory, dict):
                        predictions = trajectory.get('predictions', [])
                        for prediction in predictions[:3]:  # Show top 3 predictions
                            log_or_print(f"   - {prediction}", logger)
                    elif isinstance(trajectory, list):
                        for prediction in trajectory[:3]:
                            log_or_print(f"   - {prediction}", logger)
                    log_or_print("", logger)
            elif 'author_profile' in response.data:
                # Author statistics results
                author_profile = response.data.get('author_profile', {})
                insights = response.data.get('insights', [])
                
                if author_profile.get('found'):
                    try:
                        # Author found - display comprehensive profile
                        author_name = author_profile.get('author_name', 'Unknown Author')
                        stats = author_profile.get('stats', {})
                        
                        # Ensure stats is a dictionary to avoid type errors
                        if not isinstance(stats, dict):
                            log_or_print(f"Error: Invalid stats data type for {author_name}: {type(stats)}", logger)
                            return
                        
                        log_or_print(f"Author Profile: {author_name}", logger)
                        log_or_print("="*60, logger)
                        
                        # Basic statistics
                        total_papers = stats.get('total_papers', 0)
                        years_active = stats.get('years_active', [])
                        
                        # Ensure years_active is a list
                        if not isinstance(years_active, list):
                            years_active = []
                        
                        years_span = f"{min(years_active)}-{max(years_active)}" if len(years_active) > 1 else str(years_active[0]) if years_active else "Unknown"
                        
                        log_or_print(f"Publication Metrics:", logger)
                        log_or_print(f"  • Total Papers: {total_papers}", logger)
                        log_or_print(f"  • Years Active: {years_span} ({len(years_active)} years)", logger)
                        
                        # Research areas with main subject identification
                        subjects = stats.get('subjects', [])
                        main_subject = stats.get('main_subject') or stats.get('primary_subject')
                        
                        if subjects and isinstance(subjects, list):
                            log_or_print("", logger)
                            log_or_print("Research Areas Analysis:", logger)
                            
                            # Show main/primary subject if available
                            if main_subject:
                                log_or_print(f"  • Main Subject: {main_subject}", logger)
                            
                            # Show all subjects with detailed breakdown
                            log_or_print(f"  • All Research Areas ({len(subjects)} total):", logger)
                            for i, subject in enumerate(subjects[:10], 1):  # Show top 10 subjects
                                marker = " ⭐" if subject == main_subject else ""
                                log_or_print(f"    {i}. {subject}{marker}", logger)
                            
                            if len(subjects) > 10:
                                log_or_print(f"    (and {len(subjects) - 10} additional areas)", logger)
                        
                        # Collaboration metrics
                        num_collaborators = stats.get('num_collaborators', 0)
                        unique_coauthors = stats.get('unique_coauthors', 0)
                        
                        if num_collaborators > 0 or unique_coauthors > 0:
                            collab_count = max(num_collaborators, unique_coauthors)  # Use the available metric
                            log_or_print(f"  • Unique Collaborators: {collab_count}", logger)
                            
                            # Average collaborators per paper
                            if total_papers > 0:
                                avg_collab_per_paper = collab_count / total_papers
                                log_or_print(f"  • Average Collaborators per Paper: {avg_collab_per_paper:.1f}", logger)
                        
                        # Yearly distribution
                        yearly_papers = stats.get('yearly_papers', {})
                        
                        if yearly_papers and isinstance(yearly_papers, dict):
                            log_or_print("", logger)
                            log_or_print("Publication Timeline:", logger)
                            
                            # Sort years and show distribution
                            try:
                                sorted_years = sorted(yearly_papers.items())
                                for year, count in sorted_years:
                                    log_or_print(f"  • {year}: {count} paper{'s' if count != 1 else ''}", logger)
                                
                                # Find peak year
                                peak_year = max(yearly_papers, key=yearly_papers.get)
                                peak_count = yearly_papers[peak_year]
                                log_or_print(f"  • Most Productive Year: {peak_year} ({peak_count} papers)", logger)
                            except Exception as e:
                                log_or_print(f"Error processing yearly distribution: {e}", logger)
                        
                        # Top collaborators if available
                        top_collaborators = stats.get('top_collaborators', [])
                        
                        if top_collaborators and isinstance(top_collaborators, list):
                            log_or_print("", logger)
                            log_or_print("Detailed Collaborators Analysis:", logger)
                            for i, collab in enumerate(top_collaborators[:10], 1):  # Show top 10 collaborators
                                if isinstance(collab, dict):
                                    name = collab.get('name', collab.get('collaborator', 'Unknown'))
                                    count = collab.get('count', collab.get('papers', 1))
                                    percentage = (count / total_papers) * 100 if total_papers > 0 else 0
                                    
                                    # Show collaboration details
                                    log_or_print(f"  {i}. {name}", logger)
                                    log_or_print(f"     • Joint Papers: {count} ({percentage:.1f}% of total)", logger)
                                    
                                    # Show common subjects/domains if available
                                    common_subjects = collab.get('common_subjects', [])
                                    if common_subjects and isinstance(common_subjects, list):
                                        log_or_print(f"     • Common Research Areas: {', '.join(common_subjects)}", logger)
                                        if len(common_subjects) > 3:
                                            log_or_print(f"       (and {len(common_subjects) - 3} more areas)", logger)
                                    
                                    # Show collaboration timeframe if available
                                    years = collab.get('collaboration_years', [])
                                    if years and isinstance(years, list):
                                        year_span = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
                                        log_or_print(f"     • Collaboration Period: {year_span} ({len(years)} years)", logger)
                                    
                                    # Show recent collaboration status
                                    last_collab = collab.get('last_collaboration_year')
                                    if last_collab:
                                        current_year = 2025  # Adjust as needed
                                        years_since = current_year - int(last_collab)
                                        if years_since <= 2:
                                            log_or_print(f"     • Status: Active (last collaboration: {last_collab})", logger)
                                        else:
                                            log_or_print(f"     • Status: Inactive (last collaboration: {last_collab}, {years_since} years ago)", logger)
                                    
                                    log_or_print("", logger)
                                else:
                                    log_or_print(f"  {i}. {collab}", logger)
                        
                        # Research domains or subjects breakdown with detailed analysis
                        if isinstance(stats, dict) and 'subject_distribution' in stats:
                            subject_dist = stats['subject_distribution']
                            
                            if isinstance(subject_dist, dict):
                                log_or_print("", logger)
                                log_or_print("Subject Distribution Analysis:", logger)
                                
                                try:
                                    sorted_subjects = sorted(subject_dist.items(), key=lambda x: x[1], reverse=True)
                                    
                                    # Show top subjects with detailed stats
                                    for i, (subject, count) in enumerate(sorted_subjects[:8], 1):  # Show top 8
                                        percentage = (count / total_papers) * 100 if total_papers > 0 else 0
                                        
                                        # Determine subject strength
                                        if percentage >= 50:
                                            strength = "Primary Focus"
                                        elif percentage >= 25:
                                            strength = "Major Area"
                                        elif percentage >= 10:
                                            strength = "Significant Area"
                                        else:
                                            strength = "Contributing Area"
                                        
                                        log_or_print(f"  {i}. {subject}", logger)
                                        log_or_print(f"     • Papers: {count} ({percentage:.1f}% of total)", logger)
                                        log_or_print(f"     • Classification: {strength}", logger)
                                        
                                        # Show trend if we have yearly data
                                        yearly_papers = stats.get('yearly_papers', {})
                                        if yearly_papers and len(yearly_papers) >= 3:  # Need at least 3 years for trend
                                            recent_years = sorted(yearly_papers.keys())[-3:]  # Last 3 years
                                            trend_info = f"Recent activity in {', '.join(recent_years)}"
                                            log_or_print(f"     • Trend: {trend_info}", logger)
                                        
                                        log_or_print("", logger)
                                    
                                    if len(sorted_subjects) > 8:
                                        remaining = len(sorted_subjects) - 8
                                        log_or_print(f"  (and {remaining} additional subject areas)", logger)
                                    
                                    # Subject diversity analysis
                                    total_subjects = len(sorted_subjects)
                                    if total_subjects > 0:
                                        # Calculate diversity metrics
                                        top_subject_percentage = (sorted_subjects[0][1] / total_papers) * 100 if total_papers > 0 else 0
                                        
                                        log_or_print("", logger)
                                        log_or_print("Research Diversity Metrics:", logger)
                                        log_or_print(f"  • Total Distinct Subjects: {total_subjects}", logger)
                                        log_or_print(f"  • Primary Subject Dominance: {top_subject_percentage:.1f}%", logger)
                                        
                                        if top_subject_percentage >= 70:
                                            diversity_level = "Highly Specialized"
                                        elif top_subject_percentage >= 50:
                                            diversity_level = "Moderately Specialized"
                                        elif top_subject_percentage >= 30:
                                            diversity_level = "Balanced Focus"
                                        else:
                                            diversity_level = "Highly Diverse"
                                        
                                        log_or_print(f"  • Research Profile: {diversity_level}", logger)
                                    
                                except Exception as e:
                                    log_or_print(f"Error processing detailed subject distribution: {e}", logger)
                        
                        # Show paper details if available
                        paper_details = stats.get('paper_details', [])
                        if paper_details and isinstance(paper_details, list):
                            log_or_print("", logger)
                            log_or_print("Publication Details:", logger)
                            for i, paper in enumerate(paper_details[:3], 1):  # Show first 3 papers
                                if isinstance(paper, dict):
                                    title = paper.get('title', 'Unknown Title')
                                    year = paper.get('year', 'Unknown')
                                    paper_id = paper.get('paper_id', 'No ID')
                                    authors = paper.get('authors', [])
                                    log_or_print(f"  {i}. [{year}] {title}", logger)
                                    log_or_print(f"     Paper ID: {paper_id}", logger)
                                    if isinstance(authors, list) and len(authors) > 1:
                                        other_authors = [a for a in authors if a != author_name]
                                        if other_authors:
                                            log_or_print(f"     Co-authors: {', '.join(other_authors[:3])}{'...' if len(other_authors) > 3 else ''}", logger)
                            
                            if len(paper_details) > 3:
                                log_or_print(f"     (and {len(paper_details) - 3} more papers)", logger)
                        
                        # Enhanced insights from LLM if available
                        enhanced_data = response.data.get('enhanced', {})
                        
                        if enhanced_data and isinstance(enhanced_data, dict) and 'llm_insights' in enhanced_data:
                            llm_insights = enhanced_data['llm_insights']
                            if llm_insights:
                                log_or_print("", logger)
                                log_or_print("Enhanced Analysis:", logger)
                                if isinstance(llm_insights, list):
                                    for insight in llm_insights[:3]:
                                        log_or_print(f"  • {insight}", logger)
                                else:
                                    log_or_print(f"  • {llm_insights}", logger)
                    
                    except Exception as e:
                        log_or_print(f"ERROR: Exception in author profile processing: {e}", logger)
                        log_or_print(f"ERROR: Exception type: {type(e)}", logger)
                        import traceback
                        log_or_print(f"ERROR: Traceback: {traceback.format_exc()}", logger)
                
                else:
                    # Author not found - show suggestions
                    author_name = author_profile.get('author_name', 'Unknown')
                    log_or_print(f"Author Profile Search Results", logger)
                    log_or_print("="*50, logger)
                    log_or_print(f"❌ No exact match found for: '{author_name}'", logger)
                    
                    similar_authors = author_profile.get('similar_authors', [])
                    if similar_authors:
                        log_or_print("", logger)
                        log_or_print("💡 Similar author names found in database:", logger)
                        for i, similar in enumerate(similar_authors[:10], 1):
                            if isinstance(similar, dict):
                                name = similar.get('name', similar.get('author', 'Unknown'))
                                similarity = similar.get('similarity', similar.get('score', 0))
                                if similarity > 0:
                                    log_or_print(f"  {i}. {name} (similarity: {similarity:.2f})", logger)
                                else:
                                    log_or_print(f"  {i}. {name}", logger)
                            else:
                                log_or_print(f"  {i}. {similar}", logger)
                        
                        log_or_print("", logger)
                        log_or_print("💭 Suggestion: Try querying one of the similar names above, or check the spelling.", logger)
                    else:
                        log_or_print("", logger)
                        log_or_print("❌ No similar author names found.", logger)
                        log_or_print("💭 Suggestion: Check the spelling or try a different author name.", logger)
            elif 'cross_domain_authors' in response.data or 'cross_domain_analysis' in response.data:
                # Cross-domain analysis results
                log_or_print("Cross-Domain Research Analysis", logger)
                log_or_print("="*60, logger)
                
                # Get the cross-domain data - it might be nested under 'cross_domain_analysis'
                cross_domain_data = response.data.get('cross_domain_analysis', response.data)
                
                # Show interdisciplinary authors
                cross_domain_authors = cross_domain_data.get('cross_domain_authors', [])
                if cross_domain_authors:
                    log_or_print(f"Found {len(cross_domain_authors)} interdisciplinary researchers", logger)
                    log_or_print("", logger)
                    log_or_print("Top 5 Interdisciplinary Researchers:", logger)

                    for i, author_info in enumerate(cross_domain_authors[:5], 1):  # Show top 5
                        if isinstance(author_info, dict):
                            name = author_info.get('author', author_info.get('name', 'Unknown'))
                            domains = author_info.get('domains', [])
                            domain_count = author_info.get('domain_count', len(domains))
                            score = author_info.get('interdisciplinary_score', 0)
                            papers_count = author_info.get('paper_count', 0)
                            
                            log_or_print(f"  {i:2d}. {name}", logger)
                            log_or_print(f"      Domains: {domain_count}, Papers: {papers_count}", logger)
                            log_or_print(f"      Interdisciplinary Score: {score}", logger)
                            if domains:
                                # Show domain names, limiting to readable length
                                domain_display = domains[:4] if len(domains) <= 4 else domains[:3] + [f"(+{len(domains)-3} more)"]
                                log_or_print(f"      Research Areas: {', '.join(domain_display)}", logger)
                            log_or_print("", logger)
                        else:
                            log_or_print(f"  {i}. {author_info}", logger)
                    log_or_print("", logger)
                
                # Show analysis statistics
                total_authors = cross_domain_data.get('total_authors_analyzed', 0)
                # Use the actual length of cross_domain_authors (top interdisciplinary researchers) not the total count
                actual_interdisciplinary_count = len(cross_domain_authors)
                # The 'interdisciplinary_authors' key contains the total count of all authors with multiple domains
                total_interdisciplinary_count = cross_domain_data.get('interdisciplinary_authors', 0)
                single_domain_count = cross_domain_data.get('single_domain_authors', 0)
                avg_domains = cross_domain_data.get('average_domains_per_author', 0)
                
                if total_authors > 0:
                    log_or_print("Analysis Summary:", logger)
                    log_or_print(f"   • Total Authors Analyzed: {total_authors:,}", logger)
                    log_or_print(f"   • Top Interdisciplinary Researchers Shown: {actual_interdisciplinary_count}", logger)
                    
                    # Only show the large total count if it's different and meaningful
                    if total_interdisciplinary_count > actual_interdisciplinary_count:
                        # Clarify what this large number represents
                        log_or_print(f"   • Authors Working in Multiple Domains: {total_interdisciplinary_count:,}", logger)
                        log_or_print(f"     (Note: This includes all authors with papers in 2+ domains)", logger)
                    
                    # Only show single domain count if it's meaningful (should be 0 for current analysis)
                    if single_domain_count > 0:
                        log_or_print(f"   • Single-Domain Researchers: {single_domain_count}", logger)
                    
                    log_or_print(f"   • Average Domains per Author: {avg_domains}", logger)
                    
                    # Calculate meaningful percentages - but only show if the rate is reasonable
                    if total_interdisciplinary_count < total_authors:  # If not 100%
                        interdisciplinary_pct = (total_interdisciplinary_count / total_authors) * 100
                        log_or_print(f"   • Multi-Domain Research Rate: {interdisciplinary_pct:.1f}%", logger)
                    
                    log_or_print("", logger)
                
                # Show top domains
                top_domains = cross_domain_data.get('top_domains', [])
                if top_domains:
                    log_or_print("Most Active Research Domains:", logger)
                    for i, (domain, count) in enumerate(top_domains[:8], 1):  # Show top 8 domains
                        log_or_print(f"  {i}. {domain}: {count} papers", logger)
                    log_or_print("", logger)
                
                # Show domain distribution
                domain_dist = cross_domain_data.get('domain_distribution', {})
                if domain_dist:
                    log_or_print("Domain Coverage Distribution:", logger)
                    for domain_count in sorted(domain_dist.keys()):
                        author_count = domain_dist[domain_count]
                        log_or_print(f"   • {domain_count} domains: {author_count} researchers", logger)
                    log_or_print("", logger)
                
                # Show synthesis insights from LLM if available
                llm_insights = response.data.get('llm_insights', '')
                synthesis_insights = response.data.get('synthesis_insights', '')
                
                insights_text = llm_insights or synthesis_insights
                if insights_text and isinstance(insights_text, str) and insights_text.strip():
                    log_or_print("Research Synthesis & Insights:", logger)
                    # Format synthesis nicely
                    insights_lines = insights_text.split('\n')
                    for line in insights_lines:
                        line = line.strip()
                        if line:
                            if line.startswith('# '):  # Handle main headers
                                header_text = line.replace('# ', '').strip()
                                log_or_print("", logger)
                                log_or_print(f"{header_text}:", logger)
                            elif line.startswith('## '):  # Handle sub headers
                                header_text = line.replace('## ', '').strip()
                                log_or_print("", logger)
                                log_or_print(f"{header_text}:", logger)
                            elif line.startswith('### '):  # Handle trip headers
                                header_text = line.replace('### ', '').strip()
                                log_or_print("", logger)
                                log_or_print(f"{header_text}:", logger)
                            elif line.startswith('**') and line.endswith('**'):  # Handle bold text
                                log_or_print(f"   {line.replace('**', '').strip()}", logger)
                            elif line.startswith('- ') or line.startswith('• '):  # Handle bullet points
                                log_or_print(f"   {line}", logger)
                            elif line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):  # Handle numbered lists
                                log_or_print(f"   {line}", logger)
                            else:
                                log_or_print(f"   {line}", logger)
                    log_or_print("", logger)
                
                # Show opportunities if available
                opportunities = response.data.get('interdisciplinary_opportunities', [])
                if opportunities:
                    log_or_print("Interdisciplinary Research Opportunities:", logger)
                    for i, opportunity in enumerate(opportunities[:8], 1):  # Show top 8
                        if isinstance(opportunity, str):
                            log_or_print(f"   • {opportunity}", logger)
                        else:
                            log_or_print(f"   • {opportunity}", logger)
                    log_or_print("", logger)
            else:
                # Fallback: show raw data structure for debugging
                available_keys = list(response.data.keys()) if response.data else "None"
                log_or_print(f"Available data keys: {available_keys}", logger)
                for key, value in response.data.items():
                    if isinstance(value, list) and value:
                        log_or_print(f"", logger)
                        log_or_print(f"{key} (showing first few items):", logger)
                        for i, item in enumerate(value[:5], 1):
                            log_or_print(f"  {i}. {item}", logger)







