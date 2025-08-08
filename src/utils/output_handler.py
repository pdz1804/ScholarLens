"""
Output handling utilities for TechAuthor system.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .logger import setup_logger


def save_results_json(results: List[object], output_file: str, logger) -> None:
    """Save results to JSON file."""
    logger.info(f"Saving results to JSON: {output_file}")
    
    try:
        output_data = []
        
        for response in results:
            result_dict = {
                "query": response.query.text,
                "type": response.response_type,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "timestamp": datetime.now().isoformat(),
                "summary": response.summary,
                "insights": response.insights,
                "data": response.data,
                "agent_chain": [agent.__class__.__name__ for agent in response.agent_chain]
            }
            output_data.append(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(results)} results to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON results: {str(e)}")
        raise


def save_results_csv(results: List[object], output_file: str, logger) -> None:
    """Save results to CSV file."""
    logger.info(f"Saving results to CSV: {output_file}")
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Query', 'Type', 'Confidence', 'Processing_Time', 
                'Summary', 'Insights', 'Top_Authors', 'Top_Papers'
            ])
            
            # Write data
            for response in results:
                insights_str = '; '.join(response.insights) if response.insights else ''
                
                # Extract top authors/papers from data
                top_authors = ''
                top_papers = ''
                
                if response.data:
                    if 'authors' in response.data and response.data['authors']:
                        authors_list = response.data['authors'][:3]  # Top 3
                        if authors_list:
                            if isinstance(authors_list[0], dict):
                                top_authors = '; '.join([a.get('name', str(a)) for a in authors_list])
                            else:
                                top_authors = '; '.join([str(a) for a in authors_list])
                    
                    if 'papers' in response.data and response.data['papers']:
                        papers_list = response.data['papers'][:3]  # Top 3
                        if papers_list:
                            if isinstance(papers_list[0], dict):
                                top_papers = '; '.join([p.get('title', str(p)) for p in papers_list])
                            else:
                                top_papers = '; '.join([str(p) for p in papers_list])
                
                writer.writerow([
                    response.query.text,
                    response.response_type,
                    response.confidence,
                    response.processing_time,
                    response.summary,
                    insights_str,
                    top_authors,
                    top_papers
                ])
        
        logger.info(f"Successfully saved {len(results)} results to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save CSV results: {str(e)}")
        raise


def print_summary_stats(results: List[object], logger) -> None:
    """Print summary statistics of results."""
    if not results:
        logger.info("No results to summarize.")
        return
        
    total_queries = len(results)
    successful_queries = sum(1 for r in results if r.response_type != "error")
    avg_confidence = sum(r.confidence for r in results) / total_queries
    avg_processing_time = sum(r.processing_time for r in results) / total_queries
    
    # Count response types
    type_counts = {}
    for result in results:
        type_counts[result.response_type] = type_counts.get(result.response_type, 0) + 1
    
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total Queries: {total_queries}")
    logger.info(f"Successful: {successful_queries}")
    logger.info(f"Failed: {total_queries - successful_queries}")
    logger.info(f"Success Rate: {(successful_queries/total_queries)*100:.1f}%")
    logger.info(f"Average Confidence: {avg_confidence:.2f}")
    logger.info(f"Average Processing Time: {avg_processing_time:.2f}s")
    
    logger.info("Response Types:")
    for response_type, count in sorted(type_counts.items()):
        logger.info(f"  {response_type}: {count}")
    
    logger.info(f"Summary: {total_queries} queries, {successful_queries} successful, "
               f"avg confidence: {avg_confidence:.2f}, avg time: {avg_processing_time:.2f}s")


def generate_report(results: List[object], report_file: str, logger) -> None:
    """Generate a detailed text report."""
    logger.info(f"Generating detailed report: {report_file}")
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("TechAuthor Analysis Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Queries: {len(results)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\nQuery {i}/{len(results)}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Query: {result.query.text}\n")
                f.write(f"Type: {result.response_type}\n")
                f.write(f"Confidence: {result.confidence:.2f}\n")
                f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"\nSummary:\n{result.summary}\n")
                
                if result.insights:
                    f.write(f"\nKey Insights:\n")
                    for insight in result.insights:
                        f.write(f"  - {insight}\n")
                
                if result.data:
                    f.write(f"\nData Summary:\n")
                    for key, value in result.data.items():
                        if isinstance(value, list) and value:
                            f.write(f"  {key}: {len(value)} items\n")
                            # Show first few items
                            for j, item in enumerate(value[:3]):
                                if isinstance(item, dict):
                                    item_name = item.get('name') or item.get('title') or str(item)
                                else:
                                    item_name = str(item)
                                f.write(f"    {j+1}. {item_name}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                
                f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Report generated successfully: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        raise


def export_data_for_visualization(results: List[object], export_dir: str, logger) -> None:
    """Export data in formats suitable for visualization tools."""
    logger.info(f"Exporting visualization data to: {export_dir}")
    
    try:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Extract all authors mentioned across queries
        all_authors = {}
        all_subjects = {}
        collaboration_pairs = []
        
        for result in results:
            if result.data and 'authors' in result.data:
                for author_data in result.data['authors']:
                    if isinstance(author_data, dict):
                        name = author_data.get('name', '')
                        count = author_data.get('count', 0)
                        subjects = author_data.get('subjects', [])
                        
                        all_authors[name] = all_authors.get(name, 0) + count
                        
                        for subject in subjects:
                            all_subjects[subject] = all_subjects.get(subject, 0) + 1
            
            # Extract collaboration data if available
            if result.data and 'collaborations' in result.data:
                collaboration_pairs.extend(result.data['collaborations'])
        
        # Save author network data
        if all_authors:
            authors_file = export_path / "authors_network.json"
            with open(authors_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "nodes": [{"id": name, "papers": count} for name, count in all_authors.items()],
                    "links": collaboration_pairs
                }, f, indent=2, ensure_ascii=False)
        
        # Save subject distribution
        if all_subjects:
            subjects_file = export_path / "subject_distribution.json"
            with open(subjects_file, 'w', encoding='utf-8') as f:
                json.dump(all_subjects, f, indent=2, ensure_ascii=False)
        
        logger.info("Visualization data exported successfully")
        
    except Exception as e:
        logger.error(f"Failed to export visualization data: {str(e)}")
        raise
