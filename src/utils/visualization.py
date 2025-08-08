"""
Visualization utilities for TechAuthor system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
import re
from datetime import datetime
from pathlib import Path


class TechnologyTrendVisualizer:
    """
    Provides utilities for visualizing technology trends analysis results, including
    temporal trends and current technology distributions. This class generates and saves
    plots summarizing technology trends based on analysis data.

    Main Methods:
        - sanitize_filename(query_text): Sanitize a string to create a valid filename.
        - create_trend_visualization(trends_data, query_text, analysis_type): Generate and save
          a visualization image for the given trends data.

    Usage Example:
        >>> visualizer = TechnologyTrendVisualizer(output_dir="output/visuals")
        >>> img_path = visualizer.create_trend_visualization(
        ...     trends_data=analysis_results,
        ...     query_text="AI in Healthcare",
        ...     analysis_type="temporal_trends"
        ... )
        >>> print(f"Visualization saved to: {img_path}")

    Args:
        output_dir (str): Directory to save visualization images.
    """
    
    def __init__(self, output_dir: str = "data/output/visualize"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def sanitize_filename(self, query_text: str) -> str:
        """
        Sanitize query text to create a valid filename.
        
        Args:
            query_text: Original query text
            
        Returns:
            Sanitized filename
        """
        # Remove special characters and replace spaces with underscores
        sanitized = re.sub(r'[^\w\s-]', '', query_text)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        # Limit length to avoid very long filenames
        sanitized = sanitized[:50]
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{sanitized}_{timestamp}"
    
    def create_trend_visualization(
        self,
        trends_data: Dict[str, Any],
        query_text: str,
        analysis_type: str = "temporal_trends"
    ) -> str:
        """
        Create comprehensive visualization for technology trends.
        
        Args:
            trends_data: Technology trends analysis results
            query_text: Original query text for filename
            analysis_type: Type of analysis (temporal_trends or current_technologies)
            
        Returns:
            Path to saved visualization image
        """
        # Create filename
        base_filename = self.sanitize_filename(query_text)
        filepath = self.output_dir / f"tech_trends_{base_filename}.png"
        
        # Extract data
        trends = trends_data.get("trends", [])
        time_series = trends_data.get("time_series_data", {})
        emerging = trends_data.get("emerging_technologies", [])
        declining = trends_data.get("declining_technologies", [])
        analysis_period = trends_data.get("analysis_period", "Unknown")
        total_papers = trends_data.get("total_papers", 0)
        
        if analysis_type == "temporal_trends" and time_series.get("years"):
            return self._create_temporal_trends_plot(
                trends, time_series, emerging, declining, 
                analysis_period, total_papers, query_text, filepath
            )
        else:
            return self._create_current_technologies_plot(
                trends, analysis_period, total_papers, query_text, filepath
            )
    
    def _create_temporal_trends_plot(
        self,
        trends: List[Dict],
        time_series: Dict,
        emerging: List[str],
        declining: List[str],
        analysis_period: str,
        total_papers: int,
        query_text: str,
        filepath: Path
    ) -> str:
        """Create multi-panel visualization for temporal trends."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Time Series Plot (top, spanning both columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_time_series(ax1, time_series, query_text)
        
        # 2. Trend Slopes Bar Chart (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_trend_slopes(ax2, trends[:10])
        
        # 3. Emerging vs Declining (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_emerging_declining(ax3, emerging, declining)
        
        # 4. Technology Popularity (third row, left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_technology_popularity(ax4, trends[:10])
        
        # 5. Analysis Summary (third row, right)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_analysis_summary(ax5, trends, analysis_period, total_papers)
        
        # Overall title
        fig.suptitle(f'Technology Trends Analysis\nQuery: "{query_text}"', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def _create_current_technologies_plot(
        self,
        trends: List[Dict],
        analysis_period: str,
        total_papers: int,
        query_text: str,
        filepath: Path
    ) -> str:
        """Create visualization for current technology popularity (single-year data)."""
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Current Technology Landscape\nQuery: "{query_text}"', 
                     fontsize=16, fontweight='bold')
        
        # 1. Technology Popularity Bar Chart
        if trends:
            top_10 = trends[:10]
            technologies = [t["technology"] for t in top_10]
            paper_counts = [t["total_papers"] for t in top_10]
            
            bars = ax1.bar(range(len(technologies)), paper_counts, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(technologies))))
            ax1.set_xlabel('Technologies')
            ax1.set_ylabel('Number of Papers')
            ax1.set_title('Top Technologies by Paper Count')
            ax1.set_xticks(range(len(technologies)))
            ax1.set_xticklabels(technologies, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, paper_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontsize=9)
        
        # 2. Technology Relevance Pie Chart
        if len(trends) >= 5:
            top_5 = trends[:5]
            labels = [t["technology"] for t in top_5]
            sizes = [t["total_papers"] for t in top_5]
            
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Top 5 Technologies Distribution')
        
        # 3. Relevance Score Analysis
        if trends and all("relevance_score" in t for t in trends[:10]):
            technologies = [t["technology"] for t in trends[:10]]
            relevance_scores = [t["relevance_score"] for t in trends[:10]]
            
            bars = ax3.barh(range(len(technologies)), relevance_scores,
                           color=plt.cm.plasma(np.linspace(0, 1, len(technologies))))
            ax3.set_yticks(range(len(technologies)))
            ax3.set_yticklabels(technologies)
            ax3.set_xlabel('Relevance Score')
            ax3.set_title('Technology Relevance Scores')
            ax3.invert_yaxis()
        
        # 4. Analysis Summary
        self._plot_analysis_summary(ax4, trends, analysis_period, total_papers)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def _plot_time_series(self, ax, time_series: Dict, query_text: str):
        """Plot time series data for top technologies."""
        years = time_series.get("years", [])
        subject_counts = time_series.get("subject_counts", {})
        
        if not years or not subject_counts:
            ax.text(0.5, 0.5, 'No time series data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Technology Trends Over Time')
            return
        
        # Plot lines for each technology
        colors = plt.cm.Set3(np.linspace(0, 1, len(subject_counts)))
        
        for i, (technology, counts) in enumerate(subject_counts.items()):
            if len(counts) == len(years):
                ax.plot(years, counts, marker='o', linewidth=2, 
                       label=technology[:30], color=colors[i])
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Papers')
        ax.set_title('Technology Trends Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set integer ticks for years
        if years:
            ax.set_xticks(years)
    
    def _plot_trend_slopes(self, ax, trends: List[Dict]):
        """Plot trend slopes as horizontal bar chart."""
        if not trends:
            ax.text(0.5, 0.5, 'No trend data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trend Slopes')
            return
        
        technologies = [t["technology"][:20] for t in trends]
        slopes = [t.get("trend_slope", 0) for t in trends]
        
        # Color code: green for positive, red for negative, gray for stable
        colors = ['green' if s > 0.1 else 'red' if s < -0.1 else 'gray' for s in slopes]
        
        bars = ax.barh(range(len(technologies)), slopes, color=colors, alpha=0.7)
        ax.set_yticks(range(len(technologies)))
        ax.set_yticklabels(technologies)
        ax.set_xlabel('Trend Slope (papers/year)')
        ax.set_title('Technology Trend Slopes')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.invert_yaxis()
        
        # Add value labels
        for bar, slope in zip(bars, slopes):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2.,
                   f'{slope:.2f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
    
    def _plot_emerging_declining(self, ax, emerging: List[str], declining: List[str]):
        """Plot emerging vs declining technologies."""
        categories = []
        technologies = []
        
        for tech in emerging[:5]:
            categories.append('Emerging')
            technologies.append(tech[:20])
        
        for tech in declining[:5]:
            categories.append('Declining')
            technologies.append(tech[:20])
        
        if not technologies:
            ax.text(0.5, 0.5, 'No emerging/declining\ntechnologies identified', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Emerging vs Declining Technologies')
            return
        
        # Create categorical plot
        df = pd.DataFrame({'Technology': technologies, 'Category': categories})
        category_counts = df['Category'].value_counts()
        
        colors = ['lightgreen', 'lightcoral']
        wedges, texts, autotexts = ax.pie(category_counts.values, 
                                         labels=category_counts.index,
                                         colors=colors, autopct='%1.0f',
                                         startangle=90)
        ax.set_title('Emerging vs Declining\nTechnologies')
    
    def _plot_technology_popularity(self, ax, trends: List[Dict]):
        """Plot technology popularity scatter plot."""
        if not trends:
            ax.text(0.5, 0.5, 'No trend data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Technology Popularity')
            return
        
        total_papers = [t.get("total_papers", 0) for t in trends]
        recent_papers = [t.get("recent_papers", 0) for t in trends]
        technologies = [t["technology"][:15] for t in trends]
        
        scatter = ax.scatter(total_papers, recent_papers, 
                           s=100, alpha=0.6, c=range(len(trends)), 
                           cmap='viridis')
        
        # Add labels for top technologies
        for i, (x, y, tech) in enumerate(zip(total_papers[:5], recent_papers[:5], technologies[:5])):
            ax.annotate(tech, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Total Papers')
        ax.set_ylabel('Recent Papers')
        ax.set_title('Technology Popularity\n(Total vs Recent)')
        ax.grid(True, alpha=0.3)
    
    def _plot_analysis_summary(self, ax, trends: List[Dict], analysis_period: str, total_papers: int):
        """Plot analysis summary statistics."""
        ax.axis('off')
        
        # Calculate summary statistics
        if trends:
            avg_papers = np.mean([t.get("total_papers", 0) for t in trends])
            max_papers = max([t.get("total_papers", 0) for t in trends])
            num_technologies = len(trends)
            
            # Count trend directions
            increasing = len([t for t in trends if t.get("trend_direction") == "increasing"])
            decreasing = len([t for t in trends if t.get("trend_direction") == "decreasing"])
            stable = len([t for t in trends if t.get("trend_direction") == "stable"])
            
            summary_text = f"""
Analysis Summary

Period: {analysis_period}
Total Papers: {total_papers}
Technologies Analyzed: {num_technologies}

Trend Directions:
• Increasing: {increasing}
• Decreasing: {decreasing}
• Stable: {stable}

Statistics:
• Avg Papers/Tech: {avg_papers:.1f}
• Max Papers: {max_papers}
            """
        else:
            summary_text = f"""
Analysis Summary

Period: {analysis_period}
Total Papers: {total_papers}
Technologies: No data available
            """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    def display_image(self, image_path: str) -> None:
        """
        Display the saved image (for Jupyter notebooks or IDE with image display).
        
        Args:
            image_path: Path to the saved image
        """
        try:
            from IPython.display import Image, display
            display(Image(image_path))
        except ImportError:
            # In non-Jupyter environment, just print the path
            print(f"Visualization saved to: {image_path}")
            print("Open the file to view the generated chart.")


def create_technology_trends_visualization(
    trends_data: Dict[str, Any],
    query_text: str,
    output_dir: str = "data/output/visualize"
) -> str:
    """
    Convenience function to create technology trends visualization.
    
    Args:
        trends_data: Technology trends analysis results
        query_text: Original query text for filename
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization image
    """
    visualizer = TechnologyTrendVisualizer(output_dir)
    analysis_type = trends_data.get("analysis_type", "temporal_trends")
    
    image_path = visualizer.create_trend_visualization(
        trends_data, query_text, analysis_type
    )
    
    # Display the image
    visualizer.display_image(image_path)
    
    return image_path
