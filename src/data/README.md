# Data Directory

This directory contains the data management modules for the TechAuthor system.

## Structure

- `data_manager.py` - Main data management class that handles loading, processing, and indexing
- `processed_data_manager.py` - Handles caching and storage of processed data

## Features

- **LLM-based Author Parsing**: Automatically parses author names and institutions using LLM
- **Comprehensive Author Statistics**: Generates detailed author profiles with institutions, career metrics, and collaboration patterns  
- **Incremental Processing**: Caches processed data to avoid reprocessing
- **Multi-format Dataset Support**: Supports both single-file and multi-file dataset formats

## Author Statistics

The system automatically generates `data/processed/authors/author_stats.json` containing:

- Basic metrics: total papers, years active, domains, subjects
- Institution affiliations from LLM parsing
- Career span and productivity analysis
- Research breadth and domain diversity
- Collaboration intensity classification
- Primary subject rankings
- Peak productivity periods

This file is used by AUTHOR_STATS queries like "Tell me about John Smolin's research profile".
