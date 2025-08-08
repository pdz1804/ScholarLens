#!/usr/bin/env python3
"""
Quick test to verify centralized logging is working.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import configure_logger, get_logger
from src.agents.analysis_agent import AnalysisAgent

def test_centralized_logging():
    print("=== Testing Centralized Logging ===")
    
    # Configure the global logger (simulating CLI setup)
    logger = configure_logger(level="INFO", log_file="test_logging.log")
    
    print(f"\n1. Testing main logger:")
    logger.info("Main logger test message")
    logger.warning("Main logger warning message")
    
    print(f"\n2. Testing get_logger() function:")
    test_logger = get_logger()
    test_logger.info("get_logger() test message")
    
    print(f"\n3. Testing agent logging:")
    # Create an analysis agent
    agent = AnalysisAgent()
    
    # Test the new logging methods
    agent._log_info("Analysis agent info message")
    agent._log_warning("Analysis agent warning message")
    agent._log_error("Analysis agent error message")
    agent._log_debug("Analysis agent debug message (might not show if level is INFO)")
    
    print(f"\n4. Verifying all use the same logger instance:")
    print(f"Main logger ID: {id(logger)}")
    print(f"get_logger() ID: {id(test_logger)}")  
    print(f"Agent logger ID: {id(agent.logger)}")
    
    # Check if they're the same instance
    if id(logger) == id(test_logger) == id(agent.logger):
        print("✓ All loggers are using the same instance!")
    else:
        print("✗ Loggers are different instances")
    
    print(f"\n5. Check log file:")
    if os.path.exists("test_logging.log"):
        with open("test_logging.log", "r") as f:
            log_content = f.read()
            line_count = len(log_content.strip().split('\n')) if log_content.strip() else 0
            print(f"Log file created with {line_count} lines")
            print("Last few log entries:")
            lines = log_content.strip().split('\n')
            for line in lines[-3:]:  # Show last 3 lines
                print(f"  {line}")
    else:
        print("Log file not created")

if __name__ == "__main__":
    test_centralized_logging()
