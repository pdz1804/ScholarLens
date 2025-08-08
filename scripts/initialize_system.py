"""
System initialization script for TechAuthor.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root and src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

try:
    from src import create_system
    from src.core.config import config_manager
    from src.utils.logger import setup_logger
except ImportError:
    # Fallback import method
    try:
        from core.system import TechAuthorSystem
        from core.config import config_manager
        from utils.logger import setup_logger
        
        def create_system():
            return TechAuthorSystem(config_manager)
    except ImportError as e:
        # For initialization script, we'll use basic print since logger may not be available yet
        print(f"Failed to import required modules: {e}")
        print(f"Project root: {project_root}")
        print(f"Src path: {src_path}")
        print(f"Python path: {sys.path}")
        sys.exit(1)


async def initialize_system():
    """Initialize the TechAuthor system."""
    
    print("TechAuthor System Initialization")
    print("=" * 50)
    
    # Check environment
    print("1. Checking environment...")
    
    # Check if .env file exists
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print("   Warning: .env file not found. Creating from template...")
        env_template = Path(__file__).parent.parent / ".env.example"
        if env_template.exists():
            import shutil
            shutil.copy(env_template, env_file)
            print(f"   Created .env file. Please update it with your API keys.")
        else:
            print("   Error: .env.example not found!")
            return False
    
    # Validate configuration
    print("2. Validating configuration...")
    if not config_manager.validate():
        print("   Warning: Configuration validation failed!")
        print("   This is expected if you haven't set up your API keys yet.")
        print("   The system can still be initialized for testing purposes.")
        print("   Please update your .env file with:")
        print("   - OPENAI_API_KEY=your_openai_api_key_here")
        print("   - Other optional configuration as needed")
        print("   Continuing with initialization...")
    else:
        print("   Configuration validated successfully!")
    
    # Check for dataset
    print("3. Checking dataset...")
    # Check for new multi-file format
    data_dir = Path(__file__).parent.parent / "data"
    multi_files = [
        data_dir / "arxiv_cs_2021.csv",
        data_dir / "arxiv_cs_2022.csv", 
        data_dir / "arxiv_cs_2023.csv",
        data_dir / "arxiv_cs_2024.csv",
        data_dir / "arxiv_cs_2025.csv"
    ]
    
    found_files = [f for f in multi_files if f.exists()]
    
    if found_files:
        print(f"   Found {len(found_files)} dataset files:")
        for f in found_files:
            print(f"     - {f.name}")
    else:
        # Check for legacy single file
        legacy_dataset = data_dir / "arxiv_cs.csv"
        if legacy_dataset.exists():
            print(f"   Found legacy dataset: {legacy_dataset}")
        else:
            print("   Warning: No dataset files found!")
            print("   Please place your arxiv_cs_YYYY.csv files in the data/ directory.")
            print("   Expected files: arxiv_cs_2021.csv, arxiv_cs_2022.csv, etc.")
            print("   The system will work with limited functionality without the dataset.")
    
    # Create necessary directories
    print("4. Creating directories...")
    directories = [
        Path(__file__).parent.parent / "data" / "processed",
        Path(__file__).parent.parent / "data" / "embeddings", 
        Path(__file__).parent.parent / "data" / "chroma_db",
        Path(__file__).parent.parent / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Test system initialization
    print("5. Testing system initialization...")
    try:
        system = create_system()
        await system.initialize()
        print("   System initialized successfully!")
        
        # Test basic functionality
        print("6. Testing basic functionality...")
        health = await system.health_check()
        if health.get("status") == "healthy":
            print("   Health check passed!")
        else:
            print(f"   Health check warning: {health}")
        
        # Test a simple query if dataset is available
        if found_files:
            print("7. Testing query processing...")
            try:
                response = await system.aquery("test query", {"top_k": 1})
                if response and not response.result.get("error"):
                    print("   Query processing test passed!")
                else:
                    print("   Query processing test had issues (this is normal without proper dataset)")
            except Exception as e:
                print(f"   Query processing test failed: {e}")
        
        await system.shutdown()
        
    except Exception as e:
        print(f"   System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("TechAuthor System Initialization Complete!")
    print("=" * 50)
    
    print("\nNext Steps:")
    print("1. Update .env file with your OpenAI API key")
    print("2. Place arxiv_cs_YYYY.csv dataset files in data/ directory")
    print("   Expected: arxiv_cs_2021.csv, arxiv_cs_2022.csv, etc.")
    print("3. Run: python main.py 'your query here'")
    print("4. Or try: python examples/basic_usage.py")
    
    return True


async def main():
    """Main initialization function."""
    try:
        success = await initialize_system()
        if not success:
            print("\nInitialization failed. Please check the errors above.")
            sys.exit(1)
        else:
            print("\nInitialization completed successfully!")
    
    except KeyboardInterrupt:
        print("\nInitialization cancelled by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nUnexpected error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
