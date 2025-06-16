import logging
from src.browser.search import SearchEngine
from src.browser.browser import Browser
from src.config import Config
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_and_scrape():
    """Test the search and scraping functionality."""
    browser = None
    try:
        # Initialize browser and search engine
        config = Config()
        browser = Browser()
        browser.start()  # Initialize the browser
        search_engine = SearchEngine(config)
        
        # Test queries
        test_queries = [
            "Python programming language",
            "Machine learning basics",
            "Web development tutorial"
        ]
        
        # Test each query
        for query in test_queries:
            print(f"\nTesting query: {query}")
            
            # Perform search
            results = search_engine.search(query, max_results=3)
            print(f"Found {len(results)} results")
            
            # Scrape each result
            for i, result in enumerate(results, 1):
                print(f"\nScraping result {i}: {result['title']}")
                print(f"URL: {result['link']}")
                
                # Navigate to the page
                content = browser.go_to(result['link'])
                if content:
                    # Extract text
                    text = browser.extract_text()
                    print(f"Extracted {len(text)} characters of text")
                else:
                    print("Failed to load page")
                
                # Small delay between requests
                time.sleep(2)
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
    finally:
        # Clean up
        if browser:
            browser.close()

if __name__ == "__main__":
    test_search_and_scrape()