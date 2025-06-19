import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

logger = logging.getLogger(__name__)

async def start_interaction(url: str) -> tuple:
    """Start browser interaction with a URL.
    
    Args:
        url: The URL to interact with
        
    Returns:
        tuple: (browser instance, raw content, processed content)
    """
    browser = Browser()
    try:
        browser.start()
        success = browser.go_to(url)
        if not success:
            return browser, None, None
            
        raw_content = browser.extract_text()
        processed_content = raw_content.strip() if raw_content else None
        
        return browser, raw_content, processed_content
    except Exception as e:
        logger.error(f"Error in browser interaction: {str(e)}")
        browser.close()
        return browser, None, None

class Browser:
    def __init__(self):
        self.driver = None
        self._started = False

    def start(self):
        """Start the browser."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)  # 30 seconds timeout
            self._started = True
            logger.info("Browser started successfully")
        except Exception as e:
            logger.error(f"Error starting browser: {str(e)}")
            raise

    def go_to(self, url: str) -> bool:
        """Navigate to a URL."""
        try:
            if not self._started:
                raise Exception("Browser not initialized. Call start() first.")
            self.driver.get(url)
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return True
        except Exception as e:
            logger.error(f"Error navigating to {url}: {str(e)}")
            return False

    def extract_text(self) -> str:
        """Extract text content from the current page."""
        try:
            if not self._started:
                raise Exception("Browser not initialized. Call start() first.")
            # Wait for body to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Get the text content
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

    def close(self):
        """Close the browser and clean up resources."""
        try:
            if self._started and self.driver:
                self.driver.quit()
                self._started = False
                logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            raise
