import asyncio
import base64
import os

from playwright.sync_api import sync_playwright, TimeoutError, Page
from playwright.async_api import async_playwright, TimeoutError
from markdownify import markdownify as md
from pdfminer.high_level import extract_text
from src.socket_instance import emit_agent
from src.config import Config
from src.state import AgentState


class Browser:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.agent = AgentState()
        self.config = Config()

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--disable-setuid-sandbox',
                '--no-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
            ]
        )
        self.page = await self.browser.new_page(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        # Set default timeout
        self.page.set_default_timeout(30000)  # 30 seconds
        
        # Enable JavaScript
        await self.page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        return self

    # def new_page(self):
    #     return self.browser.new_page()

    async def go_to(self, url):
        try:
            await self.page.goto(url, timeout=30000, wait_until='networkidle')
        except TimeoutError as e:
            print(f"TimeoutError: {e} when trying to navigate to {url}")
            return False
        return True

    async def screenshot(self, project_name):
        screenshots_save_path = self.config.get_screenshots_dir()

        page_metadata = await self.page.evaluate("() => { return { url: document.location.href, title: document.title } }")
        page_url = page_metadata['url']
        random_filename = os.urandom(20).hex()
        filename_to_save = f"{random_filename}.png"
        path_to_save = os.path.join(screenshots_save_path, filename_to_save)

        await self.page.emulate_media(media="screen")
        await self.page.screenshot(path=path_to_save, full_page=True)
        screenshot = await self.page.screenshot()
        screenshot_bytes = base64.b64encode(screenshot).decode()
        new_state = self.agent.new_state()
        new_state["internal_monologue"] = "Browsing the web right now..."
        new_state["browser_session"]["url"] = page_url
        new_state["browser_session"]["screenshot"] = path_to_save
        self.agent.add_to_current_state(project_name, new_state)
        # self.close()
        return path_to_save, screenshot_bytes

    def get_html(self):
        return self.page.content()

    def get_markdown(self):
        return md(self.page.content())

    def get_pdf(self):
        pdfs_save_path = self.config.get_pdfs_dir()

        page_metadata = self.page.evaluate("() => { return { url: document.location.href, title: document.title } }")
        filename_to_save = f"{page_metadata['title']}.pdf"
        save_path = os.path.join(pdfs_save_path, filename_to_save)

        self.page.pdf(path=save_path)

        return save_path

    def pdf_to_text(self, pdf_path):
        return extract_text(pdf_path).strip()

    def get_content(self):
        pdf_path = self.get_pdf()
        return self.pdf_to_text(pdf_path)

    async def extract_text(self):
        return await self.page.evaluate("() => document.body.innerText")

    async def close(self):
        await self.page.close()
        await self.browser.close()
