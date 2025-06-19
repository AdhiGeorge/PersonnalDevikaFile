from fpdf import FPDF
import os
from datetime import datetime

class PDFGenerator:
    def __init__(self, title: str):
        self.pdf = FPDF()
        self.title = title
        self.setup_pdf()
        
    def setup_pdf(self):
        """Setup the PDF with basic formatting."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, self.title, ln=True, align="C")
        self.pdf.set_font("Arial", "", 12)
        
    def add_section(self, title: str, content: str):
        """Add a section to the PDF."""
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.set_font("Arial", "", 12)
        self.pdf.multi_cell(0, 10, content)
        
    def add_code(self, code: str, language: str = "python"):
        """Add code to the PDF."""
        self.pdf.set_font("Courier", "", 10)
        self.pdf.multi_cell(0, 10, code)
        self.pdf.set_font("Arial", "", 12)
        
    def save(self, filename: str = None):
        """Save the PDF to a file."""
        if filename is None:
            filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
        # Ensure the PDFs directory exists
        pdfs_dir = os.path.join(os.getcwd(), "data", "pdfs")
        os.makedirs(pdfs_dir, exist_ok=True)
        
        # Save the PDF
        filepath = os.path.join(pdfs_dir, filename)
        self.pdf.output(filepath)
        return filepath 