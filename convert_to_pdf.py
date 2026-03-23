from fpdf import FPDF
import os

def text_to_pdf(txt_path, pdf_path, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Replace common problematic characters
            content = content.replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '--')
            # Encode as latin-1, ignoring others for PDF safety
            content = content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=content)
    
    pdf.output(pdf_path)
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    text_to_pdf("docs/abstract.txt", "docs/abstract.pdf", "Project Abstract")
    text_to_pdf("docs/problem_statement.txt", "docs/problem_statement.pdf", "Problem Statement")
