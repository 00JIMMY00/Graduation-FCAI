from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import PyPDF2

def download_file(file_id):
    # Authenticate and download file
    pass

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
