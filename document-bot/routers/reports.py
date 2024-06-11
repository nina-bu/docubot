import io
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from fpdf import FPDF
from pydantic import BaseModel
from routers.documents import find_documents_by_project, summarize_document
from routers.projects import hybrid_search

router = APIRouter()

class Project(BaseModel):
    id: int
    name: str
    description: str
    budget: int
    type: str

class Document(BaseModel):
    project_id: int
    document_id: int
    version: str
    name: str
    summary: Optional[str]

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Projects Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_project_section(self, index, project):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, f"{index}. {project.name} [{project.type}]", 0, 1)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, f"{project.description}", 0, 1)
        self.multi_cell(0, 5, f"Budget: {project.budget} EUR", 0, 1)
        self.ln(7.5)

    def add_document_section(self, document):
        self.set_left_margin(20)
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, f"{document.name} [V{document.version}]", 0, 1)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, f"Summary: {document.summary}", 0, 1)
        self.set_left_margin(10)
        self.ln(5)

    def add_paragraph(self, content):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, content)
        self.ln(5)

    def add_title(self, content):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 5, content)
        self.ln(0)

@router.get("/api/v1/reports")
async def report(search_term: str = Query(..., description="Term to search projects")):
    pdf_buffer = io.BytesIO(create_pdf(search_term))

    return StreamingResponse(pdf_buffer, 
                             media_type='application/pdf', 
                             headers={"Content-Disposition": "attachment; filename=report.pdf"})

def create_pdf(search_term):
    pdf = PDF()
    pdf.add_page()

    intro_paragraph = (
        f"The following document contains top-rated projects related to the search term "
        f"'{search_term}'. These projects have been selected based on their "
        "relevance, which is calculated not only based on their names but more significantly "
        "on their descriptions."
    )    

    pdf.add_paragraph(intro_paragraph)
    generate_content(pdf, search_term)

    return pdf.output(dest='S').encode('latin-1')

def generate_content(pdf, search_term):    
    # FEAT REPORT SECTION 1 (COMPLEX)
    projects = hybrid_search(search_term)[0]
    for index, result in enumerate(projects, start=1):
        project = Project(
            id=result.id,
            name=result.entity.get('name'),
            description=result.entity.get('description'),
            budget=result.entity.get('budget'),
            type=result.entity.get('type')
        )
        pdf.add_project_section(index, project)

        pdf.add_title('Documentation:')
        # FEAT REPORT SECTION 2 (SIMPLE)
        documents = find_documents_by_project(project.id)
        for result in documents:
            document = Document(
                project_id=result.get('project_id'),
                document_id=result.get('document_id'),
                version=result.get('version'),
                name=result.get('name')
            )

            # FEAT REPORT SECTION 3 (SIMPLE)
            document.summary = summarize_document(document.project_id, 
                                                     document.document_id, 
                                                     document.version)
            pdf.add_document_section(document)