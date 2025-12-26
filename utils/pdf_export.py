from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os

def export_chat_to_pdf(messages, role):
    os.makedirs("exports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"exports/chat_{timestamp}.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(
        f"<b>Enterprise RAG Chat Export</b><br/>Role: {role}<br/><br/>",
        styles["Title"]
    ))

    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("\n", "<br/>")

        story.append(Paragraph(
            f"<b>{role_label}:</b><br/>{content}<br/><br/>",
            styles["Normal"]
        ))

    doc.build(story)

    return file_path
