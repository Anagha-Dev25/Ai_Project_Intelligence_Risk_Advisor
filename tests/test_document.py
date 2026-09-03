from app.models.document import Document


document = Document(
    content="Authentication is incomplete.",
    source="meeting_notes.docx",
    file_type="docx"
)

print("----- DOCUMENT MODEL -----")
print("Content:", document.content)
print("Source:", document.source)
print("File Type:", document.file_type)
print("Metadata:", document.metadata)