import json
import csv
import io
from typing import List, Dict, Any
import PyPDF2

def process_file(content: bytes, file_type: str) -> List[Dict[str, Any]]:
    """
    Process a file based on its type and extract documents
    
    Args:
        content: The file content as bytes
        file_type: The type of file ('json', 'csv', or 'pdf')
        
    Returns:
        A list of document dictionaries
    """
    if file_type == 'json':
        return process_json(content)
    elif file_type == 'csv':
        return process_csv(content)
    elif file_type == 'pdf':
        return process_pdf(content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def process_json(content: bytes) -> List[Dict[str, Any]]:
    """Process JSON file content"""
    try:
        # Parse JSON
        data = json.loads(content.decode('utf-8'))
        
        documents = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # If it's a list, process each item
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # If the item is a dictionary, convert it to text
                    text = json.dumps(item)
                    documents.append({
                        "id": f"json_{i}",
                        "text": text,
                        "metadata": {
                            "source": "json",
                            "index": i,
                            "original": item
                        }
                    })
                else:
                    # If the item is a primitive value, convert to string
                    documents.append({
                        "id": f"json_{i}",
                        "text": str(item),
                        "metadata": {
                            "source": "json",
                            "index": i,
                            "original": item
                        }
                    })
        elif isinstance(data, dict):
            # If it's a dictionary, process each key-value pair
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    # If the value is complex, convert to JSON string
                    text = json.dumps(value)
                else:
                    # If the value is a primitive, use as is
                    text = str(value)
                
                documents.append({
                    "id": f"json_{key}",
                    "text": text,
                    "metadata": {
                        "source": "json",
                        "key": key,
                        "original": value
                    }
                })
        else:
            # If it's a primitive value, just add it as a single document
            documents.append({
                "id": "json_0",
                "text": str(data),
                "metadata": {
                    "source": "json",
                    "original": data
                }
            })
        
        return documents
    
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")

def process_csv(content: bytes) -> List[Dict[str, Any]]:
    """Process CSV file content"""
    try:
        # Decode bytes to string
        text_content = content.decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.reader(io.StringIO(text_content))
        rows = list(csv_reader)
        
        if not rows:
            return []
        
        documents = []
        headers = rows[0]
        
        # Process each row
        for i, row in enumerate(rows[1:], 1):
            # Create a dictionary for the row
            row_dict = {headers[j]: value for j, value in enumerate(row) if j < len(headers)}
            
            # Convert row to text
            text = ", ".join([f"{header}: {row_dict.get(header, '')}" for header in headers])
            
            documents.append({
                "id": f"csv_{i}",
                "text": text,
                "metadata": {
                    "source": "csv",
                    "row": i,
                    "original": row_dict
                }
            })
        
        return documents
    
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")

def process_pdf(content: bytes) -> List[Dict[str, Any]]:
    """Process PDF file content"""
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        documents = []
        
        # Process each page
        for i, page in enumerate(pdf_reader.pages):
            # Extract text from page
            text = page.extract_text()
            
            if text.strip():  # Only add non-empty pages
                documents.append({
                    "id": f"pdf_page_{i+1}",
                    "text": text,
                    "metadata": {
                        "source": "pdf",
                        "page": i+1,
                        "total_pages": len(pdf_reader.pages)
                    }
                })
        
        return documents
    
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")
