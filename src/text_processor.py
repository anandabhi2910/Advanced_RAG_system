import os
import re
import glob

def clean_text(text):
    """
    Cleans a string by removing HTML tags, non-alphabetic characters,
    and extra spaces, then converts it to lowercase.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text string.
    """
    # Remove HTML tags (similar to Project 1)
    text = re.sub(r'<.*?>', '', text)
    # Keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_and_preprocess_documents(directory_path):
    """
    Reads all text files from a directory, cleans the content, and
    returns a list of cleaned document strings.

    Args:
        directory_path (str): The path to the directory containing the text files.

    Returns:
        list: A list of preprocessed document strings.
    """
    cleaned_documents = []
    # Use glob to find all files ending with .txt or .md
    file_paths = glob.glob(os.path.join(directory_path, '*.txt')) + \
                 glob.glob(os.path.join(directory_path, '*.md'))
    
      # Add this line to see exactly what glob found
    print(f"Files found by glob: {file_paths}")

    if not file_paths:
        print(f"No text files found in the directory: {directory_path}")
        return []

    for file_path in file_paths:
        print(f"Reading and cleaning file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='latin-1') as file:
            raw_text = file.read()
            cleaned_text = clean_text(raw_text)
            cleaned_documents.append(cleaned_text)

    print(f"Successfully processed {len(cleaned_documents)} documents.")
    return cleaned_documents

if __name__ == "__main__":
    # --- Configuration ---
    # Get the path to the 'data/raw' directory relative to the current script
    RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

    # Read and preprocess the documents
    documents = read_and_preprocess_documents(RAW_DATA_PATH)

    # Display a sample of the processed output
    if documents:
        print("\n--- Sample of Processed Document ---")
        print(documents[0][:500]) # Print the first 500 characters of the first document
        print("...")
