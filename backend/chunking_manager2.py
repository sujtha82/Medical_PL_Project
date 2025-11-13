import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer, util

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ChunkingManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.embed_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embed_model = None

    def page_level_chunking(self, pdf_path):
        """Extracts text per page from PDF."""
        chunks = []
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text")
                    if text.strip():
                        chunks.append(f"Page {page_number}: {text.strip()}")
        except Exception as e:
            print(f"Error reading PDF file: {e}")
        return chunks

    def recursive_chunking(self, text, chunk_size=500, overlap=100):
        """Split text using recursive character splitting."""
        if not text or not text.strip():
            return []
        
        separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            if end < text_length:
                for separator in separators:
                    if separator:
                        split_pos = text.rfind(separator, start, end)
                        if split_pos != -1:
                            end = split_pos + len(separator)
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end - overlap > start else end
            
        return chunks

    def sliding_window_chunking(self, text, window_size=300, overlap=100):
        """Split text using sliding window approach."""
        if not text or not text.strip():
            return []
            
        words = text.split()
        if len(words) <= window_size:
            return [" ".join(words)]
            
        chunks = []
        for i in range(0, len(words), window_size - overlap):
            chunk = " ".join(words[i:i + window_size])
            chunks.append(chunk)
        return chunks

    def token_based_chunking(self, text, tokens_per_chunk=256):
        """Split text based on approximate token count."""
        if not text or not text.strip():
            return []
        
        words = text.split()
        words_per_chunk = tokens_per_chunk * 4
        
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i:i + words_per_chunk])
            chunks.append(chunk)
        return chunks

    def semantic_chunking(self, text, similarity_threshold=0.6):
        """Split text based on semantic similarity between sentences."""
        if not text or not text.strip():
            return []
            
        if self.embed_model is None:
            print("Embedding model not available. Using recursive chunking as fallback.")
            return self.recursive_chunking(text)
        
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 1:
                return [" ".join(sentences)]
                
            embeddings = self.embed_model.encode(sentences, normalize_embeddings=True)
            chunks, current_chunk = [], [sentences[0]]

            for i in range(1, len(sentences)):
                sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
                if sim < similarity_threshold:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])

            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
            
        except Exception as e:
            print(f"Error in semantic chunking: {e}. Using recursive chunking as fallback.")
            return self.recursive_chunking(text)

    def load_document(self, file_path):
        """Load document from PDF or TXT file."""
        if not file_path:
            raise ValueError("File path cannot be empty!")
            
        if file_path.endswith(".pdf"):
            return "\n".join(self.page_level_chunking(file_path))
        elif file_path.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
        else:
            raise ValueError("Unsupported file format! Use .pdf or .txt")