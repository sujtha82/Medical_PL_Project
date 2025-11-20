import fitz  # PyMuPDF
import nltk
import pandas as pd
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

    def csv_to_nlp_chunks(self, csv_path):
        """Convert medical claims CSV to NLP-optimized text chunks."""
        try:
            df = pd.read_csv(csv_path)
            chunks = []
            
            print(f"📊 Converting {len(df)} medical claims to NLP-friendly text...")
            
            # Method 1: Individual claims in natural language
            for _, row in df.iterrows():
                # Natural language description
                claim_text = (
                    f"Medical claim {row['Claim ID']} involves patient {row['Patient']} "
                    f"who received {row['Procedure'].lower()} for the condition {row['Diagnosis']}. "
                    f"The service was provided by {row['Provider']} on {row['Date of Service']}. "
                    f"The total amount billed was ${row['Amount Billed ($)']} with ${row['Amount Paid ($)']} paid by insurance. "
                    f"This claim is currently {row['Claim Status'].lower()}."
                )
                chunks.append(claim_text)
            
            # Method 2: Contextual summaries for better semantic understanding
            chunks.extend(self._create_contextual_summaries(df))
            
            # Method 3: Question-answer style chunks
            chunks.extend(self._create_qa_chunks(df))
            
            print(f"✅ Created {len(chunks)} NLP-optimized chunks from CSV")
            return chunks
            
        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}")
            return []

    def _create_contextual_summaries(self, df):
        """Create contextual summaries for better semantic understanding."""
        summaries = []
        
        # Summary by claim status
        status_groups = df.groupby('Claim Status')
        for status, group in status_groups:
            summary = (
                f"There are {len(group)} claims with {status.lower()} status. "
                f"These include procedures like {', '.join(group['Procedure'].head(3).tolist())}. "
                f"The total amount billed for these {status.lower()} claims is ${group['Amount Billed ($)'].sum()} "
                f"with ${group['Amount Paid ($)'].sum()} paid out."
            )
            summaries.append(summary)
        
        # Summary by provider
        provider_groups = df.groupby('Provider')
        for provider, group in provider_groups:
            common_diagnoses = group['Diagnosis'].value_counts().head(2)
            summary = (
                f"{provider} has processed {len(group)} medical claims. "
                f"Common diagnoses treated include {', '.join(common_diagnoses.index.tolist())}. "
                f"The most frequent procedures are {group['Procedure'].mode().iloc[0] if not group['Procedure'].mode().empty else 'various services'}."
            )
            summaries.append(summary)
        
        return summaries

    def _create_qa_chunks(self, df):
        """Create question-answer style chunks for better retrieval."""
        qa_chunks = []
        
        for _, row in df.iterrows():
            # Multiple question formats for the same claim
            questions = [
                f"What is the status of claim {row['Claim ID']}? Answer: The claim is {row['Claim Status'].lower()}.",
                f"Who is the patient for claim {row['Claim ID']}? Answer: The patient is {row['Patient']}.",
                f"What procedure was performed for {row['Patient']}? Answer: {row['Procedure']} was performed.",
                f"How much was billed for {row['Patient']}'s {row['Procedure']}? Answer: ${row['Amount Billed ($)']} was billed.",
                f"When was {row['Patient']}'s service provided? Answer: On {row['Date of Service']}.",
                f"Which provider handled {row['Patient']}'s claim? Answer: {row['Provider']} handled the claim."
            ]
            qa_chunks.extend(questions)
        
        return qa_chunks

    def load_document(self, file_path):
        """Load document from PDF, TXT, or CSV file."""
        if not file_path:
            raise ValueError("File path cannot be empty!")
            
        if file_path.endswith(".pdf"):
            print(f"📄 Loading PDF: {file_path}")
            return "\n".join(self.page_level_chunking(file_path))
        elif file_path.endswith(".txt"):
            print(f"📄 Loading TXT: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
        elif file_path.endswith(".csv"):
            print(f"📄 Loading CSV with NLP conversion: {file_path}")
            chunks = self.csv_to_nlp_chunks(file_path)
            return "\n".join(chunks)
        else:
            raise ValueError("Unsupported file format! Use .pdf, .txt, or .csv")

    # chunking methods
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

"""import fitz  # PyMuPDF
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
        Extracts text per page from PDF
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
        Split text using recursive character splitting.
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
        Split text using sliding window approach.
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
        Split text based on approximate token count
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
        Split text based on semantic similarity between sentences.
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
        Load document from PDF or TXT file.
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
            raise ValueError("Unsupported file format! Use .pdf or .txt")"""