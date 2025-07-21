import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any

class FAISSVectorDB:
    def __init__(self, dimension=1536, index_file="vector_index.faiss", data_file="vector_data.pkl"):
        """
        Initialize the FAISS vector database.
        
        Args:
            dimension: Dimension of the vectors (1536 for OpenAI embeddings)
            index_file: File to save/load the FAISS index
            data_file: File to save/load the document data
        """
        self.dimension = dimension
        self.index_file = index_file
        self.data_file = data_file
        self.documents = []
        
        # Try to load existing index and data
        if os.path.exists(index_file) and os.path.exists(data_file):
            try:
                print(f"[INFO] Found existing index file: {index_file}")
                self.index = faiss.read_index(index_file)
                with open(data_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"[SUCCESS] Loaded existing index with {self.index.ntotal} vectors and {len(self.documents)} documents")
            except Exception as e:
                print(f"[ERROR] Error loading existing index: {e}")
                self._create_new_index()
        else:
            print(f"[INFO] No existing index found, creating new one")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        print(f"[INFO] Created new FAISS index with dimension {self.dimension}")
    
    def add_vector(self, vector: List[float], document: Dict[str, Any]):
        """
        Add a vector and its associated document to the database
        
        Args:
            vector: The embedding vector
            document: The document data associated with the vector
        """
        # Convert vector to numpy array
        vector_np = np.array([vector]).astype('float32')
        
        # Add to FAISS index
        self.index.add(vector_np)
        
        # Store document data
        self.documents.append(document)
        
        # Save index and data
        self._save()
        print(f"[INFO] Added vector for document {document['id']}, total vectors: {self.index.ntotal}")
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: The query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents with similarity scores
        """
        if self.index.ntotal == 0:
            print("[WARNING] Search attempted on empty index")
            return []
        
        # Convert query vector to numpy array
        query_np = np.array([query_vector]).astype('float32')
        
        # Search the index
        k_to_search = min(k, self.index.ntotal)
        print(f"[INFO] Searching for {k_to_search} nearest neighbors among {self.index.ntotal} vectors")
        distances, indices = self.index.search(query_np, k_to_search)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means no result
                doc = self.documents[idx].copy()
                similarity = float(1 - distances[0][i] / 2)  # Convert L2 distance to similarity score
                doc["similarity_score"] = similarity
                results.append(doc)
                print(f"[DEBUG] Match {i+1}: ID={doc['id']}, Score={similarity:.4f}")
        
        print(f"[INFO] Found {len(results)} matches")
        return results
    
    def clear(self):
        """Clear the database and create a new index"""
        print("[INFO] Clearing vector database")
        self._create_new_index()
        
        # Remove saved files if they exist
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
            print(f"[INFO] Removed index file: {self.index_file}")
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
            print(f"[INFO] Removed data file: {self.data_file}")
    
    def _save(self):
        """Save the index and document data to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"[INFO] Saved index with {self.index.ntotal} vectors to {self.index_file}")
