import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.max_position = max_position
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        batch_size, seq_len = token_ids.shape
        if (batch_size < 1 or batch_size > 32):
            raise ValueError("Invalid batch size")
            
        if (seq_len < 1 or seq_len > self.max_position):
            raise ValueError("Invalid sequence length")

        embeddings = []
        for j in range(batch_size):
            seq_emb = []
            for i, token in enumerate(token_ids[j]):
                token_emb = self.token_embeddings[token]
                pos_emb = self.position_embeddings[i]
                seg_emb = self.segment_embeddings[segment_ids[j][i]]
                seq_emb.append(token_emb + pos_emb + seg_emb)
            embeddings.append(seq_emb)
            
        return np.asarray(embeddings)
                
                
        
