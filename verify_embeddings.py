import numpy as np
from ares.databases.embedding_database import IndexManager, FaissIndex, EMBEDDING_DB_PATH

def verify_embeddings():
    print(f"Checking embeddings in {EMBEDDING_DB_PATH}...")
    manager = IndexManager(EMBEDDING_DB_PATH, FaissIndex)
    
    stats = manager.get_overall_stats()
    print("Overall Stats:", stats)
    
    print("\nIndices found:")
    for name in manager.indices:
        print(f"- {name}: {manager.metadata[name]}")
        
    # Check if we have embeddings for our dataset
    # We expect indices like 'states', 'actions', 'task_language_instruction', 'description_estimate'
    
    required_indices = ['states', 'actions', 'task_language_instruction', 'description_estimate']
    missing = [idx for idx in required_indices if idx not in manager.indices]
    
    if missing:
        print(f"\n❌ Missing indices: {missing}")
    else:
        print("\n✅ All required indices present.")
        
    # Check count
    if manager.indices:
        first_index = list(manager.indices.keys())[0]
        count = manager.metadata[first_index]['n_entries']
        print(f"\nTotal entries in {first_index}: {count}")
        if count > 0:
            print("✅ Embeddings found!")
        else:
            print("❌ No entries found in index.")

if __name__ == "__main__":
    verify_embeddings()
