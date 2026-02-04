import os
import sqlite3
from tqdm import tqdm

def cleanup_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all speech files
    cursor.execute("SELECT id, path FROM speechfile")
    rows = cursor.fetchall()
    
    to_delete = []
    print("Checking for missing files...")
    for id, path in tqdm(rows):
        if not os.path.exists(path):
            to_delete.append(id)
            
    if to_delete:
        print(f"Deleting {len(to_delete)} records for missing files...")
        # Chunk deletion for safety
        chunk_size = 999
        for i in range(0, len(to_delete), chunk_size):
            chunk = to_delete[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            cursor.execute(f"DELETE FROM speechfile WHERE id IN ({placeholders})", chunk)
        
        conn.commit()
        print("Cleanup complete.")
    else:
        print("No missing files found in DB.")
        
    conn.close()

if __name__ == "__main__":
    cleanup_db("data/metadata.db")
