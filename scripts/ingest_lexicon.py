import sqlite3
import requests
import json

conn = sqlite3.connect('data/hebrew_tutor.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS lexicon
             (word TEXT PRIMARY KEY, root TEXT, definition TEXT, grammar TEXT)''')

# Example: Fetch sample lexicon from open source (replace with Sefaria API if available)
# For PoC, use placeholder data; in production, use Sefaria's Mongo dump or API
lexicon_data = [
    ("בראשית", "ראש", "In the beginning", "Preposition + Noun"),
    ("אלהים", "אל", "God", "Noun, Masculine Plural"),
    # Add more from dataset
]
for word, root, defn, grammar in lexicon_data:
    c.execute("INSERT OR REPLACE INTO lexicon (word, root, definition, grammar) VALUES (?, ?, ?, ?)", (word, root, defn, grammar))

conn.commit()
conn.close()
print("Lexicon ingested.")