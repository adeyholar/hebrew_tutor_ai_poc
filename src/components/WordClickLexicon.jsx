import React from 'react';
import axios from 'axios';

interface LexiconData {
  root: string;
  definition: string;
  grammar: string;
}

const WordClickLexicon: React.FC<{ word: string }> = ({ word }) => {
    const handleClick = async () => {
        try {
            const response = await axios.get<LexiconData>(`https://localhost:8000/lexicon/${word}`, {
                httpsAgent: new https.Agent({ rejectUnauthorized: false }),  // Dev-only
            });
            const data = response.data;
            alert(`Root: ${data.root}\nDefinition: ${data.definition}\nGrammar: ${data.grammar}`);
        } catch (error) {
            alert("Lexicon not found for word: " + word);
        }
    };

    return (
        <span onClick={handleClick} className="cursor-pointer text-blue-600 underline mx-1">
            {word}
        </span>
    );
};

export default WordClickLexicon;