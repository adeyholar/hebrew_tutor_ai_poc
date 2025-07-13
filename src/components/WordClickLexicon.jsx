import React from 'react';
import axios from 'axios';

const WordClickLexicon = ({ word }) => {
    const handleClick = async () => {
        try {
            const response = await axios.get(`https://localhost:8000/lexicon/${word}`, {
                httpsAgent: new https.Agent({ rejectUnauthorized: false }),  // Dev-only for self-signed cert
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