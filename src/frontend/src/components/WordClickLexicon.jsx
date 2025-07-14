import { useState } from 'react';
import axios from 'axios';

interface LexiconProps {
  word: string;
}

const WordClickLexicon = ({ word }: LexiconProps) => {
  const [lexiconData, setLexiconData] = useState(null);

  const handleClick = async () => {
    try {
      const encodedWord = encodeURIComponent(word);  # Sanitize for URI
      const response = await axios.get(`/lexicon/${encodedWord}`);
      setLexiconData(response.data);
      console.log(response.data);
    } catch (error) {
      console.error('Lexicon fetch error', error);
    }
  };

  return (
    <span onClick={handleClick} className="cursor-pointer hover:underline">
      {word}
    </span>
  );
};

export default WordClickLexicon;