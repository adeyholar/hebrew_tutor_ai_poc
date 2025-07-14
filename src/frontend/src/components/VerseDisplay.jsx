import React from 'react';
import WordClickLexicon from './WordClickLexicon';

interface VerseProps {
  verseText: string;
  audioPath: string;
}

const VerseDisplay: React.FC<VerseProps> = ({ verseText, audioPath }) => {
    return (
        <div className="rtl text-lg">
            <p>
                {verseText.split(' ').map((word, index) => (
                    <WordClickLexicon key={index} word={word} />
                ))}
            </p>
            <audio controls src={audioPath} className="mt-4">
                Your browser does not support the audio element.
            </audio>
        </div>
    );
};

export default VerseDisplay;