import React, { useState, useEffect } from 'react';

interface ReadingProps {
  verseText: string;
}

const UserReadingHighlight: React.FC<ReadingProps> = ({ verseText }) => {
    const [recognizedText, setRecognizedText] = useState('');
    const [highlightedIndex, setHighlightedIndex] = useState(-1);
    const words = verseText.split(' ');

    useEffect(() => {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'he-IL';  // Hebrew
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onresult = (event) => {
            const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase();
            setRecognizedText(transcript);
            const spokenWords = transcript.split(' ');
            const index = spokenWords.length - 1;
            if (index < words.length && spokenWords[index] === words[index].toLowerCase()) {
                setHighlightedIndex(index);
            }
        };

        recognition.start();

        return () => recognition.stop();
    }, [words]);

    const handleSessionEnd = async () => {
        // Call backend for WhisperX analysis and feedback
        try {
            const response = await axios.post('https://localhost:8000/feedback', { recognizedText }, { rejectUnauthorized: false });
            alert(`Feedback: ${response.data.improvements}`);
        } catch (error) {
            alert("Feedback error");
        }
    };

    return (
        <div className="rtl text-lg">
            <p>
                {words.map((word, index) => (
                    <span key={index} className={index === highlightedIndex ? 'bg-green-200' : ''}>
                        {word} 
                    </span>
                ))}
            </p>
            <button onClick={handleSessionEnd} className="mt-4 bg-blue-500 text-white p-2">End Session and Get Feedback</button>
        </div>
    );
};

export default UserReadingHighlight;