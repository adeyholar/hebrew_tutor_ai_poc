import React from 'react';
import VerseDisplay from './components/VerseDisplay';
import UserReadingHighlight from './components/UserReadingHighlight';
import './index.css';  // Import Tailwind

const App: React.FC = () => {
    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold mb-4">Hebrew Tutor AI PoC</h1
            <VerseDisplay verseText="בראשית אלהים" audioPath="D:/AI/Tanach_Audio/Genesis_Chapter_1.mp3" />
            <UserReadingHighlight verseText="בראשית אלהים" />
        </div>
    );
};

export default App;