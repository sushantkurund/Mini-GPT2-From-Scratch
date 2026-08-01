import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(300);
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  const generateText = async () => {
    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/generate",
        {
          prompt,
          temperature,
          max_tokens: maxTokens,
        }
      );

      setOutput(response.data.generated_text);
    } catch (error) {
      console.error(error);
      setOutput("Unable to generate text. Please ensure the server is running.");
    }

    setLoading(false);
  };

  const copyText = () => {
    if (!output) return;
    navigator.clipboard.writeText(output);
  };

  const downloadText = () => {
    if (!output) return;

    const blob = new Blob([output], {
      type: "text/plain",
    });

    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "generated_text.txt";
    a.click();

    window.URL.revokeObjectURL(url);
  };

  const clearAll = () => {
    setPrompt("");
    setOutput("");
  };

  return (
    <div className="container">

      <h1>GPT-2 From Scratch</h1>

      <p className="subtitle">
        Character-Level Language Model trained on the Tiny Shakespeare dataset
      </p>

      <div className="about-card">
        <h3>About this Project</h3>

        <p>
          This application demonstrates a GPT-2 inspired character-level
          language model trained on the Tiny Shakespeare dataset. Enter a prompt
          below to generate Shakespeare-style text using a transformer model
          implemented completely from scratch.
        </p>
      </div>

      <textarea
        placeholder="Try: KING, ROMEO, JULIET, To be..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <div className="controls">

        <div>
          <label>
            Temperature
            <strong> {temperature}</strong>
          </label>

          <input
            type="range"
            min="0.1"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
        </div>

        <div>
          <label>
            Maximum Tokens
            <strong> {maxTokens}</strong>
          </label>

          <input
            type="range"
            min="50"
            max="1000"
            step="50"
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
          />
        </div>

      </div>

      <button
        className="generate-btn"
        onClick={generateText}
        disabled={loading}
      >
        {loading ? "✨ Generating Shakespeare..." : "🚀 Generate Text"}
      </button>

      <h2>Generated Output</h2>

      <div className="output">
        {output}
      </div>

      <div className="actions">

        <button onClick={copyText}>
          📋 Copy
        </button>

        <button onClick={downloadText}>
          💾 Download
        </button>

        <button onClick={clearAll}>
          🗑 Clear
        </button>

      </div>

    </div>
  );
}

export default App;