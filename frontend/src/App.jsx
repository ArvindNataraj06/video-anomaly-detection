import { useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function ObjectTable({ title, data }) {
  if (!data) return null;

  return (
    <div className="section-card">
      <h3>{title}</h3>
      <div className="table">
        {Object.entries(data).map(([key, value]) => (
          <div className="table-row" key={key}>
            <span>{key.replaceAll("_", " ")}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeVideo = async () => {
    if (!file) {
      alert("Please select a video first");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/analyze-traffic-video`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Error analyzing video");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const r = result?.result;
  const risk = r?.traffic_risk_analysis?.final_traffic_risk || "N/A";

  return (
    <div className="page">
      <header className="hero">
        <p className="tag">AI Traffic Analytics</p>
        <h1>Traffic Video Intelligence System</h1>
        <p>
          Upload a traffic video to detect objects, estimate movement direction,
          and analyze traffic risk.
        </p>
      </header>

      <div className="upload-card">
        <div>
          <label className="file-label">Select traffic video</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files[0])}
          />
          {file && <p className="file-name">{file.name}</p>}
        </div>

        <button onClick={analyzeVideo} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Video"}
        </button>
      </div>

      {loading && (
        <div className="loading-card">
          <div className="loader"></div>
          <p>Processing video. This may take a few minutes...</p>
        </div>
      )}

      {r && (
        <main className="results">
          <div className="results-header">
            <div>
              <h2>Analysis Result</h2>
              <p>{r.method}</p>
            </div>

            <span className={`risk-badge ${risk.toLowerCase()}`}>{risk}</span>
          </div>

          <div className="summary-grid">
            <div className="summary-card">
              <span>Total Frames</span>
              <strong>{r.total_frames}</strong>
            </div>

            <div className="summary-card">
              <span>Processed Frames</span>
              <strong>{r.processed_frames}</strong>
            </div>

            <div className="summary-card">
              <span>FPS</span>
              <strong>{r.fps}</strong>
            </div>

            <div className="summary-card">
              <span>Max Objects / Frame</span>
              <strong>{r.max_objects_in_frame}</strong>
            </div>
          </div>

          <div className="risk-card">
            <h3>Traffic Risk Analysis</h3>

            <div className="risk-grid">
              <div>
                <span>Density Level</span>
                <strong>{r.traffic_risk_analysis?.traffic_density_level}</strong>
              </div>

              <div>
                <span>Stationary %</span>
                <strong>{r.traffic_risk_analysis?.stationary_percentage}%</strong>
              </div>

              <div>
                <span>Dominant Direction</span>
                <strong>{r.traffic_risk_analysis?.dominant_direction}</strong>
              </div>
            </div>

            <ul>
              {r.traffic_risk_analysis?.risk_reasons?.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          </div>

          <div className="two-column">
            <ObjectTable
              title="Traffic Object Detections"
              data={r.traffic_object_detections_across_frames}
            />

            <ObjectTable
              title="Unique Tracked Objects"
              data={r.unique_tracked_traffic_objects}
            />
          </div>

          <ObjectTable title="Movement Directions" data={r.movement_directions} />

          <section className="section-card">
            <h3>Evidence Frames</h3>

            <div className="image-grid">
              {r.evidence_frames?.map((frame, index) => (
                <div className="image-card" key={index}>
                  <img
                    src={`${API_BASE}/${frame.saved_frame}`}
                    alt={`Evidence frame ${index + 1}`}
                  />
                  <div className="image-info">
                    <strong>Frame {frame.frame_number}</strong>
                    <span>{frame.objects.join(", ")}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

export default App;