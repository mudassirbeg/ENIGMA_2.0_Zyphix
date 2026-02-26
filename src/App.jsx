import { useState } from "react";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [iterations, setIterations] = useState(200);
  const [degree, setDegree] = useState(1);
  const [lambda, setLambda] = useState(0);
  const [noiseLevel, setNoiseLevel] = useState(10);

  const [weights, setWeights] = useState([0, 0]);
  const [trainLoss, setTrainLoss] = useState(0);
  const [testLoss, setTestLoss] = useState(0);

  function generateDataset() {
    const allPoints = Array.from({ length: 20 }, (_, i) => {
      const x = i;
      const y = 2 * x + 5 + (Math.random() - 0.5) * noiseLevel;
      return { x, y };
    });

    const splitIndex = Math.floor(allPoints.length * 0.7);

    return {
      trainData: allPoints.slice(0, splitIndex),
      testData: allPoints.slice(splitIndex),
    };
  }

  const [dataset, setDataset] = useState(generateDataset());
  const { trainData, testData } = dataset;

  const trainModel = () => {
    let newWeights = Array(degree + 1).fill(0);

    for (let iter = 0; iter < iterations; iter++) {
      let gradients = Array(degree + 1).fill(0);

      trainData.forEach((point) => {
        let prediction = 0;
        for (let d = 0; d <= degree; d++) {
          prediction += newWeights[d] * Math.pow(point.x, d);
        }

        const error = prediction - point.y;

        for (let d = 0; d <= degree; d++) {
          gradients[d] += error * Math.pow(point.x, d);
        }
      });

      for (let d = 0; d <= degree; d++) {
        const regularizationTerm = lambda * newWeights[d];
        newWeights[d] -=
          learningRate *
          (gradients[d] / trainData.length + regularizationTerm);
      }
    }

    let totalTrainError = 0;
    trainData.forEach((point) => {
      let prediction = 0;
      for (let d = 0; d <= degree; d++) {
        prediction += newWeights[d] * Math.pow(point.x, d);
      }
      totalTrainError += Math.pow(prediction - point.y, 2);
    });

    const trainMSE = totalTrainError / trainData.length;

    let totalTestError = 0;
    testData.forEach((point) => {
      let prediction = 0;
      for (let d = 0; d <= degree; d++) {
        prediction += newWeights[d] * Math.pow(point.x, d);
      }
      totalTestError += Math.pow(prediction - point.y, 2);
    });

    const testMSE = totalTestError / testData.length;

    setWeights(newWeights);
    setTrainLoss(trainMSE);
    setTestLoss(testMSE);
  };

  const regenerateData = () => {
    setDataset(generateDataset());
    setWeights(Array(degree + 1).fill(0));
    setTrainLoss(0);
    setTestLoss(0);
  };

  const allPoints = [...trainData, ...testData];

  const lineData = allPoints.map((point) => {
    let yValue = 0;
    for (let d = 0; d <= degree; d++) {
      yValue += weights[d] * Math.pow(point.x, d);
    }
    return { x: point.x, y: yValue };
  });

  const chartData = {
    datasets: [
      {
        label: "Train Data",
        data: trainData,
        backgroundColor: "#2563eb",
        showLine: false,
      },
      {
        label: "Test Data",
        data: testData,
        backgroundColor: "#16a34a",
        showLine: false,
      },
      {
        label: "Model Curve",
        data: lineData,
        borderColor: "#dc2626",
        borderWidth: 2,
        fill: false,
      },
    ],
  };

 return (
  <div
    style={{
      height: "100vh",
      width: "100vw",
      display: "grid",
      gridTemplateRows: "70px 1fr",
      background: "linear-gradient(135deg, #eef2ff, #fdf4ff)",
      fontFamily: "system-ui",
    }}
  >
    {/* HEADER */}
    <div
      style={{
        background: "linear-gradient(90deg, #4f46e5, #7c3aed)",
        color: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        
        fontSize: "30px",
        fontWeight: "600",
        letterSpacing: "0.5px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
      }}
    >
      ML Learning Sandbox — Interactive ML Dashboard
    </div>

    {/* MAIN GRID */}
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "360px 1fr",
        gap: "25px",
        padding: "25px",
      }}
    >
      {/* LEFT PANEL */}
      <div
        style={{
          background: "rgba(255,255,255,0.9)",
          backdropFilter: "blur(10px)",
          borderRadius: "16px",
          padding: "25px",
          boxShadow: "0 15px 40px rgba(0,0,0,0.08)",
          display: "flex",
          flexDirection: "column",
          gap: "20px",
        }}
      >
        <h3 style={{ margin: 0, color: "#4f46e5" }}>Model Controls</h3>

        {/* SLIDERS */}
        {[
          { label: "Learning Rate", value: learningRate },
          { label: "Iterations", value: iterations },
          { label: "Degree", value: degree },
          { label: "Regularization (λ)", value: lambda },
          { label: "Noise Level", value: noiseLevel },
        ].map((item, i) => (
          <div key={i}>
            <label style={{ fontSize: "14px", fontWeight: 500 }}>
              {item.label}: <span style={{ color: "#7c3aed" }}>{item.value}</span>
            </label>
            <input
              type="range"
              style={{ width: "100%", accentColor: "#7c3aed" }}
              min={
                i === 0 ? "0.0001" :
                i === 1 ? "10" :
                i === 2 ? "1" :
                i === 3 ? "0" : "0"
              }
              max={
                i === 0 ? "0.01" :
                i === 1 ? "500" :
                i === 2 ? "8" :
                i === 3 ? "1" : "30"
              }
              step={
                i === 0 ? "0.0001" :
                i === 1 ? "10" :
                i === 2 ? "1" :
                i === 3 ? "0.01" : "1"
              }
              value={item.value}
              onChange={(e) => {
                const val = Number(e.target.value);
                if (i === 0) setLearningRate(val);
                if (i === 1) setIterations(val);
                if (i === 2) setDegree(val);
                if (i === 3) setLambda(val);
                if (i === 4) setNoiseLevel(val);
              }}
            />
          </div>
        ))}

        {/* BUTTONS */}
        <button
          onClick={trainModel}
          style={{
            padding: "12px",
            background: "linear-gradient(90deg, #4f46e5, #7c3aed)",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: 600,
            boxShadow: "0 8px 20px rgba(124,58,237,0.4)",
            transition: "0.2s",
          }}
        >
          Train Model
        </button>

        <button
          onClick={regenerateData}
          style={{
            padding: "12px",
            background: "#e5e7eb",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: 600,
            boxShadow: "0 6px 15px rgba(0,0,0,0.05)",
          }}
        >
          Generate Dataset
        </button>

        {/* METRICS */}
        <div
          style={{
            background: "linear-gradient(135deg, #f3e8ff, #ede9fe)",
            padding: "15px",
            borderRadius: "12px",
            marginTop: "10px",
            boxShadow: "0 8px 20px rgba(124,58,237,0.15)",
          }}
        >
          <p><b>Training Loss:</b> {trainLoss.toFixed(4)}</p>
          <p><b>Test Loss:</b> {testLoss.toFixed(4)}</p>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div
        style={{
          background: "rgba(255,255,255,0.9)",
          backdropFilter: "blur(10px)",
          borderRadius: "16px",
          padding: "25px",
          boxShadow: "0 15px 40px rgba(0,0,0,0.08)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <h3 style={{ marginBottom: "15px", color: "#4f46e5" }}>
          Model Visualization
        </h3>

        <div style={{ flex: 1 }}>
          <Line
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: {
                  type: "linear",
                  position: "bottom",
                },
              },
            }}
          />
        </div>
      </div>
    </div>
  </div>
);
}

export default App;