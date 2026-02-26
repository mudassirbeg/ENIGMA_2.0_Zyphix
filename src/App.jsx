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
  const [learningRate, setLearningRate] = useState(0.001);
  const [iterations, setIterations] = useState(150);
  const [degree, setDegree] = useState(2);
  const [lambda, setLambda] = useState(0);
  const [noiseLevel, setNoiseLevel] = useState(5);

  const [weights, setWeights] = useState([0, 0]);
  const [trainLoss, setTrainLoss] = useState(0);
  const [testLoss, setTestLoss] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);

  const generateDataset = () => {
    const allPoints = Array.from({ length: 20 }, (_, i) => {
      const x = i / 20;
      const y = 2 * x + 0.5 + (Math.random() - 0.5) * (noiseLevel / 50);
      return { x, y };
    });

    const splitIndex = Math.floor(allPoints.length * 0.7);

    return {
      trainData: allPoints.slice(0, splitIndex),
      testData: allPoints.slice(splitIndex),
    };
  };

  const [dataset, setDataset] = useState(generateDataset());
  const { trainData, testData } = dataset;

  const trainModel = () => {
    let newWeights = Array(degree + 1).fill(0);
    let history = [];

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
        const reg = lambda * newWeights[d];
        newWeights[d] -=
          learningRate * (gradients[d] / trainData.length + reg);
      }

      let loss = 0;
      trainData.forEach((point) => {
        let prediction = 0;
        for (let d = 0; d <= degree; d++) {
          prediction += newWeights[d] * Math.pow(point.x, d);
        }
        loss += Math.pow(prediction - point.y, 2);
      });

      const mse = loss / trainData.length;
      if (!isFinite(mse)) break;
      history.push(mse);
    }

    let totalTestError = 0;
    testData.forEach((point) => {
      let prediction = 0;
      for (let d = 0; d <= degree; d++) {
        prediction += newWeights[d] * Math.pow(point.x, d);
      }
      totalTestError += Math.pow(prediction - point.y, 2);
    });

    setWeights(newWeights);
    setTrainLoss(history[history.length - 1] || 0);
    setTestLoss(totalTestError / testData.length);
    setLossHistory(history);
  };

  const regenerateData = () => {
    setDataset(generateDataset());
    setWeights(Array(degree + 1).fill(0));
    setTrainLoss(0);
    setTestLoss(0);
    setLossHistory([]);
  };

  const lineData = Array.from({ length: 100 }, (_, i) => {
    const x = i / 100;
    let yVal = 0;
    for (let d = 0; d <= degree; d++) {
      yVal += weights[d] * Math.pow(x, d);
    }
    return { x, y: yVal };
  });

  const scatterData = {
    datasets: [
      {
        label: "Train Data",
        data: trainData,
        backgroundColor: "#3b82f6",
        showLine: false,
      },
      {
        label: "Test Data",
        data: testData,
        backgroundColor: "#10b981",
        showLine: false,
      },
      {
        label: "Model Curve",
        data: lineData,
        borderColor: "#ef4444",
        borderWidth: 3,
        fill: false,
      },
    ],
  };

  const lossChartData = {
    labels: lossHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "Training Loss",
        data: lossHistory,
        borderColor: "#8b5cf6",
        borderWidth: 3,
        fill: false,
      },
    ],
  };

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "linear-gradient(135deg, #4f46e5, #9333ea)",
        fontFamily: "Segoe UI, sans-serif",
      }}
    >
      <div
        style={{
          padding: "25px",
          textAlign: "center",
          fontSize: "40px",
          fontWeight: "600",
          color: "yellow",
          letterSpacing: "1px",
        }}
      >
        ML Learning Sandbox — Interactive ML Dashboard
      </div>

      <div style={{ flex: 1, display: "flex", padding: "30px", gap: "25px" }}>
        {/* CONTROL PANEL */}
        <div
          style={{
            width: "330px",
            background: "rgba(255,255,255,0.15)",
            backdropFilter: "blur(10px)",
            padding: "25px",
            borderRadius: "20px",
            boxShadow: "0 10px 40px rgba(0,0,0,0.3)",
            color: "white",
          }}
        >
          <h3 style={{ marginBottom: "20px" }}>Model Controls</h3>

          {[
            ["Learning Rate", learningRate, setLearningRate, 0.0001, 0.003, 0.0001],
            ["Iterations", iterations, setIterations, 10, 300, 10],
            ["Degree", degree, setDegree, 1, 5, 1],
            ["Regularization (λ)", lambda, setLambda, 0, 1, 0.01],
            ["Noise Level", noiseLevel, setNoiseLevel, 0, 10, 1],
          ].map(([label, value, setter, min, max, step], i) => (
            <div key={i} style={{ marginBottom: "20px" }}>
              <label>
                {label}: <b>{value}</b>
              </label>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => setter(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          ))}

          <button
            onClick={trainModel}
            style={{
              width: "100%",
              padding: "12px",
              borderRadius: "10px",
              border: "none",
              background: "#facc15",
              fontWeight: "bold",
              cursor: "pointer",
              marginBottom: "10px",
            }}
          >
            Train Model
          </button>

          <button
            onClick={regenerateData}
            style={{
              width: "100%",
              padding: "12px",
              borderRadius: "10px",
              border: "none",
              background: "#ffffff",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            Generate Dataset
          </button>

          <div style={{ marginTop: "20px" }}>
            <p><b>Training Loss:</b> {trainLoss.toFixed(6)}</p>
            <p><b>Test Loss:</b> {testLoss.toFixed(6)}</p>
          </div>
        </div>

        {/* GRAPHS */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: "25px",
          }}
        >
          <div
            style={{
              flex: 1,
              background: "white",
              borderRadius: "20px",
              padding: "20px",
              boxShadow: "0 10px 40px rgba(0,0,0,0.25)",
            }}
          >
            <Line
              data={scatterData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: { type: "linear", min: 0, max: 1 },
                },
              }}
            />
          </div>

          <div
            style={{
              flex: 1,
              background: "white",
              borderRadius: "20px",
              padding: "20px",
              boxShadow: "0 10px 40px rgba(0,0,0,0.25)",
            }}
          >
            <Line
              data={lossChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;