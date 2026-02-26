import { useState } from "react";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale
);

function App() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [iterations, setIterations] = useState(200);
  const [degree, setDegree] = useState(1);
  const [lambda, setLambda] = useState(0);
  const [noiseLevel, setNoiseLevel] = useState(10);
  const [weights, setWeights] = useState([0, 0]);
  const [lossHistory, setLossHistory] = useState([]);

  const generateDataset = () => {
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
        newWeights[d] -=
          learningRate *
          (gradients[d] / trainData.length + lambda * newWeights[d]);
      }

      let loss = 0;
      trainData.forEach((point) => {
        let prediction = 0;
        for (let d = 0; d <= degree; d++) {
          prediction += newWeights[d] * Math.pow(point.x, d);
        }
        loss += Math.pow(prediction - point.y, 2);
      });

      history.push(loss / trainData.length);
    }

    setWeights(newWeights);
    setLossHistory(history);
  };

  const regenerateData = () => {
    setDataset(generateDataset());
    setWeights(Array(degree + 1).fill(0));
    setLossHistory([]);
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

  const lossChartData = {
    labels: lossHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "Training Loss",
        data: lossHistory,
        borderColor: "#7c3aed",
        borderWidth: 2,
        fill: false,
      },
    ],
  };

  return (
    <div style={{
      width: "100vw",
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      background: "#f3f4f6"
    }}>
      
      {/* HEADER */}
      <div style={{
        height: "70px",
        background: "#111827",
        color: "white",
        display: "flex",
        alignItems: "center",
        paddingLeft: "30px",
        fontSize: "22px",
        fontWeight: "bold"
      }}>
        ML Learning Sandbox — Full Stack ML Simulator
      </div>

      {/* CONTENT */}
      <div style={{ flex: 1, display: "flex" }}>

        {/* LEFT PANEL */}
        <div style={{
          width: "350px",
          background: "white",
          padding: "20px",
          borderRight: "1px solid #e5e7eb"
        }}>
          <h3>Controls</h3>

          <label>Learning Rate: {learningRate}</label>
          <input type="range" min="0.0001" max="0.01" step="0.0001"
            value={learningRate}
            onChange={(e) => setLearningRate(Number(e.target.value))}
          />

          <label>Iterations: {iterations}</label>
          <input type="range" min="10" max="500" step="10"
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
          />

          <label>Degree: {degree}</label>
          <input type="range" min="1" max="8" step="1"
            value={degree}
            onChange={(e) => setDegree(Number(e.target.value))}
          />

          <label>Regularization (λ): {lambda}</label>
          <input type="range" min="0" max="1" step="0.01"
            value={lambda}
            onChange={(e) => setLambda(Number(e.target.value))}
          />

          <label>Noise Level: {noiseLevel}</label>
          <input type="range" min="0" max="30" step="1"
            value={noiseLevel}
            onChange={(e) => setNoiseLevel(Number(e.target.value))}
          />

          <button onClick={trainModel} style={{ width: "100%", marginTop: "15px" }}>
            Train Model
          </button>

          <button onClick={regenerateData} style={{ width: "100%", marginTop: "10px" }}>
            Generate Dataset
          </button>
        </div>

        {/* RIGHT PANEL */}
        <div style={{
          flex: 1,
          padding: "30px",
          display: "flex",
          flexDirection: "column",
          gap: "30px"
        }}>
          <div style={{ flex: 1 }}>
            <Line data={chartData} options={{ responsive: true, maintainAspectRatio: false }} />
          </div>

          {lossHistory.length > 0 && (
            <div style={{ flex: 1 }}>
              <Line data={lossChartData} options={{ responsive: true, maintainAspectRatio: false }} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;