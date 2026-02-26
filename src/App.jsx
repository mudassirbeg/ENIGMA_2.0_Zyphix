import { useState, useMemo } from "react";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(LineElement, PointElement, LinearScale, Title, Tooltip, Legend);

function App() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [iterations, setIterations] = useState(100);
  const [degree, setDegree] = useState(1);
  const [m, setM] = useState(0);
  const [b, setB] = useState(0);
  const [loss, setLoss] = useState(0);

  const dataPoints = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => {
      const x = i;
      const y = 2 * x + 5 + (Math.random() - 0.5) * 10;
      return { x, y };
    });
  }, []);

  const trainModel = () => {
    let newM = m;
    let newB = b;

    for (let i = 0; i < iterations; i++) {
      let dm = 0;
      let db = 0;

      dataPoints.forEach((point) => {
        const prediction = newM * point.x + newB;
        const error = prediction - point.y;

        dm += error * point.x;
        db += error;
      });

      newM -= (learningRate * dm) / dataPoints.length;
      newB -= (learningRate * db) / dataPoints.length;
    }

    let totalError = 0;
    dataPoints.forEach((point) => {
      const prediction = newM * point.x + newB;
      totalError += Math.pow(prediction - point.y, 2);
    });

    const mse = totalError / dataPoints.length;

    setM(newM);
    setB(newB);
    setLoss(mse);
  };

  const lineData = dataPoints.map((point) => {
    let yValue = 0;

    for (let i = 0; i <= degree; i++) {
      yValue += m * Math.pow(point.x, i);
    }

    return {
      x: point.x,
      y: yValue + b,
    };
  });

  const chartData = {
    datasets: [
      {
        label: "Data Points",
        data: dataPoints,
        backgroundColor: "blue",
        showLine: false,
      },
      {
        label: "Model Curve",
        data: lineData,
        borderColor: "red",
        borderWidth: 2,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      x: { type: "linear", position: "bottom" },
    },
  };

  return (
    <div style={{ textAlign: "center", padding: "30px" }}>
      <h1>ML Learning Sandbox</h1>
      <p>Gradient Descent + Overfitting Simulator</p>

      <div style={{ margin: "20px" }}>
        <label>Learning Rate: {learningRate}</label>
        <br />
        <input
          type="range"
          min="0.0001"
          max="0.01"
          step="0.0001"
          value={learningRate}
          onChange={(e) => setLearningRate(Number(e.target.value))}
        />
      </div>

      <div style={{ margin: "20px" }}>
        <label>Iterations: {iterations}</label>
        <br />
        <input
          type="range"
          min="10"
          max="500"
          step="10"
          value={iterations}
          onChange={(e) => setIterations(Number(e.target.value))}
        />
      </div>

      <div style={{ margin: "20px" }}>
        <label>Model Complexity (Degree): {degree}</label>
        <br />
        <input
          type="range"
          min="1"
          max="8"
          step="1"
          value={degree}
          onChange={(e) => setDegree(Number(e.target.value))}
        />
      </div>

      <button
        onClick={trainModel}
        style={{ padding: "10px 20px", marginBottom: "20px" }}
      >
        Train Model
      </button>

      <div style={{ width: "70%", margin: "auto" }}>
        <Line data={chartData} options={options} />
      </div>

      <div style={{ marginTop: "20px" }}>
        <p>Current Equation: y = {m.toFixed(2)}x + {b.toFixed(2)}</p>
        <p>Current Loss (MSE): {loss.toFixed(2)}</p>
      </div>
    </div>
  );
}

export default App;