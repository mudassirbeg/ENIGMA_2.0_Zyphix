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
  const [trainLoss, setTrainLoss] = useState(0);
  const [testLoss, setTestLoss] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [showTheory, setShowTheory] = useState(false);

  // Generate Dataset
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
    if (isTraining) return;

    setIsTraining(true);

    let newWeights = Array(degree + 1).fill(0);
    let history = [];
    let currentIteration = 0;

    const interval = setInterval(() => {
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

      let iterationLoss = 0;
      trainData.forEach((point) => {
        let prediction = 0;
        for (let d = 0; d <= degree; d++) {
          prediction += newWeights[d] * Math.pow(point.x, d);
        }
        iterationLoss += Math.pow(prediction - point.y, 2);
      });

      history.push(iterationLoss / trainData.length);

      setWeights([...newWeights]);
      setLossHistory([...history]);

      currentIteration++;

      if (currentIteration >= iterations) {
        clearInterval(interval);

        const finalTrainLoss = history[history.length - 1];

        let totalTestError = 0;
        testData.forEach((point) => {
          let prediction = 0;
          for (let d = 0; d <= degree; d++) {
            prediction += newWeights[d] * Math.pow(point.x, d);
          }
          totalTestError += Math.pow(prediction - point.y, 2);
        });

        const finalTestLoss = totalTestError / testData.length;

        setTrainLoss(finalTrainLoss);
        setTestLoss(finalTestLoss);
        setIsTraining(false);
      }
    }, 30);
  };

  const regenerateData = () => {
    setDataset(generateDataset());
    setWeights(Array(degree + 1).fill(0));
    setTrainLoss(0);
    setTestLoss(0);
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

  const isOverfitting = testLoss > trainLoss * 1.2;
  const isUnderfitting = trainLoss > 20 && testLoss > 20;

  let modelState = "Balanced";
  if (isUnderfitting) modelState = "High Bias (Underfitting)";
  else if (isOverfitting) modelState = "High Variance (Overfitting)";

  const chartData = {
    datasets: [
      {
        label: "Train Data",
        data: trainData,
        backgroundColor: "blue",
        showLine: false,
      },
      {
        label: "Test Data",
        data: testData,
        backgroundColor: "green",
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

  const lossChartData = {
    labels: lossHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "Training Loss Over Iterations",
        data: lossHistory,
        borderColor: "purple",
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
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f6f9",
        display: "flex",
        justifyContent: "center",
        padding: "40px",
      }}
    >
      <div
        style={{
          background: "white",
          padding: "40px",
          borderRadius: "12px",
          boxShadow: "0 10px 25px rgba(0,0,0,0.1)",
          width: "100%",
          maxWidth: "900px",
          textAlign: "center",
        }}
      >
        <h1>ML Learning Sandbox</h1>
        <p>Interactive Animated Gradient Descent Simulator</p>

        {/* Controls */}
        <div style={{ margin: "20px" }}>
          <label>Learning Rate: {learningRate}</label><br />
          <input type="range" min="0.0001" max="0.01" step="0.0001"
            value={learningRate}
            onChange={(e) => setLearningRate(Number(e.target.value))}
          />
        </div>

        <div style={{ margin: "20px" }}>
          <label>Iterations: {iterations}</label><br />
          <input type="range" min="10" max="500" step="10"
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
          />
        </div>

        <div style={{ margin: "20px" }}>
          <label>Model Complexity (Degree): {degree}</label><br />
          <input type="range" min="1" max="8" step="1"
            value={degree}
            onChange={(e) => setDegree(Number(e.target.value))}
          />
        </div>

        <div style={{ margin: "20px" }}>
          <label>Regularization (λ): {lambda}</label><br />
          <input type="range" min="0" max="1" step="0.01"
            value={lambda}
            onChange={(e) => setLambda(Number(e.target.value))}
          />
        </div>

        <div style={{ margin: "20px" }}>
          <label>Noise Level: {noiseLevel}</label><br />
          <input type="range" min="0" max="30" step="1"
            value={noiseLevel}
            onChange={(e) => setNoiseLevel(Number(e.target.value))}
          />
        </div>

        <button onClick={trainModel} disabled={isTraining}>
          {isTraining ? "Training..." : "Train Model"}
        </button>

        <button onClick={regenerateData} style={{ marginLeft: "10px" }}>
          Generate New Dataset
        </button>

        <div style={{ marginTop: "30px" }}>
          <Line data={chartData} options={options} />
        </div>

        {lossHistory.length > 0 && (
          <div style={{ marginTop: "40px" }}>
            <h3>Loss vs Iterations</h3>
            <Line data={lossChartData} />
          </div>
        )}

        <div style={{ marginTop: "20px" }}>
          <p>
            Current Model:{" "}
            {weights.map((w, i) => `${w.toFixed(2)}x^${i}`).join(" + ")}
          </p>
          <p>Training Loss: {trainLoss.toFixed(2)}</p>
          <p>Test Loss: {testLoss.toFixed(2)}</p>
          <h3>Bias–Variance Analysis</h3>
          <p style={{ fontWeight: "bold" }}>{modelState}</p>
        </div>

        {/* Explanation Toggle */}
        <button
          onClick={() => setShowTheory(!showTheory)}
          style={{ marginTop: "20px" }}
        >
          {showTheory ? "Hide Explanation" : "Show ML Explanation"}
        </button>

        {showTheory && (
          <div
            style={{
              marginTop: "20px",
              textAlign: "left",
              background: "#f9f9f9",
              padding: "20px",
              borderRadius: "8px",
            }}
          >
            <h3>📘 Gradient Descent</h3>
            <p>
              Gradient Descent minimizes error by updating weights step-by-step
              in the direction that reduces loss.
            </p>

            <h3>📘 Overfitting</h3>
            <p>
              Overfitting occurs when the model performs well on training data
              but poorly on test data.
            </p>

            <h3>📘 Regularization</h3>
            <p>
              Regularization penalizes large weights to reduce model complexity
              and prevent overfitting.
            </p>

            <h3>📘 Bias–Variance Tradeoff</h3>
            <p>
              High bias leads to underfitting, high variance leads to
              overfitting. A balanced model generalizes well.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;