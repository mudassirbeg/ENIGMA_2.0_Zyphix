const express = require("express");
const cors = require("cors");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Health check route
app.get("/", (req, res) => {
  res.send("ML Sandbox Backend Running 🚀");
});

// Training API
app.post("/train", (req, res) => {
  const { learningRate, iterations, degree, lambda, noiseLevel } = req.body;

  // Generate synthetic dataset
  const allPoints = Array.from({ length: 20 }, (_, i) => {
    const x = i;
    const y = 2 * x + 5 + (Math.random() - 0.5) * noiseLevel;
    return { x, y };
  });

  const splitIndex = Math.floor(allPoints.length * 0.7);
  const trainData = allPoints.slice(0, splitIndex);
  const testData = allPoints.slice(splitIndex);

  let weights = Array(degree + 1).fill(0);

  // Gradient Descent
  for (let iter = 0; iter < iterations; iter++) {
    let gradients = Array(degree + 1).fill(0);

    trainData.forEach((point) => {
      let prediction = 0;

      for (let d = 0; d <= degree; d++) {
        prediction += weights[d] * Math.pow(point.x, d);
      }

      const error = prediction - point.y;

      for (let d = 0; d <= degree; d++) {
        gradients[d] += error * Math.pow(point.x, d);
      }
    });

    for (let d = 0; d <= degree; d++) {
      weights[d] -=
        learningRate *
        (gradients[d] / trainData.length + lambda * weights[d]);
    }
  }

  // Calculate Training Loss
  let trainLoss = 0;
  trainData.forEach((point) => {
    let prediction = 0;
    for (let d = 0; d <= degree; d++) {
      prediction += weights[d] * Math.pow(point.x, d);
    }
    trainLoss += Math.pow(prediction - point.y, 2);
  });
  trainLoss /= trainData.length;

  // Calculate Test Loss
  let testLoss = 0;
  testData.forEach((point) => {
    let prediction = 0;
    for (let d = 0; d <= degree; d++) {
      prediction += weights[d] * Math.pow(point.x, d);
    }
    testLoss += Math.pow(prediction - point.y, 2);
  });
  testLoss /= testData.length;

  res.json({
    weights,
    trainLoss,
    testLoss,
  });
});

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});