const express = require("express");
const fs = require("fs");
const app = express();
const tf = require("@tensorflow/tfjs-node");
const faceapi = require("@vladmandic/face-api");
const log = require("@vladmandic/pilogger");
const multer = require("multer");
var storage = multer.diskStorage({
  destination: function (req, file, callback) {
    callback(null, "./uploads");
  },
  filename: function (req, file, callback) {
    callback(null, file.originalname);
  },
});

const upload = multer({ storage: storage });
const path = require("path");
const modelPathRoot = "./models";
const minConfidence = 0.15;
const maxResults = 5;
let optionsSSDMobileNet;
let faceMatcher;

app.post("/profile", upload.single("avatar"), async function (req, res) {
  try {
    const tensor = await image(req.file.destination + "/" + req.file.filename);
    const result = await detect(tensor);
    for (const face of result) {
      let a = await faceMatcher.findBestMatch(face.descriptor);
      res.json([print(face), a]);
    }
    tensor.dispose();
  } catch (error) {
    res.json([error]);
  }
});

async function image(input) {
  let buffer;
  log.info("Loading image:", input);
  if (input.startsWith("http:") || input.startsWith("https:")) {
    const res = await fetch(input);
    if (res && res.ok) buffer = await res.buffer();
    else
      console.log(
        "Invalid image URL:",
        input,
        res.status,
        res.statusText,
        res.headers.get("content-type")
      );
  } else {
    buffer = fs.readFileSync(input);
  }

  // decode image using tfjs-node so we don't need external depenencies
  // can also be done using canvas.js or some other 3rd party image library
  // console.log(buffer);
  if (!buffer) return {};
  const tensor = tf.tidy(() => {
    const decode = faceapi.tf.node.decodeImage(buffer, 3);
    let expand;
    if (decode.shape[2] === 4) {
      // input is in rgba format, need to convert to rgb
      const channels = faceapi.tf.split(decode, 4, 2); // tf.split(tensor, 4, 2); // split rgba to channels
      const rgb = faceapi.tf.stack([channels[0], channels[1], channels[2]], 2); // stack channels back to rgb and ignore alpha
      expand = faceapi.tf.reshape(rgb, [
        1,
        decode.shape[0],
        decode.shape[1],
        3,
      ]); // move extra dim from the end of tensor and use it as batch number instead
    } else {
      expand = faceapi.tf.expandDims(decode, 0);
    }
    const cast = faceapi.tf.cast(expand, "float32");
    return cast;
  });
  return tensor;
}

async function main() {
  console.log("....................Loading And Trining Models................");
  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready();
  const modelPath = path.join(__dirname, modelPathRoot);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.ageGenderNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceExpressionNet.loadFromDisk(modelPath);
  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
    minConfidence,
    maxResults,
  });
  const dir = fs.readdirSync("./dataset");
  let datamodal = [];
  for (a = 1; a < dir.length; a++) {
    datamodal[a - 1] = [];
    datamodal[a - 1].push(dir[a]);
    // datamodal.push(dir[a]);
    const inrDir = fs.readdirSync(path.join("./dataset", dir[a]));
    let imageflate32 = [];
    for (const img of inrDir) {
      let imgpath = path.join("./dataset", dir[a], img);
      const tensor = await image(imgpath);
      const result = await detect(tensor);
      imageflate32.push(result[0].descriptor);
    }
    datamodal[a - 1].push(imageflate32);
  }

  const labeledDescriptors = datamodal.map((e) => {
    return new faceapi.LabeledFaceDescriptors(e[0], e[1]);
  });
  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
  console.log("....................Compeleted................");
}

async function detect(tensor) {
  try {
    const result = await faceapi
      .detectAllFaces(tensor, optionsSSDMobileNet)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withFaceDescriptors()
      .withAgeAndGender();
    return result;
  } catch (err) {
    log.error("Caught error", err.message);
    return [];
  }
}

function print(face) {
  const expression = Object.entries(face.expressions).reduce(
    (acc, val) => (val[1] > acc[1] ? val : acc),
    ["", 0]
  );
  const box = [
    face.alignedRect._box._x,
    face.alignedRect._box._y,
    face.alignedRect._box._width,
    face.alignedRect._box._height,
  ];
  const gender = `Gender: ${Math.round(100 * face.genderProbability)}% ${
    face.gender
  }`;
  log.data(
    `Detection confidence: ${Math.round(
      100 * face.detection._score
    )}% ${gender} Age: ${
      Math.round(10 * face.age) / 10
    } Expression: ${Math.round(100 * expression[1])}% ${
      expression[0]
    } Box: ${box.map((a) => Math.round(a))}`
  );
  return `Detection confidence: ${Math.round(
    100 * face.detection._score
  )}% ${gender} Age: ${Math.round(10 * face.age) / 10} Expression: ${Math.round(
    100 * expression[1]
  )}% ${expression[0]} Box: ${box.map((a) => Math.round(a))}`;
}

main();
const port = 4000;
app.listen(port, () => console.log(`Example app listening on port ${port}!`));
