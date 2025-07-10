package com.example.modeltester

import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.modeltester.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.opencv.dnn.Dnn

class MainActivity : AppCompatActivity() {

    // --- ⚙️ CONFIGURATION (Matches Python Script) --- //
    companion object {
        private const val CLASSIFIER_MODEL_FILE = "speciesaiedge.tflite"
        private const val CLASSIFIER_LABELS_FILE = "species_classifier.txt"
        private const val CLASSIFIER_INPUT_SIZE = 300

        private const val DETECTOR_MODEL_FILE = "YOLO_08_30-fp16.tflite"
        private const val DETECTOR_INPUT_WIDTH = 480
        private const val DETECTOR_INPUT_HEIGHT = 640
        private const val YOLO_CONFIDENCE_THRESHOLD = 0.8f
        private const val IOU_THRESHOLD = 0.5f
    }
    // ----------------------------------------------------

    private lateinit var binding: ActivityMainBinding
    private var currentBitmap: Bitmap? = null
    private var cameraImageUri: Uri? = null

    private lateinit var classifierInterpreter: Interpreter
    private lateinit var detectorInterpreter: Interpreter
    private lateinit var classifierLabels: List<String>

    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { handleImage(it) }
    }

    private val captureImageLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) { cameraImageUri?.let { handleImage(it) } } else { showError("Failed to capture image.") }
    }

    override fun onCreate(savedInstanceState: Bundle?) {



        super.onCreate(savedInstanceState)

        if (OpenCVLoader.initLocal()) {
            Log.i("opencv", "OpenCV loaded successfully");
        } else {
            Log.e("opencv", "OpenCV initialization failed!");
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupInterpreters()
        binding.selectImageButton.setOnClickListener { selectImageLauncher.launch("image/*") }
        binding.captureImageButton.setOnClickListener {
            cameraImageUri = createImageUri()
            captureImageLauncher.launch(cameraImageUri)
        }


    }

    private fun handleImage(uri: Uri) {
        try {
            val source = ImageDecoder.createSource(this.contentResolver, uri)
            val bitmap = ImageDecoder.decodeBitmap(source).copy(Bitmap.Config.ARGB_8888, true)
            currentBitmap = bitmap
            binding.imageViewResult.setImageBitmap(currentBitmap)
            currentBitmap?.let { runFullPipeline(it) }
        } catch (e: Exception) {
            showError("Failed to load image: ${e.message}")
        }
    }

    private fun createImageUri(): Uri {
        val imageFile = File(applicationContext.cacheDir, "camera_photo.jpg")
        return FileProvider.getUriForFile(applicationContext, "${applicationContext.packageName}.provider", imageFile)
    }

    private fun setupInterpreters() {
        Executors.newSingleThreadExecutor().execute {
            try {
                val classifierModelBuffer = FileUtil.loadMappedFile(this, CLASSIFIER_MODEL_FILE)
                classifierInterpreter = Interpreter(classifierModelBuffer, Interpreter.Options())
                classifierLabels = FileUtil.loadLabels(this, CLASSIFIER_LABELS_FILE)

                val detectorModelBuffer = FileUtil.loadMappedFile(this, DETECTOR_MODEL_FILE)
                detectorInterpreter = Interpreter(detectorModelBuffer, Interpreter.Options())

                runOnUiThread { Toast.makeText(this, "All models loaded", Toast.LENGTH_SHORT).show() }
            } catch (e: Exception) {
                showError("Failed to initialize models: ${e.message}")
            }
        }
    }



    private fun runFullPipeline(bitmap: Bitmap) {
        Executors.newSingleThreadExecutor().execute {
            try {
                // Step 1: Run YOLO detection using OpenCV
                val detectionResult = runYoloDetectionWithOpenCV(bitmap)
                if (detectionResult == null) {
                    showError("No object detected.")
                    return@execute
                }

                // Step 2: Crop the original image
                val (y1, x1, y2, x2) = detectionResult.boundingBox
                Log.d("YOLO_DEBUG", "Final Crop Coords (y1,x1,y2,x2): [${y1}, ${x1}, ${y2}, ${x2}]")
                if (y1 < 0 || x1 < 0 || y2 > bitmap.height || x2 > bitmap.width || y1 >= y2 || x1 >= x2) {
                    showError("Invalid bounding box.")
                    return@execute
                }
                val croppedBitmap = Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1)

                // Step 3: Preprocess cropped image for classifier using OpenCV
                val imageByteBuffer = preprocessForClassifierWithOpenCV(croppedBitmap)

                // Step 4 & 5: Run Classifier and Process Results
                val outputBuffer = ByteBuffer.allocateDirect(1 * classifierLabels.size * 4).apply { order(ByteOrder.nativeOrder()) }
                classifierInterpreter.run(imageByteBuffer, outputBuffer)

                outputBuffer.rewind()
                val rawScores = FloatArray(classifierLabels.size)
                outputBuffer.asFloatBuffer().get(rawScores)
                // --- DEBUG: Log the final raw logits from the classifier ---
                Log.d("LOGITS_DEBUG", "Final Classifier RAW LOGITS: ${rawScores.joinToString()}")
//                // ---------------------------------------------------------
                val classificationResultText = processClassifierScores(rawScores)

                runOnUiThread {
                    binding.imageViewResult.setImageBitmap(croppedBitmap)
                    binding.textViewClassification.text = classificationResultText
                    binding.textViewClassification.visibility = View.VISIBLE
                }
            } catch (e: Exception) {
                showError("Error during pipeline: ${e.message}")
            }
        }
    }

    data class DetectionResult(val boundingBox: IntArray, val confidence: Float)

    private fun runYoloDetectionWithOpenCV(bitmap: Bitmap): DetectionResult? {
        // --- 1. PREPARE IMAGE ---
        val originalMat = Mat()
        Utils.bitmapToMat(bitmap, originalMat)
        Imgproc.cvtColor(originalMat, originalMat, Imgproc.COLOR_RGBA2BGR)
        val originalW = originalMat.cols().toFloat()
        val originalH = originalMat.rows().toFloat()
        val resizedMat = Mat()
        Imgproc.resize(originalMat, resizedMat, Size(DETECTOR_INPUT_WIDTH.toDouble(), DETECTOR_INPUT_HEIGHT.toDouble()))
        val h = resizedMat.rows()
        val w = resizedMat.cols()
        val size = max(h, w)
        val paddedMat = Mat(size, size, resizedMat.type(), Scalar(0.0, 0.0, 0.0))
        val roi = paddedMat.submat(Rect((size - w) / 2, (size - h) / 2, w, h))
        resizedMat.copyTo(roi)
        paddedMat.convertTo(paddedMat, CvType.CV_32F, 1.0 / 255.0)
        val inputData = ByteBuffer.allocateDirect(1 * size * size * 3 * 4).apply { order(ByteOrder.nativeOrder()) }
        val floatArray = FloatArray(size * size * 3)
        paddedMat.get(0, 0, floatArray)
        inputData.asFloatBuffer().put(floatArray)
        inputData.rewind()

        // --- 2. RUN INFERENCE ---
        val outputBuffer = Array(1) { Array(25200) { FloatArray(6) } }
        detectorInterpreter.run(inputData, outputBuffer)

        // --- 3. PREPARE FOR NMS ---
        val predictions = outputBuffer[0].filter { it[4] > YOLO_CONFIDENCE_THRESHOLD }
        if (predictions.isEmpty()) return null

        val boxesForNms = mutableListOf<Rect2d>()
        val scoresForNms = mutableListOf<Float>()
        for (pred in predictions) {
            val xCenter = pred[0]
            val yCenter = pred[1]
            val width = pred[2]
            val height = pred[3]
            val x1 = (xCenter - width / 2) * 640
            val y1 = (yCenter - height / 2) * 640
            boxesForNms.add(Rect2d(x1.toDouble(), y1.toDouble(), (width * 640).toDouble(), (height * 640).toDouble()))
            scoresForNms.add(pred[4])
        }

        // --- 4. APPLY NMS ---
        val boxesMat = MatOfRect2d(*boxesForNms.toTypedArray())
        val scoresMat = MatOfFloat(*scoresForNms.toFloatArray())
        val indicesMat = MatOfInt()
        Dnn.NMSBoxes(boxesMat, scoresMat, YOLO_CONFIDENCE_THRESHOLD, IOU_THRESHOLD, indicesMat)
        if (indicesMat.empty()) return null

        // --- 5. GET BEST PREDICTION FROM NMS RESULT ---
        // This is the corrected logic. We use the first index from the NMS result.
        val bestIndex = indicesMat.toArray()[0]
        val bestPrediction = predictions[bestIndex]

        // --- 6. CALCULATE FINAL COORDINATES ---
        var xCenter = bestPrediction[0]
        var yCenter = bestPrediction[1]
        var width = bestPrediction[2]
        var height = bestPrediction[3]

        xCenter -= 0.125f
        width *= (640f / 480f)
        xCenter *= originalW * (640f / 480f)
        yCenter *= originalH
        width *= originalW
        height *= originalH

        val y1 = (yCenter - height / 2).toInt().coerceAtLeast(0)
        val x1 = (xCenter - width / 2).toInt().coerceAtLeast(0)
        val y2 = (yCenter + height / 2).toInt().coerceAtMost(originalH.toInt())
        val x2 = (xCenter + width / 2).toInt().coerceAtMost(originalW.toInt())

        return DetectionResult(intArrayOf(y1, x1, y2, x2), bestPrediction[4])
    }

    // --- THIS FUNCTION IS NOW A PERFECT 1-to-1 MATCH OF THE PYTHON SCRIPT ---
    private fun preprocessForClassifierWithOpenCV(bitmap: Bitmap): ByteBuffer {
        val originalMat = Mat()
        Utils.bitmapToMat(bitmap, originalMat)
        Imgproc.cvtColor(originalMat, originalMat, Imgproc.COLOR_RGBA2RGB)

        // 1. Padding
        val h = originalMat.rows()
        val w = originalMat.cols()
        val size = max(h, w)
        val paddedMat = Mat.zeros(size, size, originalMat.type())
        val roi = paddedMat.submat((size - h) / 2, (size - h) / 2 + h, (size - w) / 2, (size - w) / 2 + w)
        originalMat.copyTo(roi)

        // 2. Resize
        val resizedMat = Mat()
        Imgproc.resize(paddedMat, resizedMat, Size(CLASSIFIER_INPUT_SIZE.toDouble(), CLASSIFIER_INPUT_SIZE.toDouble()))

        // 3. Normalize
        resizedMat.convertTo(resizedMat, CvType.CV_32F, 1.0 / 255.0)
        val meanMat = Mat(resizedMat.size(), CvType.CV_32FC3, Scalar(0.485, 0.456, 0.406))
        val stdMat = Mat(resizedMat.size(), CvType.CV_32FC3, Scalar(0.229, 0.224, 0.225))
        Core.subtract(resizedMat, meanMat, resizedMat)
        Core.divide(resizedMat, stdMat, resizedMat)

        // 4. Transpose to CHW and load into ByteBuffer
        val floatArray = FloatArray(CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE * 3)
        resizedMat.get(0, 0, floatArray)
        val byteBuffer = ByteBuffer.allocateDirect(floatArray.size * 4).apply { order(ByteOrder.nativeOrder()) }
        for (channel in 0..2) {
            for (i in 0 until CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE) {
                byteBuffer.putFloat(floatArray[i * 3 + channel])
            }
        }
        byteBuffer.rewind()
        return byteBuffer
    }


    private fun processClassifierScores(rawScores: FloatArray): String {
        val probabilities = softmax(rawScores)
        val topScores = probabilities.mapIndexed { index, score -> Pair(index, score) }
            .sortedByDescending { it.second }
            .take(3)
        return topScores.joinToString("\n") { (index, score) ->
            "${classifierLabels[index]}: ${String.format("%.2f", score * 100)}%"
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0.0f
        val exps = logits.map { exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    private fun showError(message: String) {
        runOnUiThread { Toast.makeText(this, message, Toast.LENGTH_LONG).show() }
    }
}