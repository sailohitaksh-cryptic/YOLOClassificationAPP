package com.example.modeltester

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ObjectDetectorHelper(
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {
    private var objectDetector: ObjectDetector? = null
    private lateinit var backgroundExecutor: ExecutorService

    init {
        setupObjectDetector()
    }

    private fun setupObjectDetector() {
        backgroundExecutor = Executors.newSingleThreadExecutor()
        backgroundExecutor.execute {
            try {
                val baseOptions = BaseOptions.builder().setNumThreads(4).build()
                val options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setScoreThreshold(0.5f)
                    .setIncludeSegmentationMasks(true)
                    .build()
                objectDetector = ObjectDetector.createFromFileAndOptions(context, "abdomensegmentation.tflite", options)
            } catch (e: Exception) {
                objectDetectorListener?.onError("Failed to initialize model: ${e.message}")
                Log.e("ObjectDetectorHelper", "TFLite failed to load model with error: " + e.message)
            }
        }
    }

    fun detect(image: Bitmap) {
        if (objectDetector == null) {
            setupObjectDetector()
            return
        }

        backgroundExecutor.execute {
            val inferenceTime = SystemClock.uptimeMillis()
            val tensorImage = TensorImage.fromBitmap(image)
            val results: List<Detection>? = objectDetector?.detect(tensorImage)
            val finalInferenceTime = SystemClock.uptimeMillis() - inferenceTime

            objectDetectorListener?.onResults(
                results,
                finalInferenceTime
            )
        }
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: List<Detection>?,
            inferenceTime: Long
        )
    }
}