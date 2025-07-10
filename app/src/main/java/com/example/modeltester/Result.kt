package com.example.modeltester

import android.graphics.RectF

// Holds the final, processed result for a single detected object instance
data class Result(
    val classIndex: Int,
    val score: Float,
    val boundingBox: RectF,
    val mask: FloatArray // Holds the 2D float array of the final mask
)