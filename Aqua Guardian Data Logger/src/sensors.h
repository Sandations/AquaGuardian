/**
 * @file sensors.h
 * @brief Structs and function declarations for all sensors.
 * @date 2025-11-08
 */

#ifndef SENSORS_H
#define SENSORS_H

// --- THIS IS A CRITICAL FIX ---
// We must include Arduino.h here to get the 'String' type
#include <Arduino.h>
#include "config.h" // Include config for definitions

/**
 * @struct SensorSample
 * @brief Holds a single reading from all sensors.
 */
struct SensorSample {
    String time;       // ISO8601 timestamp
    float ph = -1.0;
    float temp = -999.0;
    float ec = -1.0;
    float turb = -1.0;
    float in_do = -1.0; // "in_do" to avoid "do" keyword
    float orp = -1.0;
};

/**
 * @struct SensorReadings
 * @brief Holds all data for a complete logging session.
 */
struct SensorReadings {
    String session_id;  // Unique ID (usually start timestamp)
    String device_id = DEVICE_ID;
    float gps_lat = 0.0;
    float gps_lon = 0.0;
    bool water_leak = false;
    int sample_count = 0;
    // Use the definitions from config.h for array size
    SensorSample samples[MAX_SAMPLES_PER_SESSION]; 
};

// Function Declarations
bool sensors_init();
void sensors_get_initial_data(SensorReadings& session_data);
void sensors_read_all(SensorSample& sample);

#endif // SENSORS_H