/**
 * @file main.cpp
 * @brief Main firmware for the Aqua Guardian Data Logger.
 * * Manages sleep cycles, data logging, and data uploading.
 * * @version 3.0 - FINAL SYNC
 * @date 2025-11-08
 */

// --- THIS IS A CRITICAL FIX ---
// config.h must be the first file included in every .cpp
#include "config.h" 

#include <Arduino.h>
#include "modem.h"
#include "sensors.h"
#include "storage.h"

// --- Global Variable Definitions ---
// (Moved from config.h to fix "multiple definition" linker errors)
const char *server = "your-api-endpoint.com";
const char *resource = "/api/v1/data";
const int port = 443; // Use 443 for HTTPS, 80 for HTTP
const char *api_key_header = "x-api-key";
const char *api_key_value = "YOUR_SECRET_API_KEY";
const char *apn = "your_apn";       // e.g., "iot.1nce.net", "hologram", "tm"
const char *gprsUser = "";          // GPRS username (if any)
const char *gprsPass = "";          // GPRS password (if any)
// -----------------------------------

// Define the two main tasks
void run_logging_window();
void run_upload_window();
void go_to_sleep();

// Global data object for the current session
SensorReadings session_data;


void setup() {
    Serial.begin(115200);
    delay(1000); // Wait for serial
    Serial.println("\n--- AQUA GUARDIAN DATA LOGGER ---");
    Serial.printf("Device ID: %s\n", DEVICE_ID);

    // Initialize filesystem
    storage_init();
    storage_list_files();

    // Initialize sensors
    if (!sensors_init()) {
        Serial.println("Sensor initialization FAILED! Sleeping...");
        go_to_sleep();
    }

    // Initialize modem
    if (!modem_init()) {
        Serial.println("Modem initialization FAILED! Proceeding to log data...");
        // Don't sleep; we can still log data even if modem fails
    }

    // Get initial data (GPS, water leak, session ID)
    sensors_get_initial_data(session_data);

    // Run the two main tasks
    run_logging_window();
    run_upload_window();

    // End of cycle
    Serial.println("Tasks complete.");
    go_to_sleep();
}


/**
 * @brief Main logging loop. Runs for a fixed duration, sampling sensors.
 */
void run_logging_window() {
    Serial.println("\n--- STARTING LOGGING WINDOW ---");
    
    // Calculate total logging duration from config
    const unsigned long logging_duration_ms = MAX_SAMPLES_PER_SESSION * SENSOR_READ_INTERVAL_MS;
    unsigned long logging_start_time = millis();
    unsigned long next_sample_time = logging_start_time;

    session_data.sample_count = 0;

    while (millis() - logging_start_time < logging_duration_ms) {
        if (millis() >= next_sample_time) {
            
            Serial.printf("Taking sample %d/%d...\n", session_data.sample_count + 1, MAX_SAMPLES_PER_SESSION);
            
            SensorSample sample;
            sample.time = modem_get_utc_time(); // Get timestamp
            sensors_read_all(sample); // Read all sensors
            
            // Add sample to the session data
            if (session_data.sample_count < MAX_SAMPLES_PER_SESSION) {
                session_data.samples[session_data.sample_count] = sample;
                session_data.sample_count++;
            }

            // Schedule next sample
            next_sample_time += SENSOR_READ_INTERVAL_MS;
        }
        // Small delay to prevent busy-looping
        delay(10); 
    }

    // Save session data to a file
    if (session_data.sample_count > 0) {
        Serial.println("Logging window complete. Saving session...");
        String filename = storage_save_session(session_data);
        if (filename.length() > 0) {
            Serial.printf("Session saved to: %s\n", filename.c_str());
        } else {
            Serial.println("Failed to save session!");
        }
    } else {
        Serial.println("No samples taken, nothing to save.");
    }
}

/**
 * @brief Main upload loop. Tries to upload all saved files.
 */
void run_upload_window() {
    Serial.println("\n--- STARTING UPLOAD WINDOW ---");
    
    // Prune old files before uploading
    storage_prune_old_files();

    if (!modem_connect_network()) {
        Serial.println("Failed to connect to network. Skipping uploads.");
        return;
    }

    Serial.println("Network connected. Checking for files to upload...");
    
    File root = FILESYSTEM.open("/");
    File file = root.openNextFile();

    while(file) {
        String filename = file.name();
        // Check if it's a session file
        if (filename.startsWith("/session_")) {
            Serial.printf("Found file: %s (%d bytes)\n", filename.c_str(), file.size());

            // Read the file content
            String payload = storage_read_file(filename);
            
            if (payload.length() > 0) {
                Serial.println("Attempting to upload...");
                if (modem_http_post(payload)) {
                    Serial.println("Upload SUCCESSFUL.");
                    // Delete file after successful upload
                    storage_delete_file(filename); 
                    Serial.printf("Deleted file: %s\n", filename.c_str());
                } else {
                    Serial.println("Upload FAILED. Will retry next cycle.");
                }
            } else {
                Serial.println("File is empty, deleting.");
                storage_delete_file(filename);
            }
        }
        file.close(); // Close the current file
        file = root.openNextFile(); // Open the next file
    }
    root.close(); // Close the root directory

    Serial.println("Upload window complete.");
    modem_disconnect();
}


/**
 * @brief Enters deep sleep for the configured duration.
 */
void go_to_sleep() {
    Serial.printf("Entering deep sleep for %d seconds.\n", DEEP_SLEEP_DURATION_SEC);
    Serial.flush();
    
    // Configure timer wakeup
    esp_sleep_enable_timer_wakeup(DEEP_SLEEP_DURATION_SEC * 1000000ULL);
    
    // Enter deep sleep
    esp_deep_sleep_start();
}


/**
 * @brief Empty loop function (required by Arduino).
 * The device spends all its time in setup() or deep_sleep().
 */
void loop() {
    // This should never be reached :/
}