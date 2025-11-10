/**
 * @file storage.cpp
 * @brief Implementation file for LittleFS storage.
 * * Handles saving, reading, and deleting session files.
 * * @version 3.0 - FINAL SYNC
 * @date 2025-11-08
 */

// --- THIS IS A CRITICAL FIX ---
// config.h must be the first file included in every .cpp
#include "config.h" 

#include "storage.h"
#include <ArduinoJson.h>

/**
 * @brief Initializes the LittleFS filesystem.
 */
void storage_init() {
    Serial.println("Initializing Filesystem...");
    if (!FILESYSTEM.begin(false)) { // false = do not format on fail
        Serial.println("Filesystem mount failed! Formatting...");
        if (!FILESYSTEM.begin(true)) { // true = format if mount fails
            Serial.println("Formatting FAILED! Filesystem unavailable.");
            return;
        }
    }
    Serial.println("Filesystem initialized.");
}

/**
 * @brief Saves a SensorReadings session to a JSON file.
 */
String storage_save_session(SensorReadings& readings) {
    if (readings.sample_count == 0) {
        return ""; // Don't save empty sessions
    }

    String filename = "/session_" + readings.session_id + ".json";
    filename.replace(":", "-"); // Sanitize filename
    
    File file = FILESYSTEM.open(filename, FILE_WRITE);
    if (!file) {
        Serial.printf("Failed to open file for writing: %s\n", filename.c_str());
        return "";
    }

    // Create JSON document
    JsonDocument doc;
    doc["session_id"] = readings.session_id;
    doc["device_id"] = readings.device_id;
    doc["water_leak"] = readings.water_leak;

    JsonObject gps = doc["gps"].to<JsonObject>();
    gps["lat"] = readings.gps_lat;
    gps["lon"] = readings.gps_lon;

    JsonArray samples = doc["samples"].to<JsonArray>();
    for (int i = 0; i < readings.sample_count; i++) {
        JsonObject s = samples.add<JsonObject>();
        s["time"] = readings.samples[i].time;
        s["pH"] = readings.samples[i].ph;
        s["temp"] = readings.samples[i].temp;
        s["EC"] = readings.samples[i].ec;
        s["turbidity"] = readings.samples[i].turb;
        s["DO"] = readings.samples[i].in_do;
        s["ORP"] = readings.samples[i].orp;
    }

    // Serialize JSON to file
    if (serializeJson(doc, file) == 0) {
        Serial.println("Failed to write JSON to file.");
        file.close();
        return "";
    }

    file.close();
    return filename;
}

/**
 * @brief Reads the content of a file as a String.
 */
String storage_read_file(String filename) {
    File file = FILESYSTEM.open(filename, FILE_READ);
    if (!file) {
        Serial.printf("Failed to open file for reading: %s\n", filename.c_str());
        return "";
    }
    String content = file.readString();
    file.close();
    return content;
}

/**
 * @brief Deletes a file from the filesystem.
 */
bool storage_delete_file(String filename) {
    if (FILESYSTEM.remove(filename)) {
        return true;
    } else {
        Serial.printf("Failed to delete file: %s\n", filename.c_str());
        return false;
    }
}

/**
 * @brief Counts the number of files in the root directory.
 */
int storage_get_file_count() {
    int count = 0;
    File root = FILESYSTEM.open("/");
    File file = root.openNextFile();
    while (file) {
        if (!file.isDirectory()) {
            count++;
        }
        file.close();
        file = root.openNextFile();
    }
    root.close();
    return count;
}

/**
 * @brief Lists all files in the root directory to the Serial monitor.
 */
void storage_list_files() {
    Serial.println("--- FILE LIST ---");
    File root = FILESYSTEM.open("/");
    File file = root.openNextFile();
    while (file) {
        if (!file.isDirectory()) {
            Serial.printf("  %s (%d bytes)\n", file.name(), file.size());
        }
        file.close();
        file = root.openNextFile();
    }
    root.close();
    Serial.println("-----------------");
}

/**
 * @brief Deletes the oldest files if the file count exceeds the limit.
 */
void storage_prune_old_files() {
    int file_count = storage_get_file_count();
    Serial.printf("File count: %d (Max: %d)\n", file_count, MAX_SESSIONS_STORED);
    
    if (file_count <= MAX_SESSIONS_STORED) {
        return; // No pruning needed
    }

    int files_to_delete = file_count - MAX_SESSIONS_STORED;
    Serial.printf("Pruning %d oldest files...\n", files_to_delete);
    
    for (int i = 0; i < files_to_delete; i++) {
        File root = FILESYSTEM.open("/");
        File file = root.openNextFile();
        String oldestFile = "";
        time_t oldestTime = 0xFFFFFFFF; // Max time_t value

        while(file) {
            if (!file.isDirectory()) {
                time_t write_time = file.getLastWrite();
                if (write_time < oldestTime) {
                    oldestTime = write_time;
                    oldestFile = file.name();
                }
            }
            file.close();
            file = root.openNextFile();
        }
        root.close();

        if (oldestFile.length() > 0) {
            Serial.printf("Deleting: %s\n", oldestFile.c_str());
            storage_delete_file(oldestFile);
        }
    }
}