/**
 * @file storage.h
 * @brief Function declarations for LittleFS storage.
 * @date 2025-11-08
 */

#ifndef STORAGE_H
#define STORAGE_H

// --- THIS IS A CRITICAL FIX ---
// Include Arduino.h for String and LittleFS.h for File
#include <Arduino.h>
#include <LittleFS.h>
#include "sensors.h" // For the SensorReadings struct

// Define the filesystem to use
#define FILESYSTEM LittleFS

// Function Declarations
void storage_init();
String storage_save_session(SensorReadings& readings);
String storage_read_file(String filename);
bool storage_delete_file(String filename);
int storage_get_file_count();
void storage_list_files();
void storage_prune_old_files();

#endif // STORAGE_H