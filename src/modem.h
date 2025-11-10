/**
 * @file modem.h
 * @brief Function declarations for the 4G modem.
 * @date 2025-11-08
 */

#ifndef MODEM_H
#define MODEM_H

// --- THIS IS A CRITICAL FIX ---
// We must include Arduino.h here to get the 'String' type
#include <Arduino.h>

// Function Declarations
bool modem_init();
bool modem_get_gps(float& lat, float& lon);
String modem_get_utc_time();
bool modem_connect_network();
void modem_disconnect();
bool modem_http_post(String payload);

#endif // MODEM_H