/**
 * @file modem.cpp
 * @brief Implementation file for A7670E 4G Modem.
 * * Uses TinyGSM library to interact with the modem via AT commands.
 * Handles initialization, GPS, network connection, and HTTP POST.
 * * @version 10.0 - Replaced GPS functions with raw AT commands
 * @date 2025-11-09
 */

// --- HEADER INCLUDES ---
// config.h must be the first file included in every .cpp
#include "config.h" 

// This include order is critical.
#include <TinyGsm.h>
#include <TinyGsmClient.h>
#include "modem.h"

// --- TinyGSM Setup (from config.h) ---
HardwareSerial SerialAT(SERIAL_AT_PORT);

TinyGsm modem(SerialAT);
TinyGsmClient client(modem);


/**
 * @brief Initializes the modem.
 */
bool modem_init() {
    Serial.println("Initializing modem...");

    // Set modem AT command port baud rate
    SerialAT.begin(SERIAL_AT_BAUD, SERIAL_8N1, PIN_MODEM_RX, PIN_MODEM_TX);
    
    Serial.println("Waiting for modem to respond...");
    if (!modem.restart()) {
        Serial.println("Failed to restart modem!");
        return false;
    }

    String modemInfo = modem.getModemInfo();
    Serial.printf("Modem Info: %s\n", modemInfo.c_str());

    // Enable GPS
    Serial.println("Enabling GPS... (This may take a moment)");
    
    // --- FIX ---
    // The A7672X library does not have a simple GPS function.
    // We must send the raw AT command to power on the GNSS (GPS) module.
    modem.sendAT(GF("AT+CGNSPWR=1"));
    if (modem.waitResponse() != 1) {
        Serial.println("Failed to power on GNSS/GPS!");
    }
    
    return true;
}

/**
 * @brief Gets GPS coordinates.
 */
bool modem_get_gps(float& lat, float& lon) {
    
    // --- FIX ---
    // The A7672X library does not have a simple getGPS function.
    // We must send AT+CGNSINF and manually parse the response.

    String res;
    // Send AT command to get GNSS info
    modem.sendAT(GF("AT+CGNSINF"));
    
    // Wait for the response, which starts with "+CGNSINF:"
    if (modem.waitResponse(10000L, GF("+CGNSINF:")) != 1) {
        Serial.println("Failed to get GNSS info response");
        return false;
    }
    
    // Read the full response line, e.g.:
    // +CGNSINF: 1,1,20251109004500.000,40.7128,-74.0060,...
    res = modem.stream.readStringUntil('\r');
    modem.waitResponse(); // Wait for the final "OK"

    // Serial.println(res); // Uncomment this line for debugging

    // --- Manually parse the comma-separated string ---
    
    // Find the first comma
    int firstComma = res.indexOf(',');
    if (firstComma == -1) { return false; }

    // Find the second comma (after fix status)
    int secondComma = res.indexOf(',', firstComma + 1);
    if (secondComma == -1) { return false; }
    
    // Extract fix status (the number between first and second comma)
    // 1 = Fix, 0 = No Fix
    String fixStatusStr = res.substring(firstComma + 1, secondComma);
    if (fixStatusStr.toInt() != 1) {
        Serial.println("No GPS fix.");
        return false; // No fix
    }

    // Find the third comma (after UTC)
    int thirdComma = res.indexOf(',', secondComma + 1);
    if (thirdComma == -1) { return false; }

    // Find the fourth comma (after latitude)
    int fourthComma = res.indexOf(',', thirdComma + 1);
    if (fourthComma == -1) { return false; }

    // Extract latitude
    String latStr = res.substring(thirdComma + 1, fourthComma);
    lat = latStr.toFloat();

    // Find the fifth comma (after longitude)
    int fifthComma = res.indexOf(',', fourthComma + 1);
    if (fifthComma == -1) { return false; }

    // Extract longitude
    String lonStr = res.substring(fourthComma + 1, fifthComma);
    lon = lonStr.toFloat();

    // Check for 0,0
    if (lat == 0.0 && lon == 0.0) {
        return false;
    }

    return true;
}

/**
 * @brief Gets the current UTC time from the cellular network.
 */
String modem_get_utc_time() {
    int year, month, day, hour, min, sec;
    float timezone;
    if (modem.getNetworkTime(&year, &month, &day, &hour, &min, &sec, &timezone)) {
        char iso_time[25];
        // Format as ISO8601: "YYYY-MM-DDTHH:MM:SSZ"
        snprintf(iso_time, sizeof(iso_time), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                 year, month, day, hour, min, sec);
        return String(iso_time);
    } else {
        // Fallback: return time based on millis()
        // This is NOT UTC, but it's better than nothing for timestamps
        unsigned long now = millis();
        char fallback_time[25];
        snprintf(fallback_time, sizeof(fallback_time), "T%luS", now / 1000);
        return String(fallback_time);
    }
}

/**
 * @brief Connects to the GPRS network.
 */
bool modem_connect_network() {
    Serial.print("Waiting for network... ");
    if (!modem.waitForNetwork()) {
        Serial.println("FAIL");
        return false;
    }
    Serial.println("OK");

    Serial.printf("Connecting to GPRS: %s... ", apn);
    if (!modem.gprsConnect(apn, gprsUser, gprsPass)) {
        Serial.println("FAIL");
        return false;
    }
    Serial.println("OK");
    return true;
}

/**
 * @brief Disconnects from the GPRS network.
 */
void modem_disconnect() {
    modem.gprsDisconnect();
    Serial.println("GPRS Disconnected.");
}

/**
 * @brief Performs an HTTP POST request.
 */
bool modem_http_post(String payload) {
    Serial.printf("Connecting to server: %s...", server);
    if (!client.connect(server, port)) {
        Serial.println("FAIL");
        return false;
    }
    Serial.println("OK");

    // Send HTTP POST request
    client.print(String("POST ") + resource + " HTTP/1.1\r\n");
    client.print(String("Host: ") + server + "\r\n");
    client.print("Connection: close\r\n");
    client.print(String("User-Agent: ") + DEVICE_ID + "\r\n");
    client.print(String(api_key_header) + ": " + api_key_value + "\r\n");
    client.print("Content-Type: application/json\r\n");
    client.print(String("Content-Length: ") + payload.length() + "\r\n");
    client.print("\r\n");
    client.print(payload);

    // Wait for response
    unsigned long timeout = millis();
    while (client.connected() && millis() - timeout < 5000L) {
        // Wait
    }

    // Read response
    String responseHeader = "";
    while (client.available()) {
        char c = client.read();
        responseHeader += c;
        if (c == '\n' && responseHeader.endsWith("\r\n\r\n")) {
            break; // End of headers
        }
    }

    // Check for "200 OK"
    if (responseHeader.indexOf("HTTP/1.1 200 OK") != -1) {
        // Serial.println("OK");
    } else {
        Serial.printf("\n--- SERVER ERROR ---\n%s\n--------------------\n", responseHeader.c_str());
        client.stop();
        return false;
    }

    // Added proper client stop for 200 OK
    client.stop();
    return true;
}