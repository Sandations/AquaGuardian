/**
 * @file sensors.cpp
 * @brief Implementation file for all sensor interactions.
 * * Includes DS18B20, pH, DO, EC, Turbidity, and ORP.
 * * @version 3.0 - FINAL SYNC
 * @date 2025-11-08
 */

// --- THIS IS A CRITICAL FIX ---
// config.h must be the first file included in every .cpp
#include "config.h" 

#include "sensors.h"
#include "modem.h" // For modem_get_utc_time()
#include <OneWire.h>
#include <DallasTemperature.h>

// --- Sensor Libraries Setup ---
// DS18B20 Temperature Sensor
OneWire oneWire(PIN_ONE_WIRE_BUS);
DallasTemperature dallas_temp(&oneWire);

// DFRobot Analog Sensor Calibration
// TODO: Calibrate these values
#define PH_OFFSET 0.0
#define DO_OFFSET 0.0
#define EC_OFFSET 0.0
#define TURB_OFFSET 0.0
#define ORP_OFFSET 0.0


/**
 * @brief Initializes all sensors.
 */
bool sensors_init() {
    Serial.println("Initializing sensors...");
    
    // Initialize DS18B20
    dallas_temp.begin();
    
    // Set ADC pins to INPUT
    pinMode(PIN_PH, INPUT);
    pinMode(PIN_DO, INPUT);
    pinMode(PIN_EC, INPUT);
    pinMode(PIN_TURBIDITY, INPUT);
    pinMode(PIN_ORP, INPUT);
    pinMode(PIN_WATER_LEAK, INPUT);

    Serial.println("Sensors initialized.");
    return true;
}

/**
 * @brief Gets initial, one-time data at the start of a session.
 */
void sensors_get_initial_data(SensorReadings& session_data) {
    Serial.println("Getting initial data (GPS, Time, Leak)...");
    
    // Use current time as session ID
    session_data.session_id = modem_get_utc_time();

    // Get GPS fix (if available)
    float lat, lon;
    if (modem_get_gps(lat, lon)) {
        Serial.printf("GPS Fix: %.4f, %.4f\n", lat, lon);
        session_data.gps_lat = lat;
        session_data.gps_lon = lon;
    } else {
        Serial.println("No GPS fix.");
    }

    // Check water leak sensor
    session_data.water_leak = (digitalRead(PIN_WATER_LEAK) == HIGH);
    if(session_data.water_leak) {
        Serial.println("WARNING: Water leak detected!");
    }

    session_data.sample_count = 0;
}

/**
 * @brief Reads all sensors and populates a SensorSample struct.
 */
void sensors_read_all(SensorSample& sample) {
    // --- Read DS18B20 Temperature ---
    dallas_temp.requestTemperatures(); 
    sample.temp = dallas_temp.getTempCByIndex(0);
    if (sample.temp == DEVICE_DISCONNECTED_C) {
        Serial.println("Error: DS18B20 disconnected.");
        sample.temp = -999.0;
    }

    // --- Read Analog Sensors ---
    // Note: These are placeholder calculations.
    // You MUST replace them with proper calibration curves.

    // Read pH
    int ph_analog = analogRead(PIN_PH);
    float ph_voltage = ph_analog * (DEFAULT_VREF / 4095.0);
    sample.ph = (ph_voltage * 3.5) + PH_OFFSET; // EXAMPLE CALCULATION
    
    // Read Dissolved Oxygen
    int do_analog = analogRead(PIN_DO);
    float do_voltage = do_analog * (DEFAULT_VREF / 4095.0);
    sample.in_do = (do_voltage * 10.0) + DO_OFFSET; // EXAMPLE CALCULATION

    // Read EC
    int ec_analog = analogRead(PIN_EC);
    float ec_voltage = ec_analog * (DEFAULT_VREF / 4095.0);
    sample.ec = (ec_voltage * 2.5) + EC_OFFSET; // EXAMPLE CALCULATION

    // Read Turbidity
    int turb_analog = analogRead(PIN_TURBIDITY);
    float turb_voltage = turb_analog * (DEFAULT_VREF / 4095.0);
    sample.turb = (turb_voltage * 1.0) + TURB_OFFSET; // EXAMPLE CALCULATION

    // Read ORP
    int orp_analog = analogRead(PIN_ORP);
    float orp_voltage = orp_analog * (DEFAULT_VREF / 4095.0);
    sample.orp = (orp_voltage * 1.0) + ORP_OFFSET; // EXAMPLE CALCULATION

    // Print values
    Serial.printf("  Temp: %.2fC, pH: %.2f, DO: %.2f, EC: %.2f, Turb: %.2f, ORP: %.2f\n",
        sample.temp, sample.ph, sample.in_do, sample.ec, sample.turb, sample.orp);
}