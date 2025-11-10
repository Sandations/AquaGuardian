Wiring & Pin Mapping (with Gravity IO Shield)

This guide assumes you are using the Gravity: IO Shield for FireBeetle 2 (SKU: DFR0762). All wiring is plug-and-play using DFRobot Gravity cables.

CRITICAL VOLTAGE WARNING:
The IO Shield has a VCC slide switch (3.3V / 5V). The FireBeetle 2 ESP32-E microcontroller is NOT 5V TOLERANT.

You MUST set this switch to 3.3V.

Setting the switch to 5V will permanently destroy the ESP32-E board.

Core Module: A7670E Modem

|

| A7670E Pin | Shield Port | ESP32-E Pin |
| TX | UART (RX) | 16 (PIN_MODEM_RX) |
| RX | UART (TX) | 17 (PIN_MODEM_TX) |
| VCC | UART (VCC) | 3.3V (from switch) |
| GND | UART (GND) | GND |
| PWR/RST | N/A | Not used. Assumes modem is always-on. |

Digital Sensors

| Sensor | Shield Port | ESP32-E Pin | Note |
| Temp (DS18B20) | D2 | 25 (PIN_ONE_WIRE_BUS) | 1-Wire. Requires 4.7k pull-up resistor. |
| Water Leak | D3 | 26 (PIN_WATER_LEAK) | Digital Signal |

Analog Sensors

All sensors connect to the 6-port analog block.

| Sensor | Shield Port | ESP32-E Pin | config.h Define |
| ORP (SEN0464) | A0 | 36 | PIN_ORP |
| (Unused) | A1 | 39 |  |
| pH (SEN0161-V2) | A2 | 34 | PIN_PH |
| DO (SEN0237) | A3 | 35 | PIN_DO |
| EC (SEN0451) | A4 | 32 | PIN_EC |
| Turbidity (SEN0189) | A5 | 33 | PIN_TURBIDITY |