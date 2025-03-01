# Getting Started
- [arduino-cli](https://arduino.github.io/arduino-cli/latest/installation/)
- [ESP8266](https://github.com/esp8266/Arduino)
- [Atmega1280](https://www.microchip.com/wwwproducts/en/ATmega1280)

## Install Core, and its Tools, Libraries
```bash
arduino-cli core install esp8266:esp8266 --config-file arduino-cli.yaml
arduino-cli lib install XYZrobotServo --install-in-builtin-dir --config-file arduino-cli.yaml
```

## Compile
```bash
arduino-cli compile --fqbn esp8266:esp8266:nodemcuv2
```

## Upload
```bash
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp8266:esp8266:nodemcuv2
```