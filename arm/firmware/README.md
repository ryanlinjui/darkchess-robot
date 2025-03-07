# Getting Started
## Prerequisite and Notes
Please set some settings before you start.
[Remote.ino](./Remote/Remote.ino) 
[Receiver.ino](./Receiver/Receiver.ino)
ip, port, ssid, password, etc.
[arm.ino](./arm/arm.ino)
[servo.ino](./arm/servo.ino)

## Flashing Firmware
#### Board & FBQN List
- [Atmega1280](https://www.microchip.com/wwwproducts/en/ATmega1280)
    - FBQN: `arduino:avr:mega`
- [ESP8266](https://github.com/esp8266/Arduino)
    - FBQN: `esp8266:esp8266:generic`
> Check board's FBQN by `arduino-cli board listall <board-name>`

#### Install Core, Libraries
```bash
arduino-cli core install esp8266:esp8266 --config-file arduino-cli.yaml
arduino-cli lib install XYZrobotServo
```

#### Attach & Set board
```
arduino-cli board attach --fqbn <your-fbqn> --port <your-usb-port-name> <your-sketch>
```
> Check usb port name by `arduino-cli board list`

#### Compile & Upload code
```
arduino-cli compile --upload <your-sketch> 
```
> You can use `--build-property build.extra_flags=-D<your-marco>` to set marco

#### Monitor Serial Output
```
arduino-cli monitor --config 115200 <your-sketch>
```