# Getting Started

Update the following settings in the files **[remote_esp8266.ino](./remote_esp8266/remote_esp8266.ino), [receiver_esp8266.ino](./receiver_esp8266/receiver_esp8266.ino)**:

- **Server:** Set the IP address and port.
- **WiFi:** Configure the SSID and password.

> Note: The ESP8266 supports only 2.4 GHz WiFi networks.

# Flashing Firmware
### Boards and FBQN

- [**Atmega1280**](https://www.microchip.com/wwwproducts/en/ATmega1280)
  - FBQN: `arduino:avr:mega:cpu=atmega1280`
- [**ESP8266**](https://github.com/esp8266/Arduino)
  - FBQN: `esp8266:esp8266:generic`

> **Note:** Check board’s FBQN using:  
> `arduino-cli board listall <board-name>`

### Install Core, Libraries
```bash
arduino-cli core install arduino:avr esp8266:esp8266 --config-file arduino-cli.yaml
arduino-cli lib install XYZrobotServo
```

### Attach & Set board
```
arduino-cli board attach --fqbn <your-fbqn> --port <your-usb-port-name> <your-sketch>
```
> **Note:** Find your USB port by:  
> `arduino-cli board list`

### Compile & Upload code
```
arduino-cli compile --upload <your-sketch>
```
> **(Optional)** Use the flag to define macros:  
> `--build-property build.extra_flags=-D<your-macro>` 

### Monitor Serial Output
```
arduino-cli monitor --config 115200 <your-sketch>
```

# Operating the Robotic Arm
Use monitor serial to input command by following form:

| CMD_INIT          | CMD_MOVE_LEFT      | CMD_MOVE_RIGHT     | CMD_GRAB           | CMD_RELEASE        | CMD_STOP          |
|-------------------|--------------------|--------------------|--------------------|--------------------|-------------------|
| 初始化系統         | 手臂向左移動       | 手臂向右移動       | 抓取物品           | 釋放物品           | 停止動作           |
| CMD_ROTATE_CW     | CMD_ROTATE_CCW     | CMD_LIFT           | CMD_LOWER          | CMD_EXTEND         | CMD_RETRACT       |
| 順時針旋轉         | 逆時針旋轉         | 提升手臂           | 降低手臂           | 伸出手臂           | 收回手臂           |
> Command are separated by **`;` semicolons**.

### Example
- **Open Chess**
```
```
> Open chess at (6)

- **Move Chess**
```
```
> Move chess from (12) to (13)

- **Eat Chess**
```
```
> Use chess at (31) to eat chess at (29)
