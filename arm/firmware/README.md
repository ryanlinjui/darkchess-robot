# Getting Started

Update the following settings in the files **[remote_esp8266.ino](./remote_esp8266/remote_esp8266.ino), [receiver_esp8266.ino](./receiver_esp8266/receiver_esp8266.ino)**:

- **Server:** Set the IP address and port.
- **WiFi:** Configure the SSID and password.

> Note: ESP8266 supports only 2.4 GHz WiFi networks.

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

| **Command**          | **Description**                                        | **Example**           |
|:--------------------:|:------------------------------------------------------:|:---------------------:|
| **p (Park)**         | Parks the arm in ininal area.         | `p;`        |
| **c (Catch)**        | Catch / Grab action.         | `c;`                  |
| **t (Turn)**         | Turn over / Flip action.            | `t;`              | 
| **r (Release)**      | Releases the held chess action. | `r;`    |
| **m (Move)**         | Moves the arm to specified coordinates (x, y, z).      | `m -10,12,15;`        |
| **x (X-axis)**       | Adjusts the arm's position along the X-axis.           | `x 15;`               |
| **y (Y-axis)**       | Adjusts the arm's position along the Y-axis.           | `y 20;`               |
| **z (Z-axis)**       | Adjusts the arm's position along the Z-axis. | `z 10;` |
| **e (Eat)**          | Eat the chess action       | `e;`       |
| **b (Buffer)**       | Move to buffer middle area to reduce positional errors.  | `b;` |
| **d (Done)**         | Print `done` as signal to response receiver system. | `d;`       |

> Command are separated by **`;` semicolons**.  
> You can use **Monitor Serial** to operate.

### Example
- **Open Chess**
```
m 2.85,9.6,10; z 0.4; c; z 10; t; z 0.4; r; z 10; p;
```
> **Open chess at (21)**  
> If No.21 position at **(2.85, 9.6, 10 , bottom-z: 0.4)**

- **Move Chess**
```
m 6.5,14.3,10; z 0.9; c; z 10; b; m 10.1,19,10; z 1.8; r; z 10; p;
```
> **Move chess from (14) to (7)**  
> If No.14 position at **(6.5, 14.3, 10, bottom-z: 0.9)**  
> and No.7 position at **(10.1, 19, 10, bottom-z: 1.8)**

- **Eat Chess**
```
m -3.35,13.9,10; z 0.7; c; z 10; e; b; m -0.4,14,10; z 0.7; c; z 10; b; m -3.35,13.9,10; z 0.7; r; z 10; p;
```
> **Use chess at (12) to eat chess at (11)**  
> If No.11 position at **(-3.35, 13.9, 10, bottom-z: 0.7)**  
> and No.12 position at **(-0.4, 14, 10, bottom-z: 0.7)**
