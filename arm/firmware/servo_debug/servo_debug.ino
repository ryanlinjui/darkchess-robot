#include <XYZrobotServo.h>

#ifdef SERIAL_PORT_HARDWARE_OPEN
    #define servoSerial SERIAL_PORT_HARDWARE_OPEN
#else
    #include <SoftwareSerial.h>
    SoftwareSerial Serial1(10, 11);
#endif

XYZrobotServo servo1(Serial1, 1); 
XYZrobotServo servo2_a(Serial1, 2);
XYZrobotServo servo3_a(Serial1, 3);
XYZrobotServo servo4(Serial1, 4); 
XYZrobotServo servo5(Serial1, 5);
XYZrobotServo servo6(Serial1, 6);
XYZrobotServo servo2_b(Serial1, 7);
XYZrobotServo servo3_b(Serial1, 8);

const uint8_t playtime = 150; // turn speed
const uint8_t turntime = 255;

void regular()
{
    servo1.setPosition(512, playtime); // regular: 512
    // servo2.setPosition(335, playtime); // regular: 120, min: 100 max: 480
    // servo3.setPosition(676, playtime); // regular: 620, min: 100 max: 750
    // servo4.setPosition(335, playtime); // regular: 790, turn 0 to 180 (790 ~ 240)
    // servo5.setPosition(512, playtime); // regular: 512, ID: 11
    // servo6.setPosition(430, playtime); // regular: 165, min: 165, max: 300 ID: 8
}

void setup()
{
    Serial1.begin(115200); // motor
    Serial.begin(115200);  // tx/rx
    regular();
}

void loop()
{
    if (Serial.available())
    { 
        String communication = Serial.readString(); 
        int servonum = communication.substring(0, 1).toInt();
        int degree = communication.substring(2, 6).toInt();  
        
        Serial.println("communication = ");
        Serial.println(communication);  
        Serial.println("servonum = ");
        Serial.println(servonum);
        Serial.println("degree = ");
        Serial.println(degree);
        
        switch(servonum)
        {
            case 1:
                servo1.setPosition(degree, playtime); 
                break;
                
            case 2:
                servo2_a.setPosition(degree, playtime);
                servo2_b.setPosition(1024 - degree, playtime);
                break;

            case 3:
                servo3_a.setPosition(degree, playtime);
                servo3_b.setPosition(1024 - degree, playtime);
                break;

            case 4:
                servo4.setPosition(degree, playtime);
                break;
            
            case 5:
                servo5.setPosition(degree, turntime);
                break;

            case 6:
                if(degree >= 400 && degree <= 600)
                {
                    servo6.setPosition(degree, playtime);
                    break;
                }
                else
                {
                    Serial.println("No");
                }
                
            default:
                break;
        }
    }
}