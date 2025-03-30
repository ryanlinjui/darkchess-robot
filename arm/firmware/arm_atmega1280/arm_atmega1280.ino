#include <XYZrobotServo.h>

#ifdef SERIAL_PORT_HARDWARE_OPEN
    #define servoSerial SERIAL_PORT_HARDWARE_OPEN
#else
    #include <SoftwareSerial.h>
    SoftwareSerial Serial1(10, 11);
#endif

// ==============================================
// Please visit Research Report link at README.md
// ==============================================

// Set servo ID on each arm joint
XYZrobotServo servo1(Serial1, 1);
XYZrobotServo servo2_a(Serial1, 2);
XYZrobotServo servo2_b(Serial1, 7); 
XYZrobotServo servo3_a(Serial1, 3); 
XYZrobotServo servo3_b(Serial1, 8); 
XYZrobotServo servo4(Serial1, 4);
XYZrobotServo servo5(Serial1, 5);
XYZrobotServo servo6(Serial1, 6);

// Set the arm length of the robot arm (unit: cm)
#define ARM_LENGTH_ALPHA 19.5
#define ARM_LENGTH_BETA 22.3
#define ARM_LENGTH_DEPTH 15.7
#define ARM_LENGTH_HEIGHT 5.5
// ==============================================

float servo_deg = (float)1024 / 330;
int playtime = 150;

enum E_CMD {P, C, T, R, M, X, Y, Z, E, B, D};
String CMD[11] = {"P", "C", "T", "R", "M", "X", "Y", "Z", "E", "B", "D"};

int target[4] = {0, 0, 0, 0};
float tx = 0, ty = 0, tz = 0;
int turn_num = 640;
int turn180 = 290;

int searchCommand(String cmd);
void park();
int Degree2Steps(int id, double degree);
void move_xyz(float x, float y, float z);
void show(float theta[4], int target[4]);
void servo_turn(int servo_num, int degree);
float GetData(String data, int i);
String GetDataStr(String data, int i);

int searchCommand(String cmd)
{
    for(int i = 0; i < 11; i++)
    {
        if(cmd == CMD[i])
        {
            return i;
        }
    }
    return 0;
}

void park()
{
    tx = -7;
    ty = 1;
    tz = 10;
    move_xyz(tx, ty, tz);
    servo_turn(6, 430);
}

int Degree2Steps(int id, double degree)
{
    int servo[4] = {233, 791, 791, 1024}; // 0, 180, 180
    return servo[id] - degree * servo_deg;
}

void move_xyz(float x, float y, float z)
{
    // ========== Arm Formula ==========
    float a = ARM_LENGTH_ALPHA; // arm argument
    float b = ARM_LENGTH_BETA; // arm argument
    float d = ARM_LENGTH_DEPTH; // arm argument
    
    float h = ARM_LENGTH_HEIGHT;
    float s = h - z;
    float e = d - s;
    float l = sqrt(x * x + y * y);
    float c = sqrt(e * e + l * l); 
    // =================================

    float theta[4] = {0, 0, 0, 0};
    theta[0] = atan2(y, x) * 180 / PI; // Polar Coordinate

    // ======================= Law of Cosines =======================
    theta[3] = (acos((c * c + e * e - l * l) / (2 * c * e)) * 180 / PI) + 
                (acos((b * b + c * c - a * a) / (2 * b * c)) * 180 / PI);
                    
    theta[2] = acos((a * a + b * b - c * c) / (2 * a * b)) * 180 / PI;
    theta[1] = 360 - theta[3] - theta[2];
    // ==============================================================
    
    target[0] = Degree2Steps(0, -theta[0]); 
    target[1] = Degree2Steps(1, theta[1]); 
    target[2] = Degree2Steps(2, theta[2]); 
    target[3] = Degree2Steps(3, theta[3]); 
    show(theta, target);
    
    // Check if the theta is NaN
    for(int i = 3; i >= 0; i--)
    {
        if(isnan(theta[i]))
        {
            return;
        }
    }
    
    int order[4]{3, 4, 2, 1};
    for(int i = 0; i < 4; i++)
    {
        servo_turn(order[i], target[order[i] - 1]);
    }
    // delay(200);
}

void show(float theta[4], int target[4])
{
    for(int i = 0; i < 4; i++)
    {
        Serial.print(i + 1);
        Serial.print(": theta:");
        Serial.print(theta[i]);
        Serial.print(" target:");
        Serial.println(target[i]);
    }
    Serial.println();
}

void servo_turn(int servo_num, int degree)
{
    switch(servo_num)
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
            servo5.setPosition(degree, playtime);
            break;
        
        case 6:
            if(degree >= 430 && degree <= 570)
            {
                servo6.setPosition(degree, playtime);
            }
            break;
        
        default:
            break;
    }
}

float GetData(String data, int i)
{
    int count = 0;
    String getdata = "";

    while(count <= i)
    {
        getdata = data.substring(0, data.indexOf(','));
        data = data.substring(data.indexOf(',') + 1);
        count++;
    }
    return getdata.toFloat();
}

String GetDataStr(String data, int i)
{
    int count = 0;
    String getdata = "";
    
    while(count <= i)
    {
        getdata = data.substring(0, data.indexOf(','));
        data = data.substring(data.indexOf(',') + 1);
        count++;
    }
    return getdata;
}

void setup()
{
    Serial1.begin(115200);
    Serial.begin(115200);
    playtime = 150;
    servo_turn(5, 350);
    park();
}

void loop()
{
    String temp_cmd = "";
    String temp_data = "";
    
    if(Serial.available()) 
    {
        temp_cmd = Serial.readString();    
        temp_cmd.replace("\n", "");
        temp_cmd.replace("\r", "");
    }

    while(temp_cmd.indexOf(';') != -1)
    {
        temp_data = temp_cmd.substring(0, temp_cmd.indexOf(';'));
        temp_cmd.remove(0, temp_cmd.indexOf(';') + 1);
        
        String cmd = temp_data.substring(0, temp_data.indexOf(' '));
        String data = temp_data.substring(temp_data.indexOf(' ') + 1);

        cmd.toUpperCase();
        Serial.println(temp_data);
        switch(searchCommand(cmd))
        {
            case P:
            {
                playtime = 150;
                park();
                break;
                
            }
            
            case C:   
            {
                playtime = 150;
                //servo_turn(6, 548);
                servo_turn(6, 560);
                break;
            }

            case T:
            {
                playtime = 100;
                servo_turn(5, turn_num + turn180);
                turn180 *= -1;
                break;
            }

            case R:
            { 
                playtime = 150;
                servo_turn(6, 430);
                break;
            }

            case M:
            {
                playtime = 120;
                tx = GetDataStr(data, 0) != "" ? GetData(data, 0) : tx;
                ty = GetDataStr(data, 1) != "" ? GetData(data, 1) : ty;
                tz = GetDataStr(data, 2) != "" ? GetData(data, 2) : tz;
                move_xyz(tx, ty, tz);
                break;
            }
        
            case X:
            { 
                playtime = 90;
                tx = GetData(data, 1);
                move_xyz(tx, ty, tz);
                break;
            }
            
            case Y:
            {  
                playtime = 90;
                ty = GetData(data, 1);
                move_xyz(tx, ty, tz);
                break;
            }
            
            case Z:
            {
                playtime = 90;
                tz = GetData(data, 1);
                move_xyz(tx, ty, tz);
                break;
            }

            case E:
            {
                playtime = 100;
                move_xyz(-18, 1, 7);
                delay(2200);
                playtime = 150;
                servo_turn(6, 430);
                break;
            }

            case B:
            {
                playtime = 120;
                move_xyz(1, 10, 10);
                break;
            }

            case D:
            {
                Serial.println("done");
                break;
            }
            
            default:
            {
                break;
            }
        }
        delay(3000);
    }
}