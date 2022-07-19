#include <XYZrobotServo.h>
#ifdef SERIAL_PORT_HARDWARE_OPEN
#define servoSerial SERIAL_PORT_HARDWARE_OPEN
#else
#include <SoftwareSerial.h>
SoftwareSerial Serial1(10, 11);
#endif

XYZrobotServo servo1(Serial1, 1);
XYZrobotServo servo2_a(Serial1, 2);
XYZrobotServo servo2_b(Serial1, 7); 
XYZrobotServo servo3_a(Serial1, 3); 
XYZrobotServo servo3_b(Serial1, 8); 
XYZrobotServo servo4(Serial1, 4);
XYZrobotServo servo5(Serial1, 5);
XYZrobotServo servo6(Serial1, 6);

float servo_deg = (float)1024 / 330;

int playtime;

enum E_CMD {N,P,C,T,M,R,X,Y,Z,E,D,B,J};
String CMD[13] = {"N","P","C","T","M","R","X","Y","Z","E","D","B","J"};

String tmp="";
int move_duration=100,motor_id=1;
int target[4];
float tx,ty,tz;
int turn_num = 640;
int turn180 = 290;

int searchCommand(String cmd){
  for(int i=0;i<100;i++){
    if(cmd == CMD[i]){
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
    move_xyz(tx,ty,tz);
    servo_turn(6,430);
}

int Degree2Steps(int id,double degree)
{
  int motor[4] = {233,791,791,1024}; //0,180,180
  return motor[id]-degree* servo_deg;
}

void move_xyz(float x,float y,float z)
{
  //Arm Formula ---------------------
  float a = 19.5; //arm argument
  float b = 22.3; //arm argument
  float d = 15.7; //arm argument
  
  float h = 5.5; 
  float s = h-z; 
  float e = d-s; 
  float l = sqrt( x * x + y * y); 
  float c = sqrt( e * e + l * l); 
  // ---------------------------------

  float theta[4];
  theta[0] = atan2( y , x ) * 180 / PI; //Polar Coordinate

  //Law of Cosines ------------------------
  theta[3] = (acos( (c*c + e*e - l*l) / (2*c*e)) * 180 / PI) + 
                 (acos( (b*b + c*c - a*a) / (2*b*c)) * 180 / PI);
                
  theta[2] = acos( (a*a + b*b - c*c) / (2*a*b)) * 180 / PI;
  theta[1] = 360-theta[3] - theta[2];
  // --------------------------------------
  
  target[0] = Degree2Steps(0,-theta[0]); 
  target[1] = Degree2Steps(1,theta[1]); 
  target[2] = Degree2Steps(2,theta[2]); 
  target[3] = Degree2Steps(3,theta[3]); 
  show(theta,target);
  
  for(int i=4;i>0;i--)
  {
    if(isnan(theta[i]))
    {
      return;
    }
  }
  int order[4]{3,4,2,1};
  for(int i=0;i<4;i++)
  {
      servo_turn(order[i],target[order[i]-1]);
  }
  //delay(200);
}

void show(float theta[4],int target[4])
{
  for(int i=0;i<4;i++)
  {
    Serial.print(i+1);Serial.print(": theta:");Serial.print(theta[i]);Serial.print(" target:");Serial.println(target[i]);
  }
  Serial.println();
}

void servo_turn(int servonum,int degree)
{
  switch(servonum)
  {
    case 1:
      servo1.setPosition(degree, playtime);
      break;
      
    case 2:
      servo2_a.setPosition(degree, playtime);
      servo2_b.setPosition(1024-degree, playtime);
      break;

    case 3:
      servo3_a.setPosition(degree, playtime);
      servo3_b.setPosition(1024-degree, playtime);
      break;
    
    case 4:
      servo4.setPosition(degree, playtime);
      break;        

    case 5:        
      servo5.setPosition(degree, playtime);
      break;
      
    case 6:
      if(degree>=430 && degree<=570)
      {
          servo6.setPosition(degree, playtime);
      }
      break;
    
    default:
      break;
  }
}


float GetData(String data,int i)
{
  int count = 0;
  String getdata;
  while(count<=i)
  {
      getdata = data.substring(0,data.indexOf(','));
      data = data.substring(data.indexOf(',')+1);
      count++;
  }
  return getdata.toFloat();
}

String GetDataStr(String data,int i)
{
  int count = 0;
  String getdata;
  while(count<=i)
  {
    getdata = data.substring(0,data.indexOf(','));
    data = data.substring(data.indexOf(',')+1);
    count++;
  }
  return getdata;
}

void setup()
{
  Serial1.begin(115200);
  Serial.begin(115200);
  playtime = 150;
  servo_turn(5,350);
  park();
}

void loop()
{
  String s;
  String ss;
  if(Serial.available()) 
  {
    s = Serial.readString();    
    s.replace("\n","");
    s.replace("\r","");
  }
  while(s.indexOf(';')!=-1)
  {
    ss = s.substring(0,s.indexOf(';'));
    s.remove(0,s.indexOf(';')+1);
    String cmd = ss.substring(0,ss.indexOf(' '));
    String data = ss.substring(ss.indexOf(' ')+1);
    int mid=0;
    int tmp;
    cmd.toUpperCase();
    Serial.println(ss);
    switch(searchCommand(cmd))
    {
      case P:
      {
        playtime = 170;
        servo_turn(6,430);
        break;
      }
        
      case C:   
      {
        playtime = 150;
        //servo_turn(6,548);
        servo_turn(6,560);
        break;
      }

      case T:
      {
        playtime = 100;
        servo_turn(5,turn_num + turn180);
        turn180 *= -1;
        break;
      }

      case M:
      {
        playtime = 120;
        tx = (GetDataStr(data,0)!="")?GetData(data,0):tx;
        ty = (GetDataStr(data,1)!="")?GetData(data,1):ty;
        tz = (GetDataStr(data,2)!="")?GetData(data,2):tz;
        move_xyz(tx,ty,tz);
        break;
      }

      case R:
      { playtime = 150;
        park();
        break;
      }
      
      case X:
      { 
        playtime = 90;
        tx = GetData(data,1);
        move_xyz(tx,ty,tz);
        break;
      }
        
      case Y:
      {  
        playtime = 90;
        ty = GetData(data,1);
        move_xyz(tx,ty,tz);
        break;
      }
        
      case Z:
      {
        playtime = 90;
        tz = GetData(data,1);
        move_xyz(tx,ty,tz);
        break;
      }

      case E:
      {
        playtime = 100;
        move_xyz(-18,1,7);
        delay(2200);
        playtime = 150;
        servo_turn(6,430);
        break;
      }

      case D:
      {
        Serial.println("done");
        break;
      }
        
      case B:
      {
        playtime = 120;
        move_xyz(1,10,10);
        break;
      }
        
      case J:
      {
        playtime = 150;
        int n = (GetDataStr(data,0)!="")?GetData(data,0):n;
        int d = (GetDataStr(data,1)!="")?GetData(data,1):d;
        servo_turn(n,d);
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
