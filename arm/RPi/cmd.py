from .py2A1motor import AImotor
from math import sqrt,pi,acos,atan2
from time import sleep

servo1 = AImotor(1)
servo2_a = AImotor(2)
servo2_b = AImotor(7)
servo3_a = AImotor(3)
servo3_b = AImotor(8)
servo4 = AImotor(4)
servo5 = AImotor(5)
servo6 = AImotor(6)

turn_num = 640
turn180 = 290
tx=0
ty=0
tz=0

CMD = ["p","c","t","m","r","x","y","z","e","j"]

def Park(playtime):
    global tx,ty,tz
    tx = -7
    ty = 1
    tz = 10
    Move_XYZ(tx,ty,tz,playtime)
    Servo_Turn(6,430,playtime)


def Degree2Steps(id_num,degree,servo_deg=1024/330):
    motor = [233,791,791,1024] #0,180,180
    return motor[id_num] - degree * servo_deg
    
def Move_XYZ(x,y,z,playtime):
    a = 19.5
    b = 22.3 
    d = 15.7
    
    h = 5.5 
    s = h-z
    e = d-s
    l = sqrt( x * x + y * y)
    c = sqrt( e * e + l * l)

    theta = [0,0,0,0]
    target = [0,0,0,0,0]
    theta[0] = atan2( y , x ) * 180 / pi

    theta[3] = (acos((c*c + e*e - l*l) / (2*c*e)) * 180 / pi) + (acos( (b*b + c*c - a*a) / (2*b*c)) * 180 / pi) 
    theta[2] = acos((a*a + b*b - c*c) / (2*a*b)) * 180 / pi
    theta[1] = 360 - theta[3] - theta[2]

    target[0] = Degree2Steps(0,-theta[0])  
    target[1] = Degree2Steps(1,theta[1]) 
    target[2] = Degree2Steps(2,theta[2]) 
    target[3] = Degree2Steps(3,theta[3]) 

    for i in range(4,0,-1):
        if theta[i] == None:
            return

    order = [3,4,2,1]

    for i in range(4):
        Servo_Turn(order[i],target[order[i]-1],playtime)

def Servo_Turn(servonum,degree,playtime):
    if servonum == 1:
        servo1.Moove(degree,playtime) 
    elif servonum == 2:
        servo2_a.Moove(degree, playtime) 
        servo2_b.Moove(1024-degree, playtime) 
    elif servonum == 3:
        servo3_a.Moove(degree, playtime)
        servo3_b.Moove(1024-degree, playtime)
    elif servonum == 4:
        servo4.Moove(degree, playtime)
    elif servonum == 5:
        servo5.Moove(degree, playtime)
    elif servonum == 6:    
        if degree>=430 and degree<=570:
           servo6.Moove(degree, playtime)

def Arm_Command(command=""):
    global tx,ty,tz
    command_list = command.split(';')
    for cmd in command_list:

        data_str = cmd.split(' ')

        if len(data_str) == 1:
            action = data_str[0]
        elif len(data_str) == 2:
            action,data = data_str
        else:
            return

        if action not in CMD:
            return

        if action == CMD[0]:
            Servo_Turn(6,430,170)

        elif action == CMD[1]:
            Servo_Turn(6,548,150)
            
        elif action == CMD[2]:
            global turn180
            Servo_Turn(5,turn_num+turn180,100)
            turn180 *= -1

        elif action == CMD[3]:
            data_str = data.split(',')
            if len(data_str) != 3:
                return
            for i in data_str:
                try:
                    int(i)
                except:
                    return
            tx,ty,tz = data_str
            Move_XYZ(tx,ty,tz,120)

        elif action == CMD[4]:
            Park(150)

        elif action == CMD[5]:
            data_str = data.split(',')
            if len(data_str) != 1:
                return
            try:
                int(data_str[0])
            except:
                return
            tx = data_str[0]
            Move_XYZ(tx,ty,tz,90)
        
        elif action == CMD[6]:
            data_str = data.split(',')
            if len(data_str) != 1:
                return
            try:
                int(data_str[0])
            except:
                return
            ty = data_str[0]
            Move_XYZ(tx,ty,tz,90)
    
        elif action == CMD[7]:
            data_str = data.split(',')
            if len(data_str) != 1:
                return
            try:
                int(data_str[0])
            except:
                return
            tz = data_str[0]
            Move_XYZ(tx,ty,tz,90)

        elif action == CMD[8]:
            Move_XYZ(-18,1,7,100)
            sleep(2)
            Servo_Turn(6,430,100)

        elif action == CMD[9]:
            data_str = data.split(',')
            if len(data_str) != 2:
                return
            for i in data_str:
                try:
                    int(i)
                except:
                    return
            n,d = data_str
            Servo_Turn(n,d,100)
