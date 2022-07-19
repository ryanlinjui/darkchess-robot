import serial

def SEND(ID,CMD,DATA=[]):
    N = 7 + len(DATA)
    check_sum_1 = (N^ID^CMD)
    for d in DATA:
        check_sum_1 = check_sum_1 ^ d
    check_sum_1 = check_sum_1 & 0xFE
    check_sum_2 = (~check_sum_1) & 0xFE
    while not AImotor.serial0.isOpen():
        AImotor.serial0.open()
    AImotor.serial0.write(bytearray(packet))
    while not AImotor.serial0.isClose():
        AImotor.serial0.close()

class AImotor:
    serial0 = serial.Serial(
        port = '/dev/ttyS0',
        baudrate = 115200,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS,
        timeout = 3
    )
    playtime = 150

    def __init__(self,id,playtime=playtime):
        self.id = id
        self.playtime = playtime

    def Moove(self,goal,playtime=None):
        if playtime!=None :
            pass
        elif self.playtime!=None:
            playtime = self.playtime
        else:
            playtime = AImotor.playtime
        SEND(254,0x06,[playtime,goal >> 8 & 0x03,goal & 0xff,0,self.id])
    
    def GetPoosa(self):
        SEND(self.id,0x07)
        ACK = bytearray(AImotor.serial0.read())
        return (ACK[14] << 8)&(ACK[13])
