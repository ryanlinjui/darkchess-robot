import numpy as np

def xyz_write(filename):
    f = open(filename,'w')
    four = np.array([(-16,18,10,3.5),(9.5,20,10,3),(-13,6,10,1.3),(9,7.5,10,1.1)])
    d1 = (four[2]-four[0])/3
    d2 = (four[3]-four[1])/3
    for r in range(4):
        s1 = four[0]+d1*r    
        s2 = four[1]+d2*r
        dx = (s2-s1) / 7
        for c in range(8):
            s = s1+dx*c
            for w in range(4):
                f.write(str(round(s[w],1)))
                if w != 3: f.write(',')
            if r*8+c != 31: f.write('\n')
    f.close()

if __name__ == '__main__':
    xyz_write(filename="new.txt")