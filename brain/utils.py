# -*- coding: utf-8 -*-

from globfile import (
    EN_CHESS
)

def available(board:list, color:int) -> list:
    all_chess = list(
        EN_CHESS[0] * 5 + EN_CHESS[1] * 2 + EN_CHESS[2] * 2 +  EN_CHESS[3] * 2 +  EN_CHESS[4] * 2 +  EN_CHESS[5] * 2 +  EN_CHESS[6] +
        EN_CHESS[7] * 5 + EN_CHESS[8] * 2 + EN_CHESS[9] * 2 + EN_CHESS[10] * 2 + EN_CHESS[11] * 2 + EN_CHESS[12] * 2 + EN_CHESS[13]
    )
    power = [1, 1, 1, 1, 1, 10, 10, 3, 3, 4, 4, 5, 5, 6, 6, 7, 1, 1, 1, 1, 1, 10, 10, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    hp =    [1, 1, 1, 1, 1,  2,  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 1, 1, 1, 1, 1,  2,  2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    availablestep=[[49] * 50 for i in range(50)]
    size = 32
    ro, co = 8, 4
    for i in range(size):
        if board[i] == EN_CHESS[1] or board[i] == EN_CHESS[8]:
            for j in range(ro):
                middlechessl = 0
                middlechessr = 0
                if board[ro*int(i/ro)+j] != EN_CHESS[15] and board[ro*int(i/ro)+j] != EN_CHESS[14]:
                    if (all_chess.index(board[ro*int(i/ro)+j]) > 15 and all_chess.index(board[i]) < 16) or (all_chess.index(board[ro*int(i/ro)+j]) < 16 and all_chess.index(board[i]) > 15):
                        if ro * int(i/ro) + j < i:
                            for k in range(ro*int(i/ro)+j+1, i, 1):
                                if board[k] != EN_CHESS[15]:
                                    middlechessl += 1
                        else:
                            for k in range(i+1, ro*int(i/ro)+j, 1):
                                if board[k] != EN_CHESS[15]:
                                    middlechessl += 1
                        if middlechessl == 1:
                            for a in range(15):
                                if availablestep[i][a] == 49:
                                    availablestep[i][a] = ro * int(i/ro) + j
                                    break
                        if middlechessr == 1:
                            for a in range(15):
                                if availablestep[i][a] == 49:
                                    availablestep[i][a] = ro * int(i/ro) + j
                                    break
            for j in range(0, co, 1):
                middlechessl = 0
                middlechessr = 0
                i2 = i
                while i2 >= ro:
                    i2 -= ro
                if board[i2+j*ro] != EN_CHESS[15] and board[i2+j*ro] != EN_CHESS[14]:
                    if (all_chess.index(board[i2+j*ro]) > 15 and all_chess.index(board[i]) < 16) or (all_chess.index(board[i2+j*ro]) < 16 and all_chess.index(board[i]) > 15):
                        if i2 + j * ro < i:
                            for k in range(i2+j*ro+ro, i, ro):
                                if board[k] != EN_CHESS[15]:
                                    middlechessl += 1
                        else:
                            for k in range(i+ro, i2+j*ro, ro):
                                if board[k] != EN_CHESS[15]:
                                    middlechessl += 1
                        if middlechessl == 1:
                            for a in range(15):
                                if availablestep[i][a] == 49:
                                    availablestep[i][a] = i2 + j * ro
                                    break
                        if middlechessr == 1:
                            for a in range(15):
                                if availablestep[i][a] == 49:
                                    availablestep[i][a] = i2 + j * ro
                                    break
        rstep=True
        lstep=True
        dstep=True
        ustep=True
        for a in range(15):
            if availablestep[i][a] == 49:
                for b in range(4):
                    if i + 1 == ro * b:
                        rstep = False
                    if i - 1 == ro * b - 1:
                        lstep = False
                    if i + ro > size - 1:
                        dstep = False
                    if i - ro < 0:
                        ustep = False
                if rstep == True and board[i] != EN_CHESS[15] and board[i] != EN_CHESS[14] and i != size - 1:
                    if ((board[i] == EN_CHESS[0] or board[i] == EN_CHESS[7]) and (board[i+1] == EN_CHESS[6] or board[i+1] == EN_CHESS[13])) or ((board[i+1] == EN_CHESS[0] or board[i+1] == EN_CHESS[7]) and (board[i] == EN_CHESS[6] or board[i] == EN_CHESS[13])):
                        power[15] = 0
                        power[31] = 0
                        hp[15] = 0
                        hp[31] = 0
                    if board[i+1] == EN_CHESS[15]:
                        availablestep[i][a] = i + 1
                    elif board[i+1] == EN_CHESS[14]:
                        p = 1
                    elif (power[all_chess.index(board[i])] >= hp[all_chess.index(board[i+1])] and ((all_chess.index(board[i+1]) > 15 and all_chess.index(board[i]) < 16) or (all_chess.index(board[i+1]) < 16 and all_chess.index(board[i]) > 15))) and board[i] != EN_CHESS[1] and board[i] != EN_CHESS[1]:
                        availablestep[i][a] = i + 1
                    power[15] = 7
                    power[31] = 7
                    hp[15] = 7
                    hp[31] = 7
                    rstep = False
                elif lstep == True and board[i] != EN_CHESS[15] and board[i] != EN_CHESS[14] and i != 0:
                    if ((board[i] == EN_CHESS[0] or board[i] == EN_CHESS[7]) and (board[i-1] == EN_CHESS[6] or board[i-1] == EN_CHESS[13])) or ((board[i-1] == EN_CHESS[0] or board[i-1] == EN_CHESS[7]) and (board[i] == EN_CHESS[6] or board[i] == EN_CHESS[13])):
                        power[15] = 0
                        power[31] = 0
                        hp[15] = 0
                        hp[31] = 0
                    if board[i-1] == EN_CHESS[15]:
                        availablestep[i][a] = i - 1
                    elif board[i-1] == EN_CHESS[14]:
                        p = 1
                    elif (power[all_chess.index(board[i])] >= hp[all_chess.index(board[i-1])] and ((all_chess.index(board[i-1]) > 15 and all_chess.index(board[i]) < 16) or (all_chess.index(board[i-1]) < 16 and all_chess.index(board[i]) > 15))) and board[i] != EN_CHESS[1] and board[i] != EN_CHESS[1]:
                        availablestep[i][a] = i - 1
                    power[15] = 7
                    power[31] = 7
                    hp[15] = 7
                    hp[31] = 7
                    lstep = False
                elif dstep == True and board[i] != EN_CHESS[15] and board[i] != EN_CHESS[14] and i < 24:
                    if ((board[i] == EN_CHESS[0] or board[i] == EN_CHESS[7]) and (board[i+ro] == EN_CHESS[6] or board[i+ro] == EN_CHESS[13])) or ((board[i+ro] == EN_CHESS[0] or board[i+ro] == EN_CHESS[7]) and (board[i] == EN_CHESS[6] or board[i] == EN_CHESS[13])):
                        power[15] = 0
                        power[31] = 0
                        hp[15] = 0
                        hp[31] = 0
                    if board[i+ro] == EN_CHESS[15]: 
                        availablestep[i][a] = i + ro
                    elif board[i+ro] == EN_CHESS[14]:
                        p = 1
                    elif (power[all_chess.index(board[i])] >= hp[all_chess.index(board[i+ro])] and ((all_chess.index(board[i+ro]) > 15 and all_chess.index(board[i]) < 16) or (all_chess.index(board[i+ro]) < 16 and all_chess.index(board[i]) > 15))) and board[i] != EN_CHESS[1] and board[i] != EN_CHESS[1]:
                        availablestep[i][a] = i + ro
                    power[15] = 7
                    power[31] = 7
                    hp[15] = 7
                    hp[31] = 7
                    dstep = False
                elif ustep == True and board[i] != EN_CHESS[15] and board[i] != EN_CHESS[14] and i > 7:
                    if ((board[i] == EN_CHESS[0] or board[i] == EN_CHESS[7]) and (board[i-ro] == EN_CHESS[6] or board[i-ro] == EN_CHESS[13])) or ((board[i-ro] == EN_CHESS[0] or board[i-ro] == EN_CHESS[7]) and (board[i] == EN_CHESS[6] or board[i] == EN_CHESS[13])):
                        power[15] = 0
                        power[31] = 0
                        hp[15] = 0
                        hp[31] = 0
                    if board[i-ro] == EN_CHESS[15]: 
                        availablestep[i][a] = i - ro
                    elif board[i-ro] == EN_CHESS[14]:
                        p = 1
                    elif (power[all_chess.index(board[i])] >= hp[all_chess.index(board[i-ro])] and ((all_chess.index(board[i-ro]) > 15 and all_chess.index(board[i]) < 16) or (all_chess.index(board[i-ro]) < 16 and all_chess.index(board[i]) > 15))) and board[i] != EN_CHESS[1] and board[i] != EN_CHESS[1]:
                        availablestep[i][a] = i - ro
                    power[15] = 7
                    power[31] = 7
                    hp[15] = 7
                    hp[31] = 7
                    ustep = False
    for i in range(size):
        while 49 in availablestep[i]:
            availablestep[i].remove(49)
    if color == 1:
        for i in range(size):
            if len(availablestep[i]) > 0 and all_chess.index(board[i]) > 15:
                availablestep[i].clear()
    elif color == -1:
        for i in range(size):
            if len(availablestep[i]) > 0 and all_chess.index(board[i]) < 16:
                availablestep[i].clear()
    temp = []
    n = 0
    for i in availablestep:
        if 49 in i:
            break
        for j in i:
            temp.append([n, j])
        n += 1
    for i in range(len(board)):
        if board[i] == EN_CHESS[14]:
            temp.append([i, i])
    return temp