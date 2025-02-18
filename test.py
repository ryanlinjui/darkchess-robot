from brain.utils import available, available2

board = list("00*P00M*M*0*0c0**K00gk*R**P0PG**")
board = [
    "0", "0", "0", "0", "M", "0", "0", "0",
    "0", "0", "P", "0", "M", "*", "M", "*",
    "K", "*", "0", "0", "c", "0", "*", "P",
    "K", "0", "0", "g", "k", "*", "R", "*",
]

print(available(board, 1))
print()
print(available2(board, 1))

print()
print(available(board, -1))
print()
print(available2(board, -1))