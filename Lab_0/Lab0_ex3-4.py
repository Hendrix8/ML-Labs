import numpy as np 

# Askhsh 3
def toDict(Coordinates):
    coorDict = {}
    for i in Coordinates:
        coorDict[i[0]] = i[1]
    return coorDict

L = [(1,2), (3,5), (0,1)]

print(toDict(L))

# ---------

# Askhsh 4

userCoor = {}
pointsCardinality = 0
running = ""
while running != "end":
    
    try:
        x = float(input("Give x: "))
        y = float(input("Give y: "))

        pointsCardinality += 1

        userCoor["point "+ str(pointsCardinality) ] = (x, y)
        running = raw_input("To stop type 'end', else press enter: ")
    except:
        print("invalid input.Try again.")

def in_circle(coorDict, radius):
    inCircle = []
    
    for value in coorDict.values():
        if (value[0]**2 + value[1]**2) <= radius**2 :
            inCircle.append(value)

    return inCircle

print(userCoor)
print("Points in the circle with radius 2: ",in_circle(userCoor,2))

