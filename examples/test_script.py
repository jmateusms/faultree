def gand(*p):
    result = 1
    for i in p:
        result *= i
    return result

def gor(*p):
    sub = 1
    for i in p:
        sub *= 1 - i
    
    result = 1 - sub
    return result

if __name__ == "__main__":
    a = 0.12
    b = 0.07
    c = 0.09

    B = gand(a, b)
    C = gand(a, c)

    A = gor(B, C)

    print(A)

    print(a * b * c + a * b * (1 - c) + a * (1 - b) * c)
