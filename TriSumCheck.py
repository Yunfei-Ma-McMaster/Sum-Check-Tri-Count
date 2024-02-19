def d(w1,w2,w3,w4,x1,x2,x3,x4):
  if w1 == 0:
    res = (1-x1)
  else:
      res = x1
  if w2 == 0:
      res = res *(1-x2)
  else:
      res = res * x2
  if w3 == 0:
      res = res *(1-x3)
  else:
      res = res * x3
  if w4 == 0:
      res = res *(1-x4)
  else:
      res = res * x4
  return res
#end

def f(x1,x2,x3,x4):
  #A = [[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]  #square
  #A = [[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,0,0,0]]  #claw
  #A = [[0,1,1,0],[1,0,1,1],[1,1,0,0],[0,1,0,0]]  #pan
  A = [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]] #spindle
  #A = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]] #complete
  a = 2*x1+x2
  b = 2*x3+x4
  return A[a][b]
#end


def F(x1,x2,x3,x4):
  res = 0
  for w1 in range(2):
    for w2 in range(2):
      for w3 in range(2):
        for w4 in range(2):
          res = res + f(w1,w2,w3,w4) * d(w1,w2,w3,w4,x1,x2,x3,x4)
  return res
#end


res = 0
for x1 in range(2):
  for x2 in range(2):
    for x3 in range(2):
      for x4 in range(2):
        for x5 in range(2):
          for x6 in range(2):
            res = res + (F(x1,x2,x3,x4) * F(x1,x2,x5,x6) * F(x3,x4,x5,x6))
print("number of triangles =",res//6)
            