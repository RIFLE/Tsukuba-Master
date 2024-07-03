import numpy

P1 = [0,0] #upper position
P2 = [-10,10] #anyway position
P3 = [10,10] #anyway position

def trilateration(P1, P2, P3, r1, r2, r3):
#r1,r2,r3 is the distance of staff and cecsor

  p1 = numpy.array([0, 0])
  p2 = numpy.array([P2[0] - P1[0], P2[1] - P1[1]])
  p3 = numpy.array([P3[0] - P1[0], P3[1] - P1[1]])
  v1 = p2 - p1
  v2 = p3 - p1

  Xn = (v1)/numpy.linalg.norm(v1)

  tmp = numpy.cross(v1, v2)

  Zn = (tmp)/numpy.linalg.norm(tmp)

  Yn = numpy.cross(Xn, Zn)

  i = numpy.dot(Xn, v2)
  d = numpy.dot(Xn, v1)
  j = numpy.dot(Yn, v2)

  X = ((r1**2)-(r2**2)+(d**2))/(2*d)
  Y = (((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(X))
  Z1 = numpy.sqrt(max(0, r1**2-X**2-Y**2))
  Z2 = -Z2

  K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
  K2 = P1 + X * Xn + Y * Yn + Z2 * Zn
  return K1,K2

