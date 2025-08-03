import numpy as np
import matplotlib.pyplot as plt
import random

class Vertex:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def Equals(self, v):
    return (self.x == v.x and self.y == v.y)

  def Distance(self, v):
    vSelf=np.array([self.x, self.y])
    vInput=np.array([v.x, v.y])
    return np.linalg.norm(vSelf-vInput)

  def Print(self):
    print(f"{self.x}, {self.y}")

  def Display(self, ax=None, color='black', label=None, s=None):
    if ax is None:
      fig, ax = plt.subplots()
    ax.plot([self.x], [self.y], 'o', color=color, label=label, ms=s)
    #ax.legend()

class Edge:
  def __init__(self, v0, v1):
    self.v0 = v0
    self.v1 = v1

  def Equals(self, e):
    return (self.v0 == e.v0 and self.v1 == e.v1) or (self.v0 == e.v1 and self.v1 == e.v0)

  def GetVertexList(self):
    return [self.v0, self.v1]

  def Display(self, ax=None, color='black', marker='o'):
    if ax is None:
      fig, ax = plt.subplots()
    ax.plot([self.v0.x,self.v1.x], [self.v0.y,self.v1.y], marker=marker, color=color)

class Circle:
  def __init__(self, c, r):
    self.c=c
    self.r=r

class Triangle:
  def __init__(self, v0, v1, v2):
    #vertexes
    self.v0 = v0
    self.v1 = v1
    self.v2 = v2

    #circum circle
    self.circumCircle = self.computeCircumCircle(v0, v1, v2)

  def GetEdgeList(self):
    return [Edge(self.v0, self.v1), Edge(self.v1, self.v2), Edge(self.v2, self.v0)]

  def GetVertexList(self):
    return [self.v0, self.v1, self.v2]

  def computeCircumCircle(self, v0, v1, v2):
    d_v0_2=v0.x**2+v0.y**2
    d_v1_2=v1.x**2+v1.y**2
    d_v2_2=v2.x**2+v2.y**2
    #ref: https://en.wikipedia.org/wiki/Circumcircle#:~:text=Straightedge%20and%20compass%20construction,-Construction%20of%20the&text=The%20circumcenter%20of%20a%20triangle,the%20point%20where%20they%20cross.
    Sx=0.5*np.linalg.det(np.array([[d_v0_2, v0.y, 1.],
                                  [d_v1_2, v1.y, 1.],
                                  [d_v2_2, v2.y, 1.]]))
    Sy=0.5*np.linalg.det(np.array([[v0.x, d_v0_2, 1],
                                  [v1.x, d_v1_2, 1],
                                  [v2.x, d_v2_2, 1]]))
    a=np.linalg.det(np.array([[v0.x, v0.y, 1],
                            [v1.x, v1.y, 1],
                            [v2.x, v2.y, 1]]))
    b=np.linalg.det(np.array([[v0.x, v0.y, d_v0_2],
                            [v1.x, v1.y, d_v1_2],
                            [v2.x, v2.y, d_v2_2]]))
    #circum center
    c=Vertex(Sx/a, Sy/a)
    r=np.sqrt(b/a + np.linalg.norm(np.array([Sx,Sy]))**2 / a**2)
    circumCircle=Circle(c,r)

    return circumCircle

  def isInCircumCircle(self, v):
    c=self.circumCircle.c
    r=self.circumCircle.r
    return c.Distance(v) <= r

  def Display(self,ax=None, withCircle=False, color='black', label=None):
    if ax is None:
      fig, ax = plt.subplots()
    ax.plot([self.v0.x,self.v1.x], [self.v0.y,self.v1.y], marker='o', color=color)
    ax.plot([self.v1.x,self.v2.x], [self.v1.y,self.v2.y], marker='o', color=color)
    ax.plot([self.v2.x,self.v0.x], [self.v2.y,self.v0.y], marker='o', color=color, label=label)
    #ax.plot([self.circumCircle.c.x], [self.circumCircle.c.y], 'ro')
    if withCircle:
      circle = plt.Circle([self.circumCircle.c.x, self.circumCircle.c.y], radius=self.circumCircle.r, color='blue', alpha=0.3)
      ax.add_patch(circle)
    #plt.show()

class RandomVertexSet:
  def __init__(self, l, h, nVertices, seed=0):
    np.random.seed(seed)
    interiorVertexSet=np.random.rand(nVertices, 2)
    eps=2*1/nVertices
    interiorVertexSet[:,0] = eps+(l-2*eps)*interiorVertexSet[:,0]
    interiorVertexSet[:,1] = eps+(h-2*eps)*interiorVertexSet[:,1]

    self.rvsNumpy=interiorVertexSet

    vertexList=[Vertex(interiorVertexSet[i,0], interiorVertexSet[i,1]) for i in range(nVertices)]
    self.rvs=vertexList

  def Display(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
      ax.set_aspect('equal')
    ax.scatter(self.rvsNumpy[:,0], self.rvsNumpy[:,1])

  def FindMinimum(self, axis):
    return np.min(self.rvsNumpy[:,axis])

  def FindMaximum(self, axis):
    return np.max(self.rvsNumpy[:,axis])

class SuperTriangle(Triangle):
  def __init__(self, rvs):
    minX=rvs.FindMinimum(0)
    minY=rvs.FindMinimum(1)
    maxX=rvs.FindMaximum(0)
    maxY=rvs.FindMaximum(1)

    dx=(maxX-minX)
    dy=(maxY-minY)

    v0=Vertex(minX-dx, minY-dy)
    v1=Vertex(maxX+dx, minY-dy)
    v2=Vertex(0.5*(maxX+minX), maxY+2*dy)

    self.rvs=rvs
    #vertexes
    self.v0 = v0
    self.v1 = v1
    self.v2 = v2

    #circum circle
    self.circumCircle = Triangle.computeCircumCircle(self, v0, v1, v2)

class TriangularMesh():
  def __init__(self, st, rvs):

    self.tList=[st]
    self.toRemove=[]
    self.badTList=[]
    self.boundaryEdges=[]
    self.st=st
    self.rvs=rvs

  #Create new triangle list from adding a vertex
  def tListUpdate(self, v):
    #Remove triangles with circumcenter containing the vertex
    for t in self.tList:
      if t.isInCircumCircle(v):
        self.badTList.append(t)
    for t in self.badTList:
      self.tList.remove(t)

    #Get edges bounding the hole
    boundingEdges=self.GetUniqueEdgesFromTriangleList(self.badTList)
    self.boundaryEdges=boundingEdges

    #Create new triangles with bounding edges and the new vertex
    for e in boundingEdges:
      ev=e.GetVertexList()
      tri=Triangle(v, ev[0], ev[1])
      self.tList.append(tri)

  def GetUniqueEdgesFromTriangleList(self, tList):
    allEdges=[]
    toRemove=[]
    for t in tList:
      eList = t.GetEdgeList()
      for e in eList:
        if allEdges==[]:
          allEdges.append(e)
        else:
          for ee in allEdges:
            if ee.Equals(e):
              toRemove.append(ee)
              toRemove.append(e)
          allEdges.append(e)

    for e in toRemove:
      allEdges.remove(e)

    return allEdges

  def removeSuperTriangleStencil(self):
    for t in self.tList:
      vList=t.GetVertexList()
      stVList=self.st.GetVertexList()
      for v in vList:
        for stv in stVList:
          if v.Equals(stv):
            if t not in self.toRemove:
              self.toRemove.append(t)
    for t in self.toRemove:
      self.tList.remove(t)

  def emptyRemove(self):
    self.toRemove=[]

  def emptyBadTri(self):
    self.badTList=[]

  def emptyBoudaryEdges(self):
    self.boundaryEdges=[]

  def Display(self, ax=None, withCircle=False):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    for t in self.tList:
      t.Display(ax=ax, withCircle=withCircle)
    for t in self.toRemove:
      t.Display(ax=ax, withCircle=withCircle, color='red')
    for t in self.badTList:
      t.Display(ax=ax, withCircle=withCircle, color='red')
    for e in self.boundaryEdges:
      e.Display(ax=ax, color="blue", marker=None)
    self.rvs.Display(ax=ax)

if __name__ == '__main__':
  #random vertex set [0,5]²
  rvs=RandomVertexSet(20,10,200,seed=100)
  rvs.Display()

  #create super triangle from vertex set
  st=SuperTriangle(rvs)

  #mesh
  tm=TriangularMesh(st, rvs)
  tm.Display()
  for i in range(len(rvs.rvs)):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    tm.tListUpdate(rvs.rvs[i])
    tm.Display(ax=ax, withCircle=False)
    rvs.rvs[i].Display(ax=ax,label="current vertex",color="green", s=7)
    plt.legend()
    plt.show()
    tm.emptyBadTri()
    tm.emptyBoudaryEdges()

  tm.removeSuperTriangleStencil()
  tm.Display()

  tm.emptyRemove()
  tm.Display()
  tm.Display(withCircle=True)