import matplotlib.pyplot as plt
import random
import math
import pickle

class Point():

    #  The total number of points that have been created.
    #  A property of the class Point.
    IDcounts = 0

    def __init__( self, x, y ):
        #  x coordinate of the point.
        self.x = x
        #  y coordinate of the point.
        self.y = y

        #  Update the number of points when a new instance is created.
        #  The ID of the point equals to the number of points when this point is created.
        self.ID = Point.IDcounts
        Point.IDcounts += 1


    """
    The print function of the instance of Point class.
    """
    def __str__( self ):
        return ( str([self.x, self.y, self.ID]) )


"""
Euclidean distance between point p and q in Point class form.
Input: Point p, Point q.
Output: Euclidean distance between p and q.
"""
def dist( p, q ):
    #  the x & y coordinate of Point p.
    x1 = p.x
    y1 = p.y
    #  the x & y coordinate of Point q.
    x2 = q.x
    y2 = q.y

    return math.sqrt( (x1 - x2) ** 2 + (y1 - y2) ** 2 )


"""
The brute force algorithm for computing the closest pair of points, O(n^2).
Input: list P of Points.
Output: the minimum distance and its corresponding IDs of the pair of points.
"""
def closest_pair_naive( P ):
    #  The number of points.
    n = len( P )

    #  Initialize the minimum distance of pair points.
    min_d = float("inf")
    min_p = 0
    min_q = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            d = dist( P[i], P[j] )
            if d < min_d:
                min_d = d
                min_p = P[i].ID
                min_q = P[j].ID
    return min_d, min_p, min_q


"""
The "combine" function in the recursive closest pair.
Find the closest pair from bottom-top order in the region whose [ mid line - d <= x coordinate <= mid line + d ].
O(n) 

Input @para: 
X = sorted list P of Point(s) by x coordinates.
Y = sorted list P of Point(s) by y coordinates.
Indices: x_low, x_mid, x_high, the X point(s) are divided as X[x_low, ...., x_mid | x_mid + 1, ..., x_high]
d is the distance of closest pair in the left and right regions.

Output: the minimum distance and its corresponding IDs of the closest pair in the crossing region.
"""
def cross_closest_pair( X, Y, x_low, x_mid, x_high, d ):
    #  mid line's x coordinate.
    midline = (X[ x_mid ].x + X[ x_mid + 1 ].x) / 2

    #  Get the Point(s) whose x coordinates is in the [mid line - d, ... mid line + d] region.
    Rl = []
    Rr = []
    for i in range(x_mid, x_low - 1, -1):
        if X[i].x  >= midline - d:
            #  Store the Point(s) in the left region.
            Rl.append(X[i])
            Rl.reverse()
    for j in range(x_mid + 1, x_high + 1, 1):
        if X[j].x <= midline + d:
            #  Store the Point(s) in the right region.
            Rr.append(X[j])
    R = Rl + Rr

    #  Compute the crossing closest pair of Point(s).
    #  Initialize the return values.
    mini_cross = float("inf")
    cross_ID1 = None
    cross_ID2 = None

    #  Get the y start and y end indices of Point(s) in the region in sorted order by y axis.
    indices = []
    #  Make sure we only check the Point(s) in the strip region.
    Y_tmp = [ Y[i] for i in range(len(Y)) if Y[i] in R ]
    for i in range(len(Y_tmp)):
        if Y_tmp[i].x >= R[0].x and Y_tmp[i].x <= R[-1].x:
            indices.append(i)

    #  No Point(s) or only one Point in the region.
    if len(indices) < 2:
        return mini_cross, cross_ID1, cross_ID2

    y_low = indices[0]
    y_high = indices[-1]
    for i in range(y_low, y_high + 1):
        for j in range(1, min(8, y_high + 1 - i)):
            tmp = dist(Y[i], Y[i + j])
            if tmp < mini_cross:
                mini_cross = tmp
                cross_ID1 = Y[i].ID
                cross_ID2 = Y[i + j].ID

    return mini_cross, cross_ID1, cross_ID2


"""
Recursively find the closest pair of points by divide-and-conquer paradigm.
Input @para: 
X = sorted list P of Point(s) by x coordinates.
Y = sorted list P of Point(s) by y coordinates.
Indices: x_low, x_high such that X[x_low, ..., x_high]

Output: the minimum distance and its corresponding IDs of the closest pair of Point(s).
"""
def closest_pair_recur( X, Y, x_low, x_high ):

    #  Base case 1: Only two points.
    if x_high - x_low + 1  == 2:
        return dist( X[x_low], X[x_high] ), X[x_low].ID, X[x_high].ID
    #  Base case 2: Only three points, we use the brute force algorithm.
    if x_high - x_low + 1  == 3:
        return closest_pair_naive( X[x_low : x_high + 1 : 1] )

    #  Get the distance of closest pair in the left, right and crossing regions.
    #  Divide.
    x_mid = ( x_low + x_high ) // 2

    mini_left, left_ID1, left_ID2 = closest_pair_recur( X, Y, x_low, x_mid )
    mini_right, right_ID1, right_ID2 = closest_pair_recur( X, Y, x_mid + 1, x_high )

    #  Conquer.
    d = min( mini_left, mini_right )
    mini_cross, cross_ID1, cross_ID2 = cross_closest_pair( X, Y, x_low, x_mid, x_high, d )

    #  Compare and find the minimum among the distance of closest pair in the left, right and crossing regions.
    if mini_cross < d:
        return mini_cross, cross_ID1, cross_ID2
    else:
        if mini_left < mini_right:
            return mini_left, left_ID1, left_ID2
        else:
            return mini_right, right_ID1, right_ID2


"""
The helper program of the recursive closest pair of points.
Input @para: list P of class Point(s)
Output: the minimum distance and its corresponding IDs of the closest pair of Point(s).
"""
def closest_pair_aux( P ):

    #  We cannot use P.sort( key=lambda point: point.x ) to assign the sorted list to X.
    #  P.sort( key=lambda point: point.x ) is the in place way.
    X = sorted( P, key = lambda point: point.x )
    Y = sorted( P, key = lambda point: point.y )

    return closest_pair_recur( X, Y, 0, len(X) - 1 )



#  Drive code.
if __name__ == "__main__":

    #  Test data 1:
    P0 = [Point(2, 3), Point(12, 30), Point(40, 50), Point(5, 1), Point(12, 10), Point(3, 4)]

    #  Create random test data :
    #  Created points in Class Point form and stored in list P.
    P = []
    for i in range(50):
        pt = Point( random.randint(0, 80), random.randint(0, 80) )
        P.append( pt )
        #  Show each coordinate of the 30 points.
        # print(pt)
    #  Store P into "points_data.pkl".
    pdata = open( "points_data.pkl", "wb")
    pickle.dump( P, pdata)
    pdata.close()

    # #  Load test data 2:
    # file = open("points_data.pkl", "rb")
    # P = pickle.load( file )
    # file.close()

    # Show the scatter plot of the test data.
    plt.scatter( [P[i].x for i in range(len(P))], [P[i].y for i in range(len(P))], marker = 'o' )

    #  Show the point ID above each point.
    for i in range( len(P) ):
        #  xy means the coordinate of point, xytext means the coordinate of the ID.
        plt.annotate( P[i].ID, xy = ( P[i].x, P[i].y ), xytext = ( P[i].x, P[i].y + 0.2 ))

    plt.show()

    #  Show the result computed by the brute force algorithm.
    print("The distance of the closest pair of Points and their pointID are", closest_pair_naive(P), "respectively.")

    print("The distance of the closest pair of Points and their pointID are", closest_pair_aux(P), "respectively.")


