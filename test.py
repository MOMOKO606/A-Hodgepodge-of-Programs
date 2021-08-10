import matplotlib.pyplot as plt
import random
import math

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
Output: the minimum distance and its corresponding IDs of point.
"""
def closest_pair_naive( P ):
    #  The number of points.
    n = len( P )

    #  Initialize the minimum distance of pair points.
    min_d = float("inf")
    min_i = 0
    min_j = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            d = dist( P[i], P[j] )
            if d < min_d:
                min_d = d
                min_i = P[i].ID
                min_j = P[j].ID
    return min_d, min_i, min_j


"""
Get the Point(s) whose x coordinate satisfies [ x_left <= ... x ... <= x_right ] in X.
X is a list of Point(s) sorted by its x coordinates.
Input @para: list X of Pair(s), the range [ x_left, ..., x_right ].
Output: the list of Point(s) such that the x coordinate of each point satisfies [ x_left <= ... x ... <= x_right ].
"""
def points_list( X, x_left, x_right ):

    """
    Get the index of the first element whose x coordinate is larger than x_left in X, O(lgn).
    Input @para: list of Pair(s) X[low, ..., high], x_left.
    Output: the index.
    """
    def left_most_index( X, low, high, x_left ):

        #  Base case 1, cannot find the left_most index.
        if low > high:
            return
        #  Base case 2, the edge point.
        if low == high:
            if X[low].x >= x_left:
                return low
            else:
                return

        mid = ( low + high ) // 2
        #  Base case 3.
        if X[mid].x >= x_left and X[ mid - 1 ].x < x_left:
            return mid

        if X[mid].x >= x_left:
            return left_most_index( X, low, mid , x_left )
        else:
            return left_most_index(X, mid + 1, high, x_left)


    """
    Get the index of the first element whose x coordinate is smaller than x_right in X, O(lgn).
    Input @para: list of Pair(s) X[low, ..., high], x_right.
    Output: the index.
    """
    def right_most_index(X, low, high, x_right):

        #  Base case 1, cannot find the right_most index.
        if low > high:
            return
        #  Base case 2, the edge point.
        if low == high:
            if X[low].x <= x_right:
                return low
            else:
                return

        mid = (low + high) // 2
        #  Base case 3.
        if X[mid].x <= x_right and X[mid + 1].x > x_right:
            return mid

        if X[mid].x >= x_right:
            return right_most_index(X, low, mid, x_right)
        else:
            return right_most_index(X, mid + 1, high, x_right)

    #  Store the left most index of X and the right most index of X.
    res = [ left_most_index( X, 0, len(X) - 1, x_left ), right_most_index(X, 0, len(X) - 1, x_right) ]
    return res



"""
Recursively find the closest pair of points.
"""
def closest_pair_recur( X, x_left, x_right ):

    #  pl_index is the points list in the region from x_left to x_right coordinate in x axis.
    #  elements in pl are sorted by their x coordinate.
    pl_index = points_list( X, x_left, x_right )
    #  Base case 1: Only one point.
    if pl_index[0] == pl_index[1]:
        return float("inf"), float("inf"), float("inf")

    #  Transfer indices into Point and return them.
    pl = X[ pl_index[0]: pl_index[1] + 1  :1]

    #  Base case 2: Only two points.
    if len( pl ) == 2:
        return dist( pl[0], pl[1] ), pl[0].ID, pl[1].ID
    #  Base case 2: Only three points, we use the brute force algorithm.
    if len( pl ) == 3:
        return closest_pair_naive( pl )

    x_mid = ( x_left + x_right ) / 2
    mini_left, left_ID1, left_ID2 = closest_pair_recur( X, x_left, x_mid )
    mini_right, right_ID1, right_ID2 = closest_pair_recur( X, x_mid, x_right )

    d = min( mini_left, mini_right )
    strip_index = points_list(X, x_mid - d, x_mid + d )
    strip = X[ strip_index[0]: strip_index[1] + 1  :1]

    #  Sort in place.
    strip.sort( key = lambda point: point.y )

    mini_cross = float("inf")
    cross_ID1 = None
    cross_ID2 = None


    for i in range( len(strip) ):
        for j in range( 1, min(8, len(strip) - i) ):
            tmp = dist(strip[i], strip[i + j])
            if tmp <= mini_cross:
                mini_cross = tmp
                cross_ID1 = strip[i].ID
                cross_ID2 = strip[i + j].ID

    if mini_cross < d:
        return mini_cross, cross_ID1, cross_ID2
    else:
        if mini_left < mini_right:
            return mini_left, left_ID1, left_ID2
        else:
            return mini_right, right_ID1, right_ID2


def closest_pair_aux( P ):
    #  We cannot use P.sort( key=lambda point: point.x ) to assign the sorted list to X.
    #  P.sort( key=lambda point: point.x ) is the in place way.
    X = sorted( P, key=lambda point: point.x )
    return closest_pair_recur(X, X[0].x, X[len(X) - 1].x)



#  Drive code.
if __name__ == "__main__":

    #  Test data 1:
    P1 = [Point(2, 3), Point(12, 30), Point(40, 50), Point(5, 1), Point(12, 10), Point(3, 4)]



    #  Test data 2:
    #  Created 30 points in Class Point form and stored in list P.
    P = []
    for i in range(30):
        pt = Point( random.randint(0, 50), random.randint(0, 50) )
        P.append( pt )
        #  Show each coordinate of the 30 points.
        # print(pt)



    # #  Show the scatter plot of the test data.
    # plt.scatter( [P[i].x for i in range(len(P))], [P[i].y for i in range(len(P))], marker = 'o' )
    #
    # #  Show the point ID above each point.
    # for i in range( len(P) ):
    #     #  xy means the coordinate of point, xytext means the coordinate of the ID.
    #     plt.annotate( P[i].ID, xy = ( P[i].x, P[i].y ), xytext = ( P[i].x, P[i].y + 0.2 ))
    #
    # plt.show()

    #  Show the result computed by the brute force algorithm.
    print(closest_pair_naive(P))
    print(closest_pair_aux(P))


