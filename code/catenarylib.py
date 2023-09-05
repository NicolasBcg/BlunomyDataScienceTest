from scipy.optimize import fmin
from math import *
def error_on_catenary(x,array,x0,y0):#return the global error of the catenary function
    error=0
    for point in array:
        error+= minimum_distance_point_catanary(point,x0,y0,x)[1]
    return error
def find_x0_y0(array,spanReduction=1000):#function to find a suitable y0 !!! can be upgraded for more precision !!!
    xmin,xmax=0,0
    for point in array:#searching for xspan
        if point[0]>xmax:
            xmax=point[0]
        elif point[0]<xmin:
            xmin=point[0]
    span=xmax-xmin
    xspan_for_y=span/spanReduction
    ysum=0
    found=0
    for point in array:
        if point[0] > 0 - xspan_for_y  and point[0] < 0+xspan_for_y:
            ysum+=point[1]
            found+=1
    if found<len(array)/50:
        return find_x0_y0(array,spanReduction/2)
    elif found>len(array)/25:
        return find_x0_y0(array,spanReduction*1.5)
    else:
        return 0,ysum/found
    
def catenary_dist_function(x,point,x0,y0,c):#gives the distance between a plane and a point from catenary
    return abs(sqrt(pow(point[0]-x,2)+ pow( point[1]-(y0+c*cosh((x-x0)/c)-c),2) ))

def minimum_distance_point_catanary(point,x0,y0,c):#gives the minimum distance between a plane and a point from catenary
    return fmin(catenary_dist_function, 0,args=(point,x0,y0,c),full_output=1,disp=0)

def findCatenary(array):
        x0,y0=find_x0_y0(array)
        return fmin(error_on_catenary, 1,args=(array,x0,y0),full_output=1,disp=0)[0][0],x0,y0