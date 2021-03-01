from PIL import Image
from math import *
import numpy as np
from tqdm import *
import matplotlib.pyplot as plt
import random
from image import *


class sphere:
    def __init__(self, c, r, col, s, ref, t = "not a light"):
        self.center = np.array(c)
        self.radius = np.array(r)
        self.color = np.array(col)
        self.specular = s
        self.reflective = ref
        self.t = t
        
        
class light:
    def __init__(self, t, i, p = (0, 0, 0)):
        self.t = t
        self.int = i  
        self.pos = np.array(p)


def findLights(p, N, v, s):
    i = 0.0
    
    for light in lights:
        if light.t == "ambient":
            i += light.int
        else:
            if light.t == "point":
                direction = np.array(light.pos) - p
                t_max = 1
            
            n_dot_l = np.sum(np.dot(N, direction))
            
            shadow_sphere, shadow_t = ClosestInter(p, direction, 0.001, t_max)
            
            if shadow_sphere == None or shadow_sphere.t == 'light':
              
                if n_dot_l > 0:
                    i += light.int * n_dot_l / (sqrt(np.sum(np.square(np.array(N)))) * sqrt(np.sum(np.square(np.array(direction)))))
                
                if s != -1:
                    R = 2 * N * np.sum(np.dot(N, direction)) - direction
                    r_dot_v =  np.sum(np.dot(R, v))
                    if r_dot_v > 0:
                        i += light.int * pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R))))*(sqrt(np.sum(np.square(np.array(v)))))), s)
               
    return i


def to3D(x, y):
    x1 = x * window_width / width
    y1 = y * window_height / height
    return (x1, -y1, camera_to_window)


def IntersectSphere(st, cam_sc, sphere):
    center = sphere.center
    rad = sphere.radius
    o_c = np.array(st) - np.array(center)

    
    
    a = np.sum(np.dot(np.array(cam_sc), np.array(cam_sc)))
    b = 2 * np.sum(np.dot(o_c, np.array(cam_sc)))
    c = np.sum(np.dot(o_c, o_c)) - rad * rad
    
    dis = b * b - 4 * c * a
   
    if dis < 0:
        return inf, inf    
        
    
    an1 = (-b + sqrt(dis)) / (2 * a)
    an2 = (-b - sqrt(dis)) / (2 * a)
    
    
    return an1, an2


def ClosestInter(O, D, t_min, t_max):
    closest_t = inf
    closest_sphere = None
    for sp in spheres:
        t1, t2 = IntersectSphere(O, D, sp)
        if t1 >= t_min and t1 <= t_max and t1 < closest_t:
            closest_t = t1
            closest_sphere = sp
        if t2 >= t_min and t2 <= t_max and t2 < closest_t:
    
            closest_t = t2
            closest_sphere = sp
        
    return closest_sphere, closest_t


def ReflectRay(R, N):

    return 2 * N * np.sum(np.dot(np.array(N), np.array(R))) - R


def Back(O, point, t_min, t_max, depth):

    closest_sphere, closest = ClosestInter(O, point, t_min, t_max)
    
    if closest_sphere == None:
        return np.array((0, 0, 0))
    if closest_sphere.t == "light":
        return np.array((255, 2555, 255))
    inters = np.array(O) + np.dot(np.array(closest), np.array(point))
    norm = inters - np.array(closest_sphere.center) 
    norm = norm / (sqrt(np.sum(np.square(np.array(norm)))))  
    
    
    local_color = np.array(closest_sphere.color) * findLights(np.array(inters), np.array(norm), -np.array(point), closest_sphere.specular)
    
    r = closest_sphere.reflective
    R = ReflectRay(-np.array(point), norm)
    if depth <= 0 or r <= 0:
        return np.array((local_color))
    if closest_sphere.t != "biggest":
        reflected_color = Back(inters, R, 0.001, inf, depth - 1)

        return np.array(local_color) * (1 - r) + np.array(reflected_color) * r
        

    reflected_color = np.array((0, 0, 0)) 
    eps = 0.01
    
    R1 = ReflectRay(-np.array(point), np.array((0, 0, 0.01)) + norm) 
    R2 = ReflectRay(-np.array(point), np.array((0, 0.03, 0.02)) + norm)
    R3 = ReflectRay(-np.array(point), np.array((0, 0.01, 0.02)) + norm)
    R4 = ReflectRay(-np.array(point), np.array((0.01, 0, 0)) + norm)
    R5 = ReflectRay(-np.array(point), np.array((0, 0.01, 0.01)) + norm)
    R6 = ReflectRay(-np.array(point), np.array((0, 0, 0.03)) + norm)
    R7 = ReflectRay(-np.array(point), np.array((0, 0.01, 0.01)) + norm)
    R8 = ReflectRay(-np.array(point), np.array((0.01, 0.01, 0)) + norm)
   
    r_dot_v = np.sum(np.dot(R1, norm))
    reflected_color = pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R1))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 2) * Back(inters, (np.array(R1)), 0.001, inf, 0)                                                                                     
                                                                                        
    r_dot_v = np.sum(np.dot(R2, norm))
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R2))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 3) * Back(inters, (np.array(R2)), 0.001, inf, 0)
                                                                                        
    r_dot_v = np.sum(np.dot(R3, norm))
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R3))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 3) * Back(inters, (np.array(R3)), 0.001, inf, 0)
    
    r_dot_v = np.sum(np.dot(R4, norm))                                                                                   
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R4))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 2) * Back(inters, (np.array(R4)), 0.001, inf, 0)
    
    r_dot_v = np.sum(np.dot(R5, norm))                                                                                   
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R5))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 2) * Back(inters, (np.array(R5)), 0.001, inf, 0)
    
    r_dot_v = np.sum(np.dot(R6, norm))                                                                                   
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R6))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 3) * Back(inters, (np.array(R6)), 0.001, inf, 0)
    
    r_dot_v = np.sum(np.dot(R7, norm))                                                                                   
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R7))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 3) * Back(inters, (np.array(R7)), 0.001, inf, 0)
    
    r_dot_v = np.sum(np.dot(R8, norm))                                                                                   
    reflected_color += pow(r_dot_v / (sqrt(np.sum(np.square(np.array(R8))))
                            * sqrt(np.sum(np.square(np.array(norm))))), 2) * Back(inters, (np.array(R8)), 0.001, inf, 0) 
    

    return np.array(local_color) * (1 - r) + np.array(reflected_color) * r


im = Image.open(r'base.png')

width, height = im.size

window_width = 10
window_height = 10
camera_to_window = 10

lights = []
recursion_depth = 3
lights.append(light("ambient", 0.4))
lights.append(light("point", 0.3, (-0.2, 3, 0.2)))
lights.append(light("point", 0.05, (0.51, 0.3, 2.1)))
lights.append(light("point", 0.05, (0.29, 0.3, 2.1)))
lights.append(light("point", 0.05, (0.4, 0.41, 2.1)))
lights.append(light("point", 0.05, (0.4, 0.19, 2.1)))
lights.append(light("point", 0.05, (0.4, 0.3, 2.21)))
lights.append(light("point", 0.05, (0.4, 0.3, 1.99)))

spheres = []
spheres.append(sphere((0, -1, 3), 0.6, (120, 100, 100), 500, 0.9))
spheres.append(sphere((0.9, 0, 2.3), 0.6, (40, 255, 15), 300, 0.1))
spheres.append(sphere((-0.2, 0.3, 2), 0.25, (50, 0, 300), 300, 0.2))
spheres.append(sphere((-0.05, -0.06, 1.4), 0.11, (120, 10, 255), 300, 0.2))
spheres.append(sphere((-0.1, 0.3, 2.4), 0.4, (255, 255, 255), 1, 0.00000001, "light"))
spheres.append(sphere((0, -5001, 0), 5000, (200, 200, 200), 100, 0.4, "biggest"))
spheres

start = (0, 0, 0)
img = np.zeros((600, 600, 3))
for x in tqdm(range(-300, 300)):
    x_1 = x + 300
    for y in range(-300, 300):
        y_1 = y + 300
        color = Back(np.array(start), to3D(x, y), 0, inf, recursion_depth)
        im.putpixel((x_1, y_1), tuple(int(c) for c in color))
im.show()
