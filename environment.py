import numpy as np
import random

rows = 10
columns = 10
warehouse = [(0,5),(5,9)]
building = [(1,1),(1,2),(1,3),
            (2,1),(3,1),(7,0),
            (8,2),(8,3),(3,6),
            (3,7),(3,8)]
tunnel_1 = (1,7)
tunnel_2 = (5,1)
red_light = [(6,2),(7,4)]
check_point_1 = (8,4)
check_point_2 = (9,9)
drop_off = (8,9)

actions = ['North','South','East','West','Stop','Pickup','Drop','verify']
    
def movement(state,action):
    man_row,man_col,package_status,check_point_status = state
    original_state = (man_row,man_col,package_status,check_point_status)
    Done = False
    reward = -1

    if action == 0:
        if man_row > 0:
            man_row -= 1
    elif action == 1:
        if man_row < rows-1:
            man_row += 1
    elif action == 2:
        if man_col < columns -1:
            man_col += 1
    elif action == 3:
        if man_col >0:
            man_col-=1
    elif action == 4:
        if (man_row,man_col) in red_light:
            reward = -2
        else:
            reward = -10
    elif action == 5:
        if (man_row,man_col) in warehouse and package_status == 0:
            package_status = 1
        else:
            reward = -10
    elif action == 7:
        if (man_row, man_col) == check_point_1 and package_status == 1 and check_point_status == 0:
            check_point_status = 1
            reward = 1
        elif (man_row, man_col) == check_point_2 and check_point_status == 1:
            check_point_status = 2
            reward = 2
        else:
            reward = -10
    elif action == 6:
        if (man_row,man_col) == drop_off and package_status == 1 and check_point_status ==2:
            reward = 10
            Done = True
        else:
            reward = -10
    if (man_row,man_col) == tunnel_1:
        next_state = (5,0,package_status,check_point_status)
        return (next_state,reward,Done)
    elif (man_row,man_col) == tunnel_2:
        next_state = (1,8,package_status,check_point_status)
        return (next_state,reward,Done)
    elif (man_row,man_col) in building:
        return (original_state,reward,Done)
    next_state = (man_row,man_col,package_status,check_point_status)
    return next_state,reward,Done

def get_valid_start():
    while True:
        r = random.randint(0, rows-1)
        c = random.randint(0, columns-1)
        if (r, c) not in building:
            return r, c
        
