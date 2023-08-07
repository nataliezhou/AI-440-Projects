# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    queue = [start]
    visited = set()
    parent = {}
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        if (curr == maze.waypoints[0]):
            break
        for neighbor in maze.neighbors(curr[0], curr[1]):
            if neighbor not in visited:
                queue.append(neighbor)
                parent[neighbor] = curr
    path = [] 
    while curr != start:
        path.append(curr)
        curr = parent[curr]
    path.append(start)
    path.reverse()
    return path

def astar_single(maze):
    """
    Runs A star search for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    visited = set()
    unvisited = []
    unvisited.append([start])
    while unvisited:
        min_path = unvisited.pop(0)
        curr = min_path[-1]    
        if curr not in visited:
            visited.add(curr)
            if curr == maze.waypoints[0]:
                return min_path
            for neighbor in maze.neighbors(curr[0], curr[1]):
                path = list(min_path)
                path.append(neighbor)
                unvisited.append(path)
        unvisited.sort(key=lambda x: len(
            x) + abs(x[-1][0] - maze.waypoints[0][0]) + abs(x[-1][1] - maze.waypoints[0][1]))
    return []


# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    