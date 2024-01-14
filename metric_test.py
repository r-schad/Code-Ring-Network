import numpy as np
import matplotlib.pyplot as plt

def curvature_subscore(curv, desired_curv, curv_sd):
    curv_score = 1 - np.exp(-1 * np.square(curv - desired_curv) / (2 * (np.square(curv_sd))))
    return curv_score

def intersec_subscore(num_intersecs, intersec_growth):
    is_score = np.exp(intersec_growth * num_intersecs) - 1
    return is_score

def metric(curv, num_intersecs, desired_curv, curv_sd, intersec_growth):
    curv_score = curvature_subscore(curv, desired_curv, curv_sd)
    is_score = intersec_subscore(num_intersecs, intersec_growth)
    return curv_score + is_score


curvs = np.linspace(0, 10, 100)
intersec_counts = np.linspace(0, 10, 100)

desired_curv=2
curv_sd = 0.6
intersec_growth = 0.3

X, Y = np.meshgrid(curvs, intersec_counts)
scores = metric(X, Y, desired_curv=desired_curv, curv_sd=curv_sd, intersec_growth=intersec_growth)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_title(f'Intersection/Curvature Metric Penalty with\nDesired Curv={desired_curv}, Curv St.Dev.={curv_sd}, Intersec Growth Rate={intersec_growth}')

ax.set_xlabel('Average Curvature')
ax.set_ylabel('Number of Intersections')
ax.set_zlabel('Metric Penalty')

ax.plot_surface(X, Y, scores, rstride=1, cstride=1, cmap='viridis', alpha=0.8)

plt.show()
pass