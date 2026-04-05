### Sequential Convex Programming for Trajectory Optimization with Continuous-Time Satisfaction of Signal Temporal Logic Specifications

This repository contains demonstrative examples of trajectory optimization under continuous-time Signal Temporal Logic (STL) specifications, solved using sequential convex programming.

#### Main Example

**Task:** Eventually visit two dynamic waypoints and reach the end zone while always avoiding dynamic obstacles.

![](Figures/qf_animation.gif)

![](Figures/qf_traj.png)  
![](Figures/qf_states.png)

---

#### Demonstrative Examples

##### Problem 1: Always avoid obstacles

![](Figures/always_stc_aug_all.png)

##### Problem 2: Eventually visit three waypoints

![](Figures/eventually_aug_all.png)

##### Problem 3: Until visiting the charging station, the speed must remain below the threshold

- **Case 1**

![](Figures/until_aug_all.png)

- **Case 2**

![](Figures/CDC/until_traj.png)  
![](Figures/CDC/until_speed.png)  
![](Figures/CDC/until_margin.png)
