## About orbital_analysis
This program creates a 3D plot of a numerically calculated satellite orbit and compares the result of the simulation to the actual measured orbit using two-line elements (TLEs) queried from Space-Track.org. Currently, the numerical simulation takes into account gravitational force from Earth and the J2 perterbation. Incorporation of other perturbations is WIP.


## How to run
1. Create an account with [Space-Track.org](https://www.space-track.org) if you don't already have an account
2. In the repo top directory, create a file called `userpass.txt` and write the username in the first line and password in the second line
3. Query TLEs from the Space Track API by running `query_tles.py`
   
     a. There is a default NORAD ID in this script, but you can change this to get TLEs of different space objects
4. Run `space_track.py`.

### Expected Output
This will create several 2D plots of orbital elementsand other parameters saved to the `plots` folder as well as a 3D plot of the propagated orbit



## Required python packages
- datetime
- spiceypy
- spacetrack
- numpy
- matplotlib
- scipy
- mlp_toolkits
- math
- os
- pickle
  


