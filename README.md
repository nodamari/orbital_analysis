## About orbital_analysis
This program creates a 3D plot of a numerically calculated satellite orbit and compares the result of the simulation to the actual measured orbit using two-line elements (TLEs) queried from Space-Track.org. Currently, the numerical simulation takes into account gravitational force from Earth and the J2 perterbation. Incorporation of other perturbations is WIP.

Running space_track.py executes the program.

## 3 Requirements for proper program execution
**1. Account for Space-Track.org:**

Please visit the [Space-Track website](https://www.space-track.org) and create an account. Sometimes, the account confirmation email ends up in the spam box.

**2. A .txt file to store your username and password:**

After creating and confirming the Space-Track account, copy the username and password onto a .txt file named userpass.txt under the same directory level as space_track.py and tle_obj.py. Paste the username into the first line of the file and the password in the second line. There should only be two lines in userpass.txt


**3. Required python packages:** 
- datetime
- spacetrack
- numpy
- matplotlib
- scipy
- mlp_toolkits
- math


