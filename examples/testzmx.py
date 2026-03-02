import matplotlib.pyplot as plt
import numpy as np

from optiland import analysis
from optiland.fileio import load_zemax_file
# link to the .zmx file on Thorlabs website
url = "https://www.thorlabs.com/_sd.cfm?fileName=20565-S03.zmx&partNumber=MAP051950-A"

lens = load_zemax_file(url)
lens.draw()
lens.info()
spot = analysis.SpotDiagram(lens)
spot.view()


# we will shift the object plane by ±3.0 mm from the nominal location
dz = np.linspace(-3.0, 3.0, 64)

# thickness between the object surface and the first lens surface
thickness = dz + 16.3412  # nominal location = 16.3412 mm

# set the wavelength and field indices
wavelength_idx = 1
field_idx = 0

# initialize variables
rms_spot_radius = []

for z in thickness:
    # change thickness on the first surface
    lens.set_thickness(value=z, surface_number=0)

    # move image plane to maintain focus
    lens.image_solve()

    # generate spot diagram data
    spot = analysis.SpotDiagram(lens)

    # calculate RMS spot radius
    rms_spot_radius.append(spot.rms_spot_radius()[field_idx][wavelength_idx])
plt.plot(dz, rms_spot_radius)
plt.xlabel("Object plane shift (mm)")
plt.ylabel("RMS Spot Radius (mm)")
plt.title("RMS Spot Radius vs. Object Plane Shift")
plt.grid()
plt.show()