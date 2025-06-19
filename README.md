# usal-diaphragm package


## Aims

The purpose of this package is to process 4D (3D volume + time) ultrasound data to track movement of the diaphragm, parameterise the movement and distil this into one or two parameters that quantify diaphragm excursion.


# Usage

The package comes as a single executable (launch.exe) which is highly parameterised. Call

    diaphragm.exe --help

for a full list of parameters.

To simplify matters, the package comes with several batch files that supply reasonable parameter values. You can drag & drop a .vol file onto the batch files in order to run the code on that file.


## show_ultrasound_peaks.bat

The first script you should run in your pipeline is show_ultrasound_peaks.bat. This will compute points of interest within the volume (usually strong bands such as along the diaphragm) and superimpose them (in pink) on slices through the ultrasound volume. This is a useful diagnostic step to check that points along the diaphragm are being detected.

If points are not being detected, you may play around with the peak finder parameters such as the method used and the threshold that determines whether to keep or discard candidate peaks.

It is unlikely that the software will be able to pick lots of points on the diaphragm without picking up a lot of peaks in the surrounding tissue, but there is some robustness to noise in later stages.


## show_ultrasound_plane.bat

When you are happy that sufficient points on the diaphragm are being detected, show_ultrasound_plane.bat will apply the same code but with the additional (and usually slower) step of looking for a plane through the volume that roughly lines up with the diaphragm. (The probe captures samples along rays that fan out from the centre of the probe, but stores the data as a rectangular volume. The result is that spherical surfaces in cartesian space are mapped approximately to planes in the rectangular space, so fitting a plane to the untransformed data is reasonable.)

At each frame of the sequence, the "best" fitting plane to the detected peaks is overlaid in cyan, so you should be able to spot quite easily when the plane fitting has succeeded and when it has failed.


## show_surfaces.bat

When the plane fit is working well, show_surfaces.bat will give you an animated view of the detected peaks in 3D along with an approximation to the surface.

At this point, you can choose to view the peaks either in the rectangular coordinate frame (where the fitted surface will be a plane) or in a cartesian coordinate frame (where the fitted surface will look like part of a sphere).

You can also choose to fit a spherical surface explicitly to the points that are thought to lie on the surface of the diaphragm. This gives a different parametersisation of the surface in terms of a centre and radius in cartesian space.

In the right hand panel, you will see a visualization of the surface parameters (either of the 3D plane or the sphere) and their progression over time. This can sometimes be helpful in seeing the periodic motion of parameters that follows the breathing cycle. (It can also give you an idea of how much noise there is in the estimated parameters).


## output_stats.bat

Finally, to process the results for yourself you can use output_stats.bat to save surface points and parameters to .csv tables that you can read in to Excel (or similar), Matlab, etc.


# Using a Config file

Because some settings with work well on one sequence while others work better on other sequences, you can run the program on a text config file that stores the name and path of the .vol file along with the settings you want to use on it.

To create a new config file, run:

    diaphragm.exe --config new <config_filename>.yaml

and a new config file will be created with default parameter settings.

Change the *input_vol* setting to the file you want to process, and change the individual settings to values that perform best on that file. (Leaving *input_root* as '' assumes that the path to the .vol file is relative to that of the config file.)

For information on what settings do, run

    diaphragm.exe --help

**Note that config options overwrite anything supplied at the command line.**


# Processing Pipeline

First, a note of caution: many of these fitting methods use random sampling so the outputs will vary from run to run. For consistent outputs over time, you should set the *randseed* to a positive integer value before processing.

The processing works via the following stages.

## Peak detection

Points of interest are detected either as maxima in the intensity of the sampled data or as maxima in the response to filters applied to the data (e.g., a gradient filter to pick up edges).

Filters can be convolved with the data in either 1D (along a ray) or in 2D (convolution with a whole slice of the volume).

This process results in a discrete set of peaks for every 3D volume in the sequence.

## Surface Fitting (rectangular)

For every volume in the sequence, the software then looks for a plane that passes close to a large number of points in the set of detected peaks. This works via a process known as Random Sample Consensus (RanSaC) which randomly chooses a minimal set of points out of all candidates, computes the equation of the surface they uniquely define and then counts the number of other candidates that lie within some distance from that surface.

This random fitting is repeated a fixed number of times, and the surface with the highest number of nearby points (the greatest "consensus") is returned as the best fit.

One advantage of this technique is its robustness to outliers (points far away from the true surface that spoil the estimation whenever they are used to estimate parameters). By only using a minimal subset of the points at every iteration, it is likely that one or more iterations will include only points that are genuinely on the surface, leading to a good estimate of parameters and large consensus.

In our case, we use sets of three points since this many uniquely define a plane (as long as the points do not lie along a straight line). We also apply some thresholds to parameter values in order to exclude surfaces whose orientation is significantly different from that of the diaphragm.

Because there can, at times, be lots of peaks along spurious edges (e.g., at the boundary of other tissues), we choose to keep the best K surfaces and try to resolve the conflict in the next step.

## Parameter Path Estimation

Having estimated K potential surfaces (with a corresponding score) for every volume in the sequence, we then attempt to find a "path" through parameter space over the whole sequence that contains surfaces with large consensus but that also form a coherent sequence.

To do this, we start with the surface that has the highest consensus over all candidates in all frames, assuming that this is indeed a good fit to the diaphragm. We then work forwards, looking for a surface in the following frame that has similar parameters *and* a large consensus, repeating this until we reach the end of the sequence. Going back to the best of all candidates, we do the same in reverse until we reach the beginning of the sequence.

The balance between large consensus and agreement with neighbouring volumes is determined by the *dist_weight* argument that penalises surfaces with significantly different parameters.

(Note that this should not entail the risk of the parameters staying the same over the whole sequence because we're always choosing from candidates that have been suggested by the data. I.e., it is hard to oversmooth the sequence.)

## Surface Fitting (cartesian)

With a reasonable surface fit at every volume in the sequence, we can reparameterise from planar parameters in the rectangular coordinate frame to spherical parameters in the cartesian coordinate frame.

Specifically, for every estimated surface in the sequence we first transform points on the plane (in the rectangular space) into points on a curved surface (in the cartesian space). We then pick pairs of points at random, compute the vector from one to the other and the plane that bisects this vector. By repeating this for many pairs of points, we get a set of planes that should (if the points are on the surface of a sphere) intersect at the centre of the sphere. After computing this intersection, we can estimate the radius of the sphere by computing the average distance of all points from the estimated centre.

This then gives us an alternative parameterisation with which to describe the excursion of the diaphragm.

## Data Output

Setting *output_csv* to True (either in the config file or by passing the "--output_csv" argument to diaphragm.exe) saves this sequence of surface parameters to a .csv file which you can process or visualise further in order to analyse the diaphragm movement.

## Visualization

It is probably wise to set the *cartesian* argument when displaying the ultrasound on the screen so that it appears in a form that is familiar to sonographers.

