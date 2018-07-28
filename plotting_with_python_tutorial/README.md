# Plotting with Python Tutorial

Being able to plot data is very important in order to communicate results.

From Alberto Cairo book "The Functional Art" he explains some of the choices we 
must do when dealing with visualization.
We can talk about Visualization Wheel Dimensions, where we have some tradeoffs to consider:

* Abstraction vs Figuration
* Functionality vs Decoration
* Density vs Lightness
* Multidimensional vs Unidimensional
* Originality vs Familiarity
* Novelty vs Redundancy

These properties can be represented by a graphic wheel, and Cairo analyzed and showed how
engineering and science plots are more towards abstraction, multidimensionality and other 
properties useful for sciences, while artists and journalists prefer less abstract representations.

Another important person in the field of information visualization is Edward Tufte,
his work "The Visual Display of Quantitative Information" he describes many
concepts related to data representation.

Tufte introduces two important graphical heuristics:

*   the **data - ink ratio**
    the data - ink is defined as the non-erasable core of a graphic,
    the non redundant ink arranged in response to variation in the numbers represented,
    Tufte defined the dataink ratio as the amount of dataink divided by the total ink 
    required to print the graphic

*   the **chart junk**








## Matplotlib Architecture
Matplotlib can be described as a three layer architecture:

*   Backend Layer:
    Matplotlib can have different backends, each with different capabilities,
    for example when we use matplotlib innside a jupyter notebook, we use a specific 
    backend called *inline*
*   Artist Layer:
    Which contains containers such as Figure, Subplot, and Axes, it also contains
    primitives such as Line2D, Rectangle, Collections and many other elements
*   Scripting Layer:
    This makes the difference between who can use matplotlib more efficiently and who
    does not. The scripting layer we will use is called *pyplot*

Generally graphing components can be divided into declarative (e.g., HTML or 3D.js) or
procedural (which just gives a sequence of commands), matplotlib is a procedural graphing
system.

```python
# This is the importing boilerplate
import matplotlib as mpl
import matplotlib.pyplot as plt
```

Let's start with basic plots:
```python

plt.plot(3, 2) # This makes a plot with a center in 3 on x axis and 2 on y axis
# We will not see any point, since we did not specify how to render our plot

plt.plot(3, 2, '.') # this will plot a point in the middle of the plot


