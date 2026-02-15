setAutoThreshold("Default dark no-reset");
//run("Threshold...");
setOption("BlackBackground", true);
run("Convert to Mask", "background=Dark calculate black");
run("Set Measurements...", "integrated stack limit redirect=z_stack.tif decimal=3");
run("Analyze Particles...", "display clear stack");
// Get the name of the current image (or the unique identifier for the batch file)
name = getTitle();
saveDir = "#path_to_save";
saveAs("Results", saveDir + name + "_Results.csv");

//in line 5 replace path for redirect z-stack with your z_stack name.
//in line 10 replace path to your output save.