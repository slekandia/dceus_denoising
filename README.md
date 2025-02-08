This is the repository that holds the dynamic contrast-enhanced ultrasound denoising that is submitted to TMI with name
"Speckle Denoising of Dynamic Contrast-enhanced Ultrasound using Low-rank Tensor Decomposition". 


The input is a dictionary with the name data4d that holds data4d['data4d'] which is the regularized DCEUS recording in cartesian domain.
It is a 4D tensor, where the first three represent the cartesian data, and the last the time. The output is the copy of the input and the data4d is the filtered version.
