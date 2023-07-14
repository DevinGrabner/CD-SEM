Required - Environment with Python version 3.10.12

In a Windows cmd or terminal, run:

      git clone https://github.com/DevinGrabner/CD-SEM
      semWin.bat

The semWin.bat file will create an environment for you with the needed version of Python and the necessary packages.
If you would rather do this manually, see the "Optional" section at the end of ReadMe.md.


  
# CD-SEM
Line Roughness Calculation for DSA images.

This project is funded by the CHiPPS EFRC at Berkeley National Laboratory.

The goal is to port Mathematica code originally written by Ricardo Ruiz into an open-source Python platform

## Users
  Use the 'RunThis_CD_SEM_Analysis.ipynb' file to run the analysis of your SEM (*.tif) image.






# Optional
If you need to create a new environment, do the following in an Anaconda PowerShell:

      conda create --name myenv python=3.10.12    # This creates your new environment and installs the specified version of Python
      conda activate myenv                        # This activates your new environment so you can use it
      python --version                            # Verify that the correct version of Python is installed
      
  If you want to install a different version of Python in an existing environment, do the following in an Anaconda PowerShell:

      conda activate your_environment_name        # This activates the environment you want to use
      conda install python=3.10.12                # This installs the specified version of Python in that environment
      python --version                            # Verify that the correct version of Python is installed
