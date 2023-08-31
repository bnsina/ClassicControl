from . import control

# multiprocessing map only works with imported modules (bug?), 
# so this method had to be put in a seperate file...
def runner(namespace):
    job = control.controller(namespace)
    job.run()
