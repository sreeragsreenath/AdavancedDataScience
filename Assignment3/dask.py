from subprocess import Popen
commands = [ 'python test.py']
procs = [ Popen(i) for i in commands ]
for p in procs:
   p.wait()