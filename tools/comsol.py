"""Compile native sources, then optionally run one class using an explicit COMSOL installation."""
from pathlib import Path
import argparse
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comsol-home', type=Path, default=os.environ.get('COMSOL_HOME'))
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('entry', help='Native class name, for example DoeAll or Build3D')
    ap.add_argument('arguments', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.comsol_home is None:
        ap.error('Set COMSOL_HOME or pass --comsol-home pointing to Multiphysics')
    home = Path(args.comsol_home).resolve()
    if os.name == 'nt':
        java_bin = home/'java/win64/jre/bin'
        suffix = '.exe'
    else:
        candidates = list((home/'java').glob('*/jre/bin'))
        if len(candidates) != 1:
            ap.error('Could not identify the COMSOL bundled Java runtime')
        java_bin, suffix = candidates[0], ''
    folders = [ROOT/'work', ROOT/'research/java', ROOT/'outputs/verify3d']
    matches = [p for folder in folders for p in folder.glob(args.entry+'.java')]
    if len(matches) != 1:
        ap.error('Class name must identify exactly one released source file')
    classes = ROOT/'.build/java'
    classes.mkdir(parents=True,exist_ok=True)
    classpath = os.pathsep.join([str(classes),str(home/'java/*'),str(home/'plugins/*')])
    sourcepath = os.pathsep.join(map(str,folders))
    subprocess.run([str(java_bin/('javac'+suffix)),'-encoding','UTF-8','-cp',classpath,'-sourcepath',sourcepath,'-d',str(classes),str(matches[0]),str(ROOT/'work/ProjectPaths.java')],cwd=ROOT,check=True)
    if not args.compile_only:
        env = dict(os.environ,SOFC_PROJECT_ROOT=str(ROOT))
        for folder in ['work/doe/steady','work/doe/transient','work/doe_b4/transient','work/doe_cont/transient','outputs/verify3d/out']:
            (ROOT/folder).mkdir(parents=True,exist_ok=True)
        subprocess.run([str(java_bin/('java'+suffix)),'-Dfile.encoding=UTF-8','-Xmx4g','-cp',classpath,args.entry,*args.arguments],cwd=ROOT,env=env,check=True)

if __name__ == '__main__':
    main()
